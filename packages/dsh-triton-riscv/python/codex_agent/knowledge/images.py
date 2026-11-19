"""Local, bounded OCR of standalone images; no diagram interpretation or model API."""
import base64
import csv
import io
import os
import re
import shutil
import subprocess
import warnings

from .parsing import ParseError, Sections

IMAGE_FORMATS = {"png", "jpg", "jpeg"}
PARSER_VERSION = "image-tesseract-v1"
MAX_PIXELS = 12_000_000
MAX_EDGE = 16000
OCR_TIMEOUT = 25
MAX_PREVIEW = 8 * 1024 * 1024


def decode_image(data, fmt):
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != ("PNG" if fmt == "png" else "JPEG"):
                    raise ParseError("Image contents do not match the filename format")
                if getattr(image, "n_frames", 1) != 1:
                    raise ParseError("Animated images are not supported; upload a single frame")
                w, h = image.size
                if w * h > MAX_PIXELS or max(w, h) > MAX_EDGE:
                    raise ParseError("Image exceeds 12 megapixels or 16000 pixels per side; resize it first")
                image.verify()
            with Image.open(io.BytesIO(data)) as original:
                original.load()
                upright = ImageOps.exif_transpose(original).convert("RGBA")
                result = Image.new("RGB", upright.size, "white")
                result.paste(upright, mask=upright.getchannel("A"))
                return result
    except ParseError:
        raise
    except Exception as error:
        raise ParseError("Invalid, truncated or oversized image; export a new PNG/JPEG") from error


def ocr_environment():
    # No credentials, user-supplied commands, external URLs or shell evaluation.
    return {"PATH": os.defpath + ":/opt/homebrew/bin:/usr/local/bin", "LANG": "C.UTF-8",
            "OMP_THREAD_LIMIT": "1"}


def recognize(png):
    env = ocr_environment()
    binary = shutil.which("tesseract", path=env["PATH"])
    if not binary:
        raise ParseError("Local OCR unavailable: install Tesseract 5 and chi_sim + eng language data, then upload again")
    languages = subprocess.run([binary, "--list-langs"], env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5, check=False)
    available = set(languages.stdout.decode("utf-8", errors="replace").splitlines())
    if languages.returncode or not {"chi_sim", "eng"}.issubset(available):
        raise ParseError("Local OCR language data missing: install chi_sim and eng, then upload again")
    proc = subprocess.run([binary, "stdin", "stdout", "-l", "chi_sim+eng", "--oem", "1",
        "--psm", "3", "tsv"], input=png, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, timeout=OCR_TIMEOUT, check=False)
    if proc.returncode or len(proc.stdout) > 8 * 1024 * 1024:
        raise ParseError("OCR failed or exceeded output limits; upload a clearer, smaller image")
    return proc.stdout.decode("utf-8", errors="strict")


def ocr_sections(tsv, width, height):
    """Preserve engine paragraph order, line breaks, and oriented-image coordinates."""
    groups = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        x, y, w, h = (int(row[k]) for k in ("left", "top", "width", "height"))
        confidence = float(row["conf"])
        if not (0 <= confidence <= 100 and 0 <= x < width and 0 <= y < height
                and 0 < w <= width - x and 0 < h <= height - y):
            raise ParseError("OCR returned invalid text coordinates")
        key = tuple(row[k] for k in ("page_num", "block_num", "par_num"))
        group = groups.setdefault(key, {"lines": {}, "boxes": [], "scores": []})
        group["lines"].setdefault(row["line_num"], []).append(row["text"].strip())
        group["boxes"].append((x, y, x + w, y + h))
        group["scores"].append(confidence)
    sections = Sections()
    for number, group in enumerate(groups.values(), 1):
        text = "\n".join(" ".join(words) for words in group["lines"].values())
        text = re.sub(r"(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])", "", text)
        x = min(b[0] for b in group["boxes"])
        y = min(b[1] for b in group["boxes"])
        w = max(b[2] for b in group["boxes"]) - x
        h = max(b[3] for b in group["boxes"]) - y
        start = len(sections.items)
        sections.add(f"Image {width}x{height} / block {number} / bbox {x},{y},{w},{h}", text)
        for section in sections.items[start:]:
            section.update(bbox=[x, y, w, h], image_size=[width, height],
                ocr_confidence=round(sum(group["scores"]) / len(group["scores"]), 2))
    return sections.items


def parse_image(data, fmt):
    image = decode_image(data, fmt)
    png = io.BytesIO()
    image.save(png, format="PNG")
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    output = io.BytesIO()
    preview.save(output, format="PNG")
    if len(output.getvalue()) > MAX_PREVIEW:
        raise ParseError("Image preview exceeds size limit")
    result = dict(sections=[], state="parse_failed", preview=base64.b64encode(output.getvalue()).decode("ascii"),
        warnings=["OCR text is unverified. Review against the image before publishing. Tables, formulas and diagram relations are not interpreted."])
    try:
        result["sections"] = ocr_sections(recognize(png.getvalue()), *image.size)
        result["state"] = "review" if result["sections"] else "ocr_empty"
        if not result["sections"]:
            result["error"] = "No readable text detected; upload a clearer text image. Image-only knowledge is not indexed."
        if any(s["ocr_confidence"] < 70 for s in result["sections"]):
            result["warnings"].append("Low OCR confidence in some blocks; check characters and numbers carefully. Confidence is not a correctness guarantee.")
    except subprocess.TimeoutExpired:
        result["error"] = "OCR timed out; upload a smaller or clearer text image"
    except (ParseError, ValueError, KeyError, UnicodeError) as error:
        result["error"] = str(error) if isinstance(error, ParseError) else "OCR returned invalid text; upload a new image"
    return result
