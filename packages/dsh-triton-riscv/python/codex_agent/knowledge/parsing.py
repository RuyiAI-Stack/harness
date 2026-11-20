"""Document extraction and standalone-image OCR in a bounded, credential-free child."""
import hashlib
import io
import json
import logging
import os
import signal
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

PARSER_VERSION = "pdf-docx-v1"
MAX_UPLOAD = 12 * 1024 * 1024
MAX_CHARS = 1_000_000
MAX_SECTION = 20000
MAX_SECTIONS = 2000
PARSE_TIMEOUT = 45


class ParseError(ValueError):
    pass


def filename_format(filename):
    if (not filename or len(filename) > 240 or filename != Path(filename).name
            or any(ord(c) < 32 or c in "\\/" for c in filename)):
        raise ParseError("Invalid filename")
    fmt = Path(filename).suffix.lower().lstrip(".")
    if fmt not in {"pdf", "docx", "png", "jpg", "jpeg"}:
        raise ParseError("Only PDF, DOCX, PNG and JPG/JPEG files are supported")
    return fmt


def check_signature(data, fmt):
    if not data or len(data) > MAX_UPLOAD:
        raise ParseError("File must be nonempty and no larger than 12 MiB")
    signatures = {"pdf": b"%PDF-", "docx": b"PK\x03\x04", "png": b"\x89PNG\r\n\x1a\n",
                  "jpg": b"\xff\xd8\xff", "jpeg": b"\xff\xd8\xff"}
    if fmt not in signatures or not data.startswith(signatures[fmt]):
        raise ParseError("File contents do not match the selected file format")


def parser_version(fmt):
    from .images import IMAGE_FORMATS, PARSER_VERSION as image_version
    return image_version if fmt in IMAGE_FORMATS else PARSER_VERSION


class Sections:
    def __init__(self):
        self.items, self.size = [], 0

    def add(self, locator, text):
        text = text.replace("\x00", "").strip()
        if not text:
            return
        self.size += len(text)
        if self.size > MAX_CHARS:
            raise ParseError("Extracted text exceeds 1,000,000 characters; split the source file")
        # Preserve line/paragraph boundaries; hard splitting is a final fallback.
        pieces = []
        while len(text) > MAX_SECTION:
            end = text.rfind("\n", MAX_SECTION // 2, MAX_SECTION)
            end = end + 1 if end >= 0 else MAX_SECTION
            pieces.append(text[:end])
            text = text[end:]
        pieces.append(text)
        for number, piece in enumerate(pieces, 1):
            location = locator + (f" / part {number}" if len(pieces) > 1 else "")
            self.items.append(dict(id=f"s{len(self.items) + 1}", locator=location,
                                   text=piece, edited=False))
            if len(self.items) > MAX_SECTIONS:
                raise ParseError("Too many text sections; split the source file")


def parse_pdf(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise ParseError("Encrypted PDF is unsupported; upload an unlocked copy")
    if len(reader.pages) > 300:
        raise ParseError("PDF exceeds 300 pages; split the source file")
    parts, warnings, empty = Sections(), [], []
    for number, page in enumerate(reader.pages, 1):
        content = page.get_contents()
        if content is not None and len(content.get_data()) > 20 * 1024 * 1024:
            raise ParseError("PDF page content is too large to extract safely")
        text = page.extract_text() or ""
        if text.strip():
            parts.add(f"PDF page {number}", text)
        else:
            empty.append(number)
    if empty:
        warnings.append("No extractable text on PDF pages: " + ", ".join(map(str, empty)) +
                        ". Scanned/image-only pages require OCR; blank pages may also be listed.")
    warnings.append("PDF text extraction does not verify reading order, tables, formulas or images. Review against the original.")
    return dict(sections=parts.items, warnings=warnings,
                state="review" if parts.items else "ocr_required")


def safe_docx(data):
    from lxml import etree
    archive = zipfile.ZipFile(io.BytesIO(data))
    infos = archive.infolist()
    if len(infos) > 1000 or len({i.filename for i in infos}) != len(infos):
        raise ParseError("DOCX has too many or duplicate archive entries")
    total = 0
    xmls = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename or info.flag_bits & 1:
            raise ParseError("DOCX contains unsafe or encrypted archive entries")
        total += info.file_size
        if (total > 48 * 1024 * 1024 or info.file_size > 20 * 1024 * 1024
                or info.file_size > max(1, info.compress_size) * 500):
            raise ParseError("DOCX archive exceeds decompression limits")
        if any(word in info.filename.lower() for word in ("vbaproject", "embeddings/", "activex/")):
            raise ParseError("DOCX macros, active content and embedded files are not supported")
        if info.filename.endswith((".xml", ".rels")):
            raw = archive.read(info)
            if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                raise ParseError("XML entities and DTDs are not allowed")
            parsed = etree.fromstring(raw, parser=etree.XMLParser(
                resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False))
            if parsed.getroottree().docinfo.doctype:
                raise ParseError("XML DTDs are not allowed")
            xmls[info.filename] = parsed
    if "word/document.xml" not in xmls or "[Content_Types].xml" not in xmls:
        raise ParseError("File is not a valid DOCX document")
    warnings = []
    for name, tree in xmls.items():
        if name.endswith(".rels"):
            for rel in tree:
                if rel.get("TargetMode") == "External":
                    if not rel.get("Type", "").endswith("/hyperlink"):
                        raise ParseError("DOCX external attachments or linked media are not supported")
                    warnings.append("External hyperlinks are retained as visible text only; targets were not fetched.")
    body = xmls["word/document.xml"]
    unsupported = {"ins": "tracked changes", "del": "tracked changes", "txbxContent": "text boxes",
                   "sdt": "content controls", "drawing": "images/drawings", "pict": "images/drawings",
                   "oMath": "formulas", "altChunk": "embedded content"}
    kinds = {unsupported[etree.QName(e).localname] for e in body.iter()
             if isinstance(e.tag, str) and etree.QName(e).localname in unsupported}
    if kinds:
        warnings.append("Some content may not be extracted: " + ", ".join(sorted(kinds)) + ". Review the original.")
    if any(name.startswith(("word/header", "word/footer", "word/footnotes", "word/endnotes", "word/comments")) for name in xmls):
        warnings.append("Headers, footers, notes and comments are not included in body extraction.")
    return list(dict.fromkeys(warnings))


def parse_docx(data):
    from docx import Document
    from docx.table import Table
    warnings = safe_docx(data)
    doc, parts = Document(io.BytesIO(data)), Sections()
    heading = ""
    for number, block in enumerate(doc.iter_inner_content(), 1):
        if isinstance(block, Table):
            rows = []
            for row_number, row in enumerate(block.rows, 1):
                cells, seen = [], set()
                for cell in row.cells:
                    if cell._tc in seen:
                        continue
                    seen.add(cell._tc)
                    cells.append(cell.text)
                    if cell.tables:
                        warnings.append("Nested tables require manual review; only outer cell text was extracted.")
                rows.append(" | ".join(cells))
            parts.add(f"DOCX table {number}" + (f" / {heading}" if heading else ""), "\n".join(rows))
        else:
            if block.style and block.style.name.startswith("Heading"):
                heading = block.text[:200]
            parts.add(f"DOCX paragraph {number}" + (f" / {heading}" if heading else ""), block.text)
    if not parts.items:
        raise ParseError("DOCX has no extractable body text; images require OCR")
    return dict(sections=parts.items, warnings=list(dict.fromkeys(warnings)), state="review")


def parse_bytes(data, fmt):
    check_signature(data, fmt)
    if fmt in {"png", "jpg", "jpeg"}:
        from .images import parse_image
        return parse_image(data, fmt)
    return parse_pdf(data) if fmt == "pdf" else parse_docx(data)


def _parse_process(path, fmt, sha256):
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "LANG", "LC_ALL"}}
    proc = subprocess.Popen([sys.executable, "-I", "-m", __name__, str(path), fmt, sha256],
        env=env, cwd=path.parent, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=PARSE_TIMEOUT)
        return proc.returncode, output
    except BaseException:
        # Include OCR grandchildren when a parser times out or crashes.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise
    finally:
        proc.stdout.close()


def isolated_parse(path, fmt, sha256):
    try:
        code, output = _parse_process(path, fmt, sha256)
    except subprocess.TimeoutExpired:
        return dict(state="parse_failed", error="Parsing timed out; split or simplify the file", sections=[], warnings=[])
    if code != 0 or len(output) > 16 * 1024 * 1024:
        return dict(state="parse_failed", error="Parser stopped or exceeded resource limits", sections=[], warnings=[])
    try:
        return json.loads(output)
    except ValueError:
        return dict(state="parse_failed", error="Parser returned an invalid result", sections=[], warnings=[])


def main():
    import resource
    from .parsing import ParseError, parse_bytes
    resource.setrlimit(resource.RLIMIT_CPU, (35, 35))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    logging.disable(logging.CRITICAL)
    try:
        path, fmt, sha = sys.argv[1:]
        with Path(path).open("rb") as source:
            data = source.read(MAX_UPLOAD + 1)
        if hashlib.sha256(data).hexdigest() != sha:
            raise ParseError("Original file hash mismatch; re-upload the file")
        result = parse_bytes(data, fmt)
    except ParseError as error:
        result = dict(state="parse_failed", error=str(error), sections=[], warnings=[])
    except Exception:
        result = dict(state="parse_failed", error="Malformed or unsupported file; export a new copy", sections=[], warnings=[])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
