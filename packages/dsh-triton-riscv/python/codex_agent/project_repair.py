"""Review-gated project patch workflow for compiler and build failures."""
from __future__ import annotations
from codex_agent.runtime_config import permission_enabled
import hashlib, json, os, re, subprocess, time, uuid
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass(frozen=True)
class ProjectPatchProposal:
    proposal_id: str
    status: str
    rationale: str
    files: list[str]
    diff: str
    proposal_path: str
    requires_review: bool = True


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "<missing>"


def _files_from_diff(diff: str) -> list[str]:
    files=[]
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            p=line[6:].strip()
            if p != "/dev/null" and p not in files: files.append(p)
    return files


def propose_project_patch(repo_root: Path, diff: str, rationale: str, *, max_files: int = 8, max_bytes: int = 500_000) -> ProjectPatchProposal:
    root=repo_root.resolve()
    if not rationale.strip(): raise ValueError("rationale cannot be empty")
    if not diff.strip(): raise ValueError("diff cannot be empty")
    if len(diff.encode()) > max_bytes: raise ValueError("patch exceeds size limit")
    files=_files_from_diff(diff)
    if not files or len(files)>max_files: raise ValueError("patch file count exceeds limit")
    allowed_ext={".py",".cpp",".cc",".h",".hpp",".td",".mlir",".cmake",".txt",".toml"}
    for name in files:
        p=(root/name).resolve()
        try: p.relative_to(root)
        except ValueError as e: raise ValueError("patch escapes repository") from e
        if p.suffix not in allowed_ext and p.name not in {"CMakeLists.txt","Makefile"}: raise ValueError(f"file type is not repairable: {name}")
    pid=f"project-repair-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    out=root/'agent-results'/'project-repairs'; out.mkdir(parents=True,exist_ok=True)
    payload={"proposal_id":pid,"status":"pending_approval","rationale":rationale.strip(),"files":files,"diff":diff,"sha256":{n:_sha(root/n) for n in files},"created_at":time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    path=out/f"{pid}.json"; path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return ProjectPatchProposal(pid,"pending_approval",rationale.strip(),files,diff,path.relative_to(root).as_posix())


def review_project_patch(repo_root: Path, proposal_id: str, *, approve: bool, reviewer: str, note: str = "") -> dict:
    if not reviewer.strip(): raise ValueError("reviewer cannot be empty")
    root=repo_root.resolve(); path=root/'agent-results'/'project-repairs'/f'{proposal_id}.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['status']!='pending_approval': raise ValueError(f"proposal is already {data['status']}")
    data['status']='approved' if approve else 'rejected'; data['review']={'reviewer':reviewer.strip(),'note':note.strip(),'decided_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); return {k:v for k,v in data.items() if k!='diff'}


def apply_project_patch(repo_root: Path, proposal_id: str) -> dict:
    root=repo_root.resolve(); path=root/'agent-results'/'project-repairs'/f'{proposal_id}.json'; data=json.loads(path.read_text(encoding='utf-8'))
    if data['status']!='approved': return {"proposal_id":proposal_id,"status":"not_approved"}
    if not permission_enabled("repair"): return {"proposal_id":proposal_id,"status":"blocked","message":"enable permissions.repair to apply reviewed patches"}
    for name,expected in data['sha256'].items():
        if _sha(root/name)!=expected: raise RuntimeError(f"source changed after proposal: {name}")
    patch=root/'agent-results'/'project-repairs'/f'{proposal_id}.diff'; patch.write_text(data['diff'],encoding='utf-8')
    check=subprocess.run(['git','apply','--check',str(patch)],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if check.returncode: raise RuntimeError(f"git apply --check failed: {check.stdout.strip()}")
    result=subprocess.run(['git','apply',str(patch)],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(result.stdout.strip())
    data['status']='applied'; data['applied_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z'); path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return {"proposal_id":proposal_id,"status":"applied","files":data['files'],"patch_path":patch.relative_to(root).as_posix()}
