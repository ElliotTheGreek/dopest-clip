"""Background job registry for long-running renders.

Real video renders (matte + composite over a multi-minute clip) take minutes — far longer
than a synchronous MCP tool call should block. So the heavy ops run in a daemon thread here:
start_render() returns a job_id immediately and render_status(job_id) is polled until done.
The work is ffmpeg subprocesses + torch CUDA, both of which release the GIL, so the MCP
server stays responsive to status polls while a render runs.

Process-local (jobs live as long as the server process). Pure-stdlib; imported cheaply.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def start(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Run fn(*args, **kwargs) in a daemon thread. Returns a job_id to poll with status()."""
    jid = "job_" + uuid.uuid4().hex[:8]
    rec: dict[str, Any] = {"job_id": jid, "label": label, "status": "running",
                           "started": time.time(), "finished": None, "result": None, "error": None}
    with _LOCK:
        _JOBS[jid] = rec

    def run() -> None:
        try:
            res = fn(*args, **kwargs)
            rec["result"] = res
            rec["status"] = "done"
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
            rec["status"] = "error"
        finally:
            rec["finished"] = time.time()

    t = threading.Thread(target=run, name=jid, daemon=True)
    rec["_thread"] = t
    t.start()
    return jid


def _public(rec: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    end = rec.get("finished") or time.time()
    out["elapsed_s"] = round(end - rec["started"], 1)
    return out


def status(job_id: str) -> dict[str, Any]:
    with _LOCK:
        rec = _JOBS.get(job_id)
    if rec is None:
        return {"error": f"no such job {job_id!r}"}
    return _public(rec)


def list_jobs() -> dict[str, Any]:
    with _LOCK:
        recs = list(_JOBS.values())
    return {"jobs": [_public(r) for r in recs], "count": len(recs)}
