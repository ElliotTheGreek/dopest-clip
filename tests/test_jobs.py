"""Background render-job registry + the start_render/render_status MCP ops."""

import time

from dopest_clip import api, jobs


def _wait(jid, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        s = jobs.status(jid)
        if s.get("status") != "running":
            return s
        time.sleep(0.02)
    return jobs.status(jid)


def test_job_runs_in_background_and_returns_result():
    jid = jobs.start("double", lambda x: {"v": x * 2}, 21)
    assert jid.startswith("job_")
    s = _wait(jid)
    assert s["status"] == "done"
    assert s["result"] == {"v": 42}
    assert s["elapsed_s"] >= 0


def test_job_captures_error_without_crashing():
    def boom():
        raise ValueError("nope")
    s = _wait(jobs.start("boom", boom))
    assert s["status"] == "error"
    assert "ValueError: nope" in s["error"]
    assert "traceback" in s


def test_status_unknown_job():
    assert "error" in jobs.status("job_does_not_exist")


def test_start_render_rejects_non_render_op():
    r = api.start_render("list_projects", {})
    assert "error" in r and "list_projects" not in api._RENDER_OPS


def test_start_render_routes_render_op_in_background(monkeypatch):
    # stub mix_camera in the OPERATIONS registry so no GPU/ffmpeg runs
    monkeypatch.setitem(api.OPERATIONS, "mix_camera", lambda **k: {"ok": True, "kw": k})
    r = api.start_render("mix_camera", {"project_id": "p", "edl_id": "e", "camera_path": "c"})
    assert r["status"] == "running" and r["job_id"].startswith("job_")
    s = _wait(r["job_id"])
    assert s["status"] == "done"
    assert s["result"]["ok"] is True and s["result"]["kw"]["project_id"] == "p"


def test_list_render_jobs_reports_jobs():
    jobs.start("noop", lambda: 1)
    out = api.list_render_jobs()
    assert "jobs" in out and out["count"] >= 1
