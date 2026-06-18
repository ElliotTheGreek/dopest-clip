import { useState } from "react";
import type { EditorApi } from "../state/useEditor";

export function Projects({ ed }: { ed: EditorApi }) {
  const [newPath, setNewPath] = useState("");

  return (
    <div className="panel projects">
      <div className="panel-head">
        <h2>Projects</h2>
        <button className="ghost" onClick={() => ed.refreshProjects()} title="Refresh">
          &#x21bb;
        </button>
      </div>

      <div className="new-project">
        <input
          type="text"
          placeholder="Absolute path to a video file…"
          value={newPath}
          onChange={(e) => setNewPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && newPath.trim()) {
              ed.createProject(newPath.trim());
              setNewPath("");
            }
          }}
        />
        <button
          disabled={!newPath.trim() || !!ed.busy}
          onClick={() => {
            ed.createProject(newPath.trim());
            setNewPath("");
          }}
        >
          Create
        </button>
      </div>
      <p className="hint">
        Paste a video file path and Create. (Electron file pickers need an OS dialog; a typed path works in
        every environment, including headless.)
      </p>

      <ul className="project-list">
        {ed.projects.length === 0 && <li className="empty">No projects yet.</li>}
        {ed.projects.map((p) => (
          <li
            key={p.project_id}
            className={ed.project?.project_id === p.project_id ? "active" : ""}
            onClick={() => ed.openProject(p.project_id)}
          >
            <span className="pid">{p.project_id}</span>
            <span className="meta">
              {p.transcribed ? "transcribed" : "not transcribed"}
              {typeof p.duration === "number" ? ` · ${p.duration.toFixed(1)}s` : ""}
            </span>
          </li>
        ))}
      </ul>

      {ed.project && (
        <div className="project-detail">
          <div className="kv">
            <span>source</span>
            <span title={ed.project.source}>{shorten(ed.project.source)}</span>
          </div>
          <div className="kv">
            <span>size</span>
            <span>
              {ed.project.width}×{ed.project.height} @ {ed.project.fps}fps
            </span>
          </div>
          <button
            className="primary block"
            disabled={!!ed.busy}
            onClick={() => ed.transcribe()}
          >
            {ed.project.transcribed ? "Re-transcribe" : "Transcribe"}
          </button>
        </div>
      )}
    </div>
  );
}

function shorten(s?: string): string {
  if (!s) return "—";
  return s.length > 40 ? "…" + s.slice(-37) : s;
}
