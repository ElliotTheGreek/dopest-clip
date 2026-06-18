import { useState } from "react";
import type { EditorApi } from "../state/useEditor";
import type { ResolvedSegment } from "../edl/types";

// The EDL segments as reorderable blocks. Non-contiguous reorder is first-class: drag a
// block onto another to move it. Each block shows the clean/hard-cut state pulled from the
// last validate_edl result (matched by position in the resolved list).
export function Timeline({ ed }: { ed: EditorApi }) {
  const [dragIdx, setDragIdx] = useState<number | null>(null);

  // validate_edl can SPLIT segments (cleanup), so resolved segments may not map 1:1 to
  // raw segments. We match the warning/clean state of the first resolved segment whose
  // from_word falls inside the raw segment's range — a best-effort indicator.
  const resolved = ed.validation?.segments ?? [];
  function cutStateFor(from: number, to: number): { startClean?: boolean; endClean?: boolean; res?: ResolvedSegment } {
    const r = resolved.find((s) => s.from_word >= from && s.from_word <= to);
    if (!r) return {};
    return { startClean: r.start_clean, endClean: r.end_clean, res: r };
  }

  return (
    <div className="panel timeline">
      <div className="panel-head">
        <h2>Timeline · {ed.edl.title || "Untitled"}</h2>
        <div className="head-actions">
          <span className="sel-info">{ed.edl.segments.length} segments</span>
          <button className="primary" disabled={!ed.project || !!ed.busy} onClick={() => ed.validate()}>
            Validate
          </button>
        </div>
      </div>

      {ed.validation && (
        <div className="timeline-summary">
          total {ed.validation.total_duration?.toFixed(2)}s ·{" "}
          <span className={ed.validation.warnings.length ? "warn" : "ok"}>
            {ed.validation.warnings.length} warning{ed.validation.warnings.length === 1 ? "" : "s"}
          </span>
        </div>
      )}

      <div className="blocks">
        {ed.edl.segments.length === 0 && <p className="empty">No segments — add some from the transcript.</p>}
        {ed.edl.segments.map((s, i) => {
          const cs = cutStateFor(s.from_word, s.to_word);
          return (
            <div
              key={i}
              className={
                "block" +
                (ed.selectedSeg === i ? " selected" : "") +
                (dragIdx === i ? " dragging" : "")
              }
              draggable
              onDragStart={() => setDragIdx(i)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => {
                if (dragIdx !== null && dragIdx !== i) ed.moveSegment(dragIdx, i);
                setDragIdx(null);
              }}
              onDragEnd={() => setDragIdx(null)}
              onClick={() => ed.setSelectedSeg(i)}
            >
              <div className="block-order">{i + 1}</div>
              <div className="block-body">
                <div className="block-label">{s.label}</div>
                <div className="block-range">
                  #{s.from_word}–#{s.to_word}
                  {cs.res ? ` · ${cs.res.dur.toFixed(2)}s` : ""}
                </div>
                {cs.res && <div className="block-text">{cs.res.text}</div>}
              </div>
              <div className="block-cuts">
                <span className={"cut " + (cs.startClean === false ? "hard" : cs.startClean ? "clean" : "unknown")} title="start cut">
                  ◖
                </span>
                <span className={"cut " + (cs.endClean === false ? "hard" : cs.endClean ? "clean" : "unknown")} title="end cut">
                  ◗
                </span>
              </div>
              <div className="block-tools">
                <button className="ghost" title="Duplicate" onClick={(e) => { e.stopPropagation(); ed.duplicateSegment(i); }}>⧉</button>
                <button className="ghost" title="Move up" onClick={(e) => { e.stopPropagation(); ed.moveSegment(i, Math.max(0, i - 1)); }}>↑</button>
                <button className="ghost" title="Move down" onClick={(e) => { e.stopPropagation(); ed.moveSegment(i, i + 1); }}>↓</button>
                <button className="ghost danger" title="Remove" onClick={(e) => { e.stopPropagation(); ed.removeSegment(i); if (ed.selectedSeg === i) ed.setSelectedSeg(null); }}>✕</button>
              </div>
            </div>
          );
        })}
      </div>

      {ed.validation && ed.validation.warnings.length > 0 && (
        <ul className="warnings">
          {ed.validation.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
