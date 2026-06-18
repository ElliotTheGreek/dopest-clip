import { useMemo, useState } from "react";
import type { EditorApi } from "../state/useEditor";

// Renders transcript words with their indices. Click a word to set the selection start,
// click another to set the end (drag also works via mousedown/enter). "Add as segment"
// pushes the [from..to] word range onto the EDL — the click-to-cut workflow.
export function Transcript({ ed }: { ed: EditorApi }) {
  const [anchor, setAnchor] = useState<number | null>(null);
  const [head, setHead] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);

  const range = useMemo(() => {
    if (anchor === null || head === null) return null;
    return { from: Math.min(anchor, head), to: Math.max(anchor, head) };
  }, [anchor, head]);

  if (!ed.project) {
    return (
      <div className="panel transcript">
        <div className="panel-head"><h2>Transcript</h2></div>
        <p className="empty">Open a project.</p>
      </div>
    );
  }
  if (!ed.project.transcribed || ed.words.length === 0) {
    return (
      <div className="panel transcript">
        <div className="panel-head"><h2>Transcript</h2></div>
        <p className="empty">Not transcribed yet — use the Transcribe button.</p>
      </div>
    );
  }

  return (
    <div className="panel transcript">
      <div className="panel-head">
        <h2>Transcript</h2>
        <div className="head-actions">
          {range && (
            <span className="sel-info">
              #{range.from}–#{range.to} ({range.to - range.from + 1} words)
            </span>
          )}
          <button
            className="primary"
            disabled={!range}
            onClick={() => {
              if (range) {
                ed.addSegment(range.from, range.to);
                setAnchor(null);
                setHead(null);
              }
            }}
          >
            Add as segment
          </button>
        </div>
      </div>

      <div
        className="words"
        onMouseUp={() => setDragging(false)}
        onMouseLeave={() => setDragging(false)}
      >
        {ed.words.map((w) => {
          const inRange = range && w.i >= range.from && w.i <= range.to;
          return (
            <span
              key={w.i}
              className={"word" + (inRange ? " in-range" : "")}
              title={`#${w.i}  ${w.start.toFixed(2)}–${w.end.toFixed(2)}s`}
              onMouseDown={() => {
                setAnchor(w.i);
                setHead(w.i);
                setDragging(true);
              }}
              onMouseEnter={() => {
                if (dragging) setHead(w.i);
              }}
              onClick={(e) => {
                if (e.shiftKey && anchor !== null) setHead(w.i);
              }}
            >
              <sup className="widx">{w.i}</sup>
              {w.w}{" "}
            </span>
          );
        })}
      </div>
      <p className="hint">
        Click-drag (or click then Shift-click) to select a word range, then "Add as segment". Ranges may
        overlap and segments are reorderable in the timeline.
      </p>
    </div>
  );
}
