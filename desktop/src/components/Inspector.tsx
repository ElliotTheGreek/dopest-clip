import { useEffect, useState } from "react";
import type { EditorApi } from "../state/useEditor";
import * as sc from "../api/sidecar";

// Edits the selected segment + EDL-level styling (reframe, captions, title card, loudnorm,
// export aspects) and a basic keyframe (reframe-shot) list editor.
export function Inspector({ ed }: { ed: EditorApi }) {
  const [presets, setPresets] = useState<string[]>([]);
  const [modes, setModes] = useState<string[]>([]);
  const [aspects, setAspects] = useState<string[]>([]);

  useEffect(() => {
    if (!ed.conn.connected) return;
    sc.listCaptionPresets().then((r) => setPresets(asNames(r.presets))).catch(() => {});
    // `modes` comes back as a dict {mode: description}; the dropdown wants the mode names.
    sc.listReframeModes().then((r) => { setModes(asNames(r.modes)); setAspects(asNames(r.aspects)); }).catch(() => {});
  }, [ed.conn.connected]);

  const edl = ed.edl;
  const seg = ed.selectedSeg !== null ? edl.segments[ed.selectedSeg] : null;

  return (
    <div className="panel inspector">
      <div className="panel-head"><h2>Inspector</h2></div>

      <section>
        <h3>Clip</h3>
        <label>
          Title
          <input
            type="text"
            value={edl.title ?? ""}
            onChange={(e) => ed.applyStyling({ title: e.target.value })}
          />
        </label>
      </section>

      <section>
        <h3>Segment {ed.selectedSeg !== null ? `#${ed.selectedSeg + 1}` : ""}</h3>
        {!seg && <p className="empty">Select a segment in the timeline.</p>}
        {seg && ed.selectedSeg !== null && (
          <div className="grid2">
            <label>
              Label
              <input
                type="text"
                value={seg.label}
                onChange={(e) => ed.updateSegment(ed.selectedSeg!, { label: e.target.value })}
              />
            </label>
            <span />
            <label>
              from_word
              <input
                type="number"
                value={seg.from_word}
                onChange={(e) => ed.updateSegment(ed.selectedSeg!, { from_word: Number(e.target.value) })}
              />
            </label>
            <label>
              to_word
              <input
                type="number"
                value={seg.to_word}
                onChange={(e) => ed.updateSegment(ed.selectedSeg!, { to_word: Number(e.target.value) })}
              />
            </label>
          </div>
        )}
      </section>

      <section>
        <h3>Cleanup</h3>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={!!edl.cleanup?.enabled}
            onChange={(e) =>
              ed.applyStyling({ cleanup: e.target.checked ? { enabled: true, remove_fillers: true } : undefined })
            }
          />
          Remove filler words + split long internal pauses
        </label>
      </section>

      <section>
        <h3>Reframe</h3>
        <label>
          Mode
          <select
            value={edl.reframe?.mode ?? "none"}
            onChange={(e) => {
              const mode = e.target.value;
              ed.applyStyling({ reframe: mode === "none" ? undefined : { ...(edl.reframe ?? {}), mode } });
            }}
          >
            <option value="none">none</option>
            {modes.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Aspect
          <select
            value={edl.reframe?.aspect ?? ""}
            onChange={(e) =>
              ed.applyStyling({ reframe: { ...(edl.reframe ?? { mode: "full" }), aspect: e.target.value || undefined } })
            }
            disabled={!edl.reframe}
          >
            <option value="">(source)</option>
            {aspects.map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </label>

        {/* Basic keyframe / shot list editor — v1 is a list, not a curve UI. */}
        <div className="keyframes">
          <div className="kf-head">
            <span>Shots / keyframes (clip seconds)</span>
            <button
              className="ghost"
              disabled={!edl.reframe}
              onClick={() => ed.addReframeShot({ start: lastShotEnd(edl.reframe?.shots), mode: edl.reframe?.mode ?? "full" })}
            >
              + add
            </button>
          </div>
          {(edl.reframe?.shots ?? []).length === 0 && <p className="empty small">No keyframes; the mode applies to the whole clip.</p>}
          {(edl.reframe?.shots ?? []).map((shot, i) => (
            <div className="kf-row" key={i}>
              <input
                type="number" step="0.1" title="start (s)"
                value={shot.start}
                onChange={(e) => ed.updateReframeShot(i, { start: Number(e.target.value) })}
              />
              <select value={shot.mode} onChange={(e) => ed.updateReframeShot(i, { mode: e.target.value })}>
                {(modes.length ? modes : ["full", "track", "zoom", "center", "focus"]).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <input
                type="number" step="0.1" title="zoom" placeholder="zoom"
                value={shot.zoom ?? ""}
                onChange={(e) => ed.updateReframeShot(i, { zoom: e.target.value === "" ? undefined : Number(e.target.value) })}
              />
              <button className="ghost danger" onClick={() => ed.removeReframeShot(i)}>✕</button>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3>Captions</h3>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={!!edl.captions?.enabled}
            onChange={(e) =>
              ed.applyStyling({
                captions: e.target.checked ? { enabled: true, preset: edl.captions?.preset ?? presets[0] ?? "karaoke-bold" } : undefined,
              })
            }
          />
          Burn captions
        </label>
        <label>
          Preset
          <select
            value={edl.captions?.preset ?? ""}
            disabled={!edl.captions?.enabled}
            onChange={(e) => ed.applyStyling({ captions: { ...(edl.captions ?? { enabled: true }), preset: e.target.value } })}
          >
            {presets.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
      </section>

      <section>
        <h3>Title card</h3>
        <label>
          Text
          <input
            type="text"
            value={edl.title_card?.text ?? ""}
            onChange={(e) =>
              ed.applyStyling({ title_card: e.target.value ? { ...(edl.title_card ?? {}), text: e.target.value } : undefined })
            }
          />
        </label>
        <label>
          Hold (s)
          <input
            type="number" step="0.5"
            value={edl.title_card?.hold_s ?? 3}
            disabled={!edl.title_card?.text}
            onChange={(e) => ed.applyStyling({ title_card: { ...(edl.title_card ?? {}), hold_s: Number(e.target.value) } })}
          />
        </label>
      </section>

      <section>
        <h3>Output</h3>
        <label className="checkbox">
          <input type="checkbox" checked={!!edl.loudnorm} onChange={(e) => ed.applyStyling({ loudnorm: e.target.checked })} />
          Loudness normalize
        </label>
        <label>
          Export aspects (comma sep)
          <input
            type="text"
            placeholder="e.g. 9:16, 1:1"
            value={(edl.export_aspects ?? []).join(", ")}
            onChange={(e) => {
              const list = e.target.value.split(",").map((s) => s.trim()).filter(Boolean);
              ed.applyStyling({ export_aspects: list.length ? list : undefined });
            }}
          />
        </label>
      </section>
    </div>
  );
}

// The sidecar returns reframe modes as a dict {mode: description} but caption presets /
// aspects as plain arrays. Normalize any of them to a list of names for the dropdowns.
function asNames(v: unknown): string[] {
  if (Array.isArray(v)) return v as string[];
  if (v && typeof v === "object") return Object.keys(v as Record<string, unknown>);
  return [];
}

function lastShotEnd(shots?: { start: number }[]): number {
  if (!shots || shots.length === 0) return 0;
  return Math.round((shots[shots.length - 1].start + 1) * 10) / 10;
}
