import { useState } from "react";
import type { EditorApi } from "../state/useEditor";
import * as sc from "../api/sidecar";
import type { RenderResult, VerifyResult } from "../api/types";

// Shows grab_frame / preview_reframe stills (disk PNG paths via the file bridge), an
// in-window VIDEO PLAYER for rendered outputs (streamed over dopest-file://), and the
// render / verify / camera-mix / make-short actions — i.e. the editor drives the SAME
// pipeline the MCP agent does, and you can watch the result without leaving the app.
export function Preview({ ed }: { ed: EditorApi }) {
  const [frameSrc, setFrameSrc] = useState<string | null>(null);
  const [frameInfo, setFrameInfo] = useState<string>("");
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [videoLabel, setVideoLabel] = useState<string>("");
  const [at, setAt] = useState(1.0);
  const [renderRes, setRenderRes] = useState<RenderResult | null>(null);
  const [verifyRes, setVerifyRes] = useState<VerifyResult | null>(null);
  const [localErr, setLocalErr] = useState<string | null>(null);

  // camera mix + short form
  const [cameraPath, setCameraPath] = useState("");
  const [removeBg, setRemoveBg] = useState(true);
  const [mixInfo, setMixInfo] = useState<string>("");
  const [cutTxt, setCutTxt] = useState<string>("");
  const [fromWord, setFromWord] = useState(0);
  const [toWord, setToWord] = useState(0);
  const [hook, setHook] = useState("");
  const [shortInfo, setShortInfo] = useState<string>("");

  const pid = ed.project?.project_id;
  const edlId = ed.edl.edl_id;

  async function showPath(path: string) {
    try {
      setFrameSrc(await sc.readFileAsDataURL(path));
    } catch (e) {
      setLocalErr(e instanceof Error ? e.message : String(e));
    }
  }

  function playPath(path: string, label: string) {
    setLocalErr(null);
    setVideoSrc(sc.fileUrl(path));
    setVideoLabel(label);
  }

  async function doGrabFrame() {
    if (!pid) return;
    setLocalErr(null);
    const r = await ed.run("grab_frame", () => sc.grabFrame(pid, at, "source", true));
    if (!r) return;
    if ((r as { error?: string }).error) { setLocalErr((r as { error?: string }).error!); return; }
    setFrameInfo(`source @ ${r.at}s · ${r.width}×${r.height}`);
    await showPath(r.frame);
  }

  async function doPreviewReframe() {
    if (!pid) return;
    setLocalErr(null);
    const r = await ed.run("preview_reframe", () => sc.previewReframe(pid, edlId, at));
    if (!r) return;
    if ((r as { error?: string }).error) { setLocalErr((r as { error?: string }).error!); return; }
    setFrameInfo(`reframe ${r.aspect} @ ${r.at}s`);
    await showPath(r.frame);
  }

  async function doRender() {
    setRenderRes(null);
    const r = (await ed.render()) as RenderResult | undefined;
    if (r) {
      setRenderRes(r);
      if (r.outputs?.cut) playPath(r.outputs.cut, "cut");
    }
  }

  async function doVerify() {
    if (!pid) return;
    setVerifyRes(null);
    const r = await ed.run("verify_clip", () => sc.verifyClip(pid, edlId));
    if (r) setVerifyRes(r);
  }

  async function doMix() {
    if (!pid || !cameraPath) return;
    setLocalErr(null);
    setMixInfo("");
    const r = await ed.run("mix_camera", () =>
      sc.mixCamera(pid, edlId, cameraPath, { remove_background: removeBg }));
    if (!r) return;
    const out = (r as { output?: string; matte_backend?: string }).output;
    setMixInfo(`mixed via ${(r as { matte_backend?: string }).matte_backend ?? "?"} → ${base(out)}`);
    if (out) playPath(out, "camera mix");
  }

  async function doCutTranscript() {
    if (!pid) return;
    setLocalErr(null);
    const r = await ed.run("get_cut_transcript", () => sc.getCutTranscript(pid, edlId));
    if (!r) return;
    setCutTxt(r.text || "");
    if (r.word_count > 0) setToWord(Math.min(r.word_count - 1, 40));
  }

  async function doMakeShort() {
    if (!pid) return;
    setLocalErr(null);
    setShortInfo("");
    const r = await ed.run("make_short", () =>
      sc.makeShort(pid, edlId, fromWord, toWord, hook || "Clip"));
    if (!r) return;
    setShortInfo(`9:16 short ${r.size?.join("×")} · ${r.duration}s → ${base(r.output)}`);
    if (r.output) playPath(r.output, "short");
  }

  return (
    <div className="panel preview">
      <div className="panel-head"><h2>Preview</h2></div>

      <div className="preview-controls">
        <label>
          at (s)
          <input type="number" step="0.5" value={at} onChange={(e) => setAt(Number(e.target.value))} />
        </label>
        <button disabled={!pid || !!ed.busy} onClick={doGrabFrame}>Grab frame</button>
        <button disabled={!pid || !!ed.busy} onClick={doPreviewReframe} title="needs a render first">Preview reframe</button>
      </div>

      {/* in-window video player for rendered outputs (streamed, seekable) */}
      <div className="video-view">
        {videoSrc ? (
          <>
            <video key={videoSrc} src={videoSrc} controls autoPlay style={{ maxWidth: "100%", maxHeight: 360 }} />
            <div className="frame-info">▶ {videoLabel}</div>
          </>
        ) : (
          <div className="empty">No video loaded — render, mix a camera, or make a short to watch it here.</div>
        )}
      </div>

      <div className="frame-view">
        {frameSrc ? <img src={frameSrc} alt="frame" /> : null}
        {frameInfo && <div className="frame-info">{frameInfo}</div>}
      </div>

      <div className="render-controls">
        <button className="primary" disabled={!pid || ed.edl.segments.length === 0 || !!ed.busy} onClick={doRender}>
          Render
        </button>
        <button disabled={!pid || !!ed.busy} onClick={doVerify} title="QA the last render against the EDL">
          Verify
        </button>
      </div>

      {localErr && <div className="error-box">{localErr}</div>}

      {renderRes && (
        <div className="result-box">
          <h4>Render</h4>
          {renderRes.error ? (
            <div className="warn">{renderRes.error}</div>
          ) : (
            <>
              <div className="kv">
                <span>cut</span>
                <span title={renderRes.outputs?.cut}>
                  {base(renderRes.outputs?.cut)}
                  {renderRes.outputs?.cut && <button className="ghost" onClick={() => playPath(renderRes.outputs!.cut, "cut")}>▶</button>}
                </span>
              </div>
              {renderRes.outputs?.styled &&
                Object.entries(renderRes.outputs.styled).map(([asp, p]) => (
                  <div className="kv" key={asp}>
                    <span>{asp}</span>
                    <span title={p}>{base(p)}<button className="ghost" onClick={() => playPath(p, asp)}>▶</button></span>
                  </div>
                ))}
              <div className="kv"><span>duration</span><span>{renderRes.expected_duration?.toFixed(2)}s</span></div>
              {renderRes.warnings && renderRes.warnings.length > 0 && (
                <ul className="warnings">{renderRes.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
              )}
            </>
          )}
        </div>
      )}

      {verifyRes && (
        <div className="result-box">
          <h4>Verify</h4>
          {verifyRes.error ? (
            <div className="warn">{verifyRes.error}</div>
          ) : (
            <div className="kv">
              <span>match ratio</span>
              <span className={(verifyRes.match_ratio ?? 0) >= 0.95 ? "ok" : "warn"}>
                {typeof verifyRes.match_ratio === "number" ? (verifyRes.match_ratio * 100).toFixed(1) + "%" : "—"}
              </span>
            </div>
          )}
        </div>
      )}

      {/* camera mix — background-removed (GPU) talking head over the cut */}
      <div className="result-box">
        <h4>Camera mix</h4>
        <label className="full">
          camera file
          <input type="text" placeholder="…\recordings\camera_*.mkv" value={cameraPath}
            onChange={(e) => setCameraPath(e.target.value)} />
        </label>
        <label className="inline">
          <input type="checkbox" checked={removeBg} onChange={(e) => setRemoveBg(e.target.checked)} />
          remove background (cut me out)
        </label>
        <button disabled={!pid || !cameraPath || !!ed.busy} onClick={doMix}>Mix camera</button>
        {mixInfo && <div className="frame-info">{mixInfo}</div>}
      </div>

      {/* make a 9:16 short — captions top / screen mid / person bottom */}
      <div className="result-box">
        <h4>Short form (9:16)</h4>
        <button disabled={!pid || !!ed.busy} onClick={doCutTranscript} title="cut-timeline word indices to design the short on">
          Cut transcript
        </button>
        {cutTxt && <pre className="cut-transcript">{cutTxt}</pre>}
        <div className="short-controls">
          <label>from <input type="number" value={fromWord} onChange={(e) => setFromWord(Number(e.target.value))} /></label>
          <label>to <input type="number" value={toWord} onChange={(e) => setToWord(Number(e.target.value))} /></label>
        </div>
        <label className="full">
          hook title
          <input type="text" placeholder="e.g. The Idea" value={hook} onChange={(e) => setHook(e.target.value)} />
        </label>
        <button className="primary" disabled={!pid || toWord <= fromWord || !!ed.busy} onClick={doMakeShort}
          title="needs a render + a camera mix (remove_background) first">
          Make short
        </button>
        {shortInfo && <div className="frame-info">{shortInfo}</div>}
      </div>
    </div>
  );
}

function base(p?: string): string {
  if (!p) return "—";
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1];
}
