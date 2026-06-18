import { describe, it, expect, vi, beforeEach } from "vitest";
import * as sidecar from "./sidecar";
import { emptyEdl, addSegment } from "../edl/edl";

// Mock the window.dopest bridge.
const rpc = vi.fn();
const ops = vi.fn();
const healthFn = vi.fn();
const readFileAsDataURL = vi.fn();

beforeEach(() => {
  rpc.mockReset();
  ops.mockReset();
  healthFn.mockReset();
  readFileAsDataURL.mockReset();
  (globalThis as unknown as { window: unknown }).window = {
    dopest: { rpc, ops: ops, health: healthFn, readFileAsDataURL },
  };
});

describe("call() envelope unwrapping", () => {
  it("returns result on ok:true", async () => {
    rpc.mockResolvedValue({ ok: true, result: { hello: "world" } });
    const r = await sidecar.call("anything", { a: 1 });
    expect(r).toEqual({ hello: "world" });
    expect(rpc).toHaveBeenCalledWith("anything", { a: 1 });
  });

  it("throws the error string on ok:false", async () => {
    rpc.mockResolvedValue({ ok: false, error: "ValueError: boom" });
    await expect(sidecar.call("x")).rejects.toThrow("ValueError: boom");
  });

  it("throws a generic message when ok:false has no error", async () => {
    rpc.mockResolvedValue({ ok: false });
    await expect(sidecar.call("doThing")).rejects.toThrow("rpc 'doThing' failed");
  });

  it("throws when the bridge returns nothing", async () => {
    rpc.mockResolvedValue(undefined);
    await expect(sidecar.call("x")).rejects.toThrow();
  });
});

describe("editing op wrappers pass params through", () => {
  it("listProjects calls list_projects with no params", async () => {
    rpc.mockResolvedValue({ ok: true, result: { projects: [], count: 0 } });
    const r = await sidecar.listProjects();
    expect(rpc).toHaveBeenCalledWith("list_projects", {});
    expect(r.count).toBe(0);
  });

  it("getProject passes project_id", async () => {
    rpc.mockResolvedValue({ ok: true, result: { project_id: "p1", transcribed: false } });
    await sidecar.getProject("p1");
    expect(rpc).toHaveBeenCalledWith("get_project", { project_id: "p1" });
  });

  it("createProject omits project_id when not given", async () => {
    rpc.mockResolvedValue({ ok: true, result: { project_id: "auto" } });
    await sidecar.createProject("/v/a.mp4");
    expect(rpc).toHaveBeenCalledWith("create_project", { video_path: "/v/a.mp4" });
  });

  it("createProject includes project_id when given", async () => {
    rpc.mockResolvedValue({ ok: true, result: { project_id: "p1" } });
    await sidecar.createProject("/v/a.mp4", "p1");
    expect(rpc).toHaveBeenCalledWith("create_project", { video_path: "/v/a.mp4", project_id: "p1" });
  });

  it("transcribe forwards options", async () => {
    rpc.mockResolvedValue({ ok: true, result: { project_id: "p1" } });
    await sidecar.transcribe("p1", { language: "en", model: "small" });
    expect(rpc).toHaveBeenCalledWith("transcribe", { project_id: "p1", language: "en", model: "small" });
  });

  it("validateEdl sends the EDL as edl_obj in wire form", async () => {
    rpc.mockResolvedValue({ ok: true, result: { segments: [], total_duration: 0, reconstructed_text: "", warnings: [], edl_id: "c" } });
    let e = addSegment(emptyEdl("c", "Title"), 0, 5, "hook");
    await sidecar.validateEdl("p1", e);
    expect(rpc).toHaveBeenCalledWith("validate_edl", {
      project_id: "p1",
      edl_obj: { edl_id: "c", title: "Title", segments: [{ from_word: 0, to_word: 5, label: "hook" }] },
    });
  });

  it("render with an EDL object sends edl_obj_or_id as wire EDL", async () => {
    rpc.mockResolvedValue({ ok: true, result: { render: "/x.mp4" } });
    const e = addSegment(emptyEdl("c"), 0, 1, "a");
    await sidecar.render("p1", e, { aspect: "9:16" });
    expect(rpc).toHaveBeenCalledWith("render", {
      project_id: "p1",
      edl_obj_or_id: { edl_id: "c", title: "Untitled clip", segments: [{ from_word: 0, to_word: 1, label: "a" }] },
      aspect: "9:16",
    });
  });

  it("render with a string id sends edl_obj_or_id as the id", async () => {
    rpc.mockResolvedValue({ ok: true, result: { render: "/x.mp4" } });
    await sidecar.render("p1", "saved-edl");
    expect(rpc).toHaveBeenCalledWith("render", { project_id: "p1", edl_obj_or_id: "saved-edl" });
  });

  it("verifyClip omits backend when not given", async () => {
    rpc.mockResolvedValue({ ok: true, result: { match_ratio: 1 } });
    await sidecar.verifyClip("p1", "e1");
    expect(rpc).toHaveBeenCalledWith("verify_clip", { project_id: "p1", edl_id: "e1" });
  });

  it("grabFrame passes defaults", async () => {
    rpc.mockResolvedValue({ ok: true, result: { frame: "/f.png", at: 1, width: 10, height: 10 } });
    await sidecar.grabFrame("p1", 1.5);
    expect(rpc).toHaveBeenCalledWith("grab_frame", { project_id: "p1", at: 1.5, source: "source", grid: true });
  });

  it("previewReframe includes aspect only when given", async () => {
    rpc.mockResolvedValue({ ok: true, result: { frame: "/f.png", edl_id: "e1", at: 1, aspect: "9:16", crop_rect: {} } });
    await sidecar.previewReframe("p1", "e1", 1);
    expect(rpc).toHaveBeenCalledWith("preview_reframe", { project_id: "p1", edl_id: "e1", at: 1 });
  });
});

describe("provider op wrappers", () => {
  it("listProviders", async () => {
    rpc.mockResolvedValue({ ok: true, result: {} });
    await sidecar.listProviders();
    expect(rpc).toHaveBeenCalledWith("list_providers", {});
  });
  it("setProvider passes capability and provider_name", async () => {
    rpc.mockResolvedValue({ ok: true, result: { capability: "tts", active: "openai" } });
    const r = await sidecar.setProvider("tts", "openai");
    expect(rpc).toHaveBeenCalledWith("set_provider", { capability: "tts", provider_name: "openai" });
    expect(r.active).toBe("openai");
  });
});

describe("camera-mix + short-form op wrappers", () => {
  it("fileUrl builds an encoded dopest-file:// url", () => {
    expect(sidecar.fileUrl("E:\\proj\\renders\\a b.mp4")).toBe(
      "dopest-file://local/E%3A%5Cproj%5Crenders%5Ca%20b.mp4");
  });

  it("mixCamera passes project/edl/camera + options", async () => {
    rpc.mockResolvedValue({ ok: true, result: { output: "/m.mp4", matte_backend: "rvm-gpu" } });
    await sidecar.mixCamera("p1", "e1", "/c.mkv", { remove_background: true });
    expect(rpc).toHaveBeenCalledWith("mix_camera", {
      project_id: "p1", edl_id: "e1", camera_path: "/c.mkv", remove_background: true,
    });
  });

  it("getCutTranscript passes project + edl", async () => {
    rpc.mockResolvedValue({ ok: true, result: { project_id: "p1", edl_id: "e1", word_count: 5, cut_transcript_txt: "/t.txt", text: "x" } });
    const r = await sidecar.getCutTranscript("p1", "e1");
    expect(rpc).toHaveBeenCalledWith("get_cut_transcript", { project_id: "p1", edl_id: "e1" });
    expect(r.word_count).toBe(5);
  });

  it("makeShort passes word range + hook + options", async () => {
    rpc.mockResolvedValue({ ok: true, result: { output: "/s.mp4", size: [1080, 1920], duration: 8, hook: "The Idea" } });
    await sidecar.makeShort("p1", "e1", 0, 30, "The Idea", { caption_preset: "karaoke-bold" });
    expect(rpc).toHaveBeenCalledWith("make_short", {
      project_id: "p1", edl_id: "e1", from_word: 0, to_word: 30, hook_title: "The Idea", caption_preset: "karaoke-bold",
    });
  });

  it("listGraphics calls list_graphics with no params", async () => {
    rpc.mockResolvedValue({ ok: true, result: { kinds: {} } });
    await sidecar.listGraphics();
    expect(rpc).toHaveBeenCalledWith("list_graphics", {});
  });
});

describe("bridge passthroughs", () => {
  it("ops/health/readFileAsDataURL call the bridge directly", async () => {
    ops.mockResolvedValue({ groups: {}, operations: [] });
    healthFn.mockResolvedValue({ ok: true, name: "dopest-clip", version: "0.1.0" });
    readFileAsDataURL.mockResolvedValue("data:image/png;base64,AAAA");
    expect((await sidecar.ops()).operations).toEqual([]);
    expect((await sidecar.health()).version).toBe("0.1.0");
    expect(await sidecar.readFileAsDataURL("/x.png")).toMatch(/^data:image\/png/);
  });
});
