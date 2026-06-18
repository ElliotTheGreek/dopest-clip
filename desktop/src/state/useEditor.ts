// Central editor session state. Holds the connection status, the loaded project +
// transcript, the EDL under edit, the last validate_edl result, and busy/error flags.
// Components read/mutate through the returned actions; EDL mutations go through the pure
// helpers in src/edl/edl.ts so state stays immutable and testable.

import { useCallback, useEffect, useRef, useState } from "react";
import * as sc from "../api/sidecar";
import type { TranscriptWord, ProjectDetail, ProjectSummary } from "../api/types";
import type { Edl } from "../edl/types";
import type { ValidateResult } from "../edl/types";
import * as E from "../edl/edl";

export interface ConnState {
  connected: boolean;
  version?: string;
  error?: string;
}

export function useEditor() {
  const [conn, setConn] = useState<ConnState>({ connected: false });
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [words, setWords] = useState<TranscriptWord[]>([]);
  const [edl, setEdl] = useState<Edl>(() => E.emptyEdl());
  const [validation, setValidation] = useState<ValidateResult | null>(null);
  const [selectedSeg, setSelectedSeg] = useState<number | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const run = useCallback(async <T,>(label: string, fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(label);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    } finally {
      setBusy(null);
    }
  }, []);

  // --- connection / discovery ---------------------------------------------------------
  const refreshProjects = useCallback(async () => {
    const r = await run("Loading projects", () => sc.listProjects());
    if (r) setProjects(r.projects);
  }, [run]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await sc.health();
        if (!cancelled) setConn({ connected: !!h.ok, version: h.version });
        await refreshProjects();
      } catch (e) {
        if (!cancelled) setConn({ connected: false, error: e instanceof Error ? e.message : String(e) });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshProjects]);

  // --- project lifecycle --------------------------------------------------------------
  const openProject = useCallback(
    async (pid: string) => {
      const detail = await run(`Opening ${pid}`, () => sc.getProject(pid));
      if (!detail) return;
      if (detail.error) {
        setError(detail.error);
        return;
      }
      setProject(detail);
      setEdl(E.emptyEdl(undefined, "Untitled clip"));
      setValidation(null);
      setSelectedSeg(null);
      setWords([]);
      if (detail.transcribed) {
        const t = await run("Loading transcript", () => sc.getTranscriptJson(pid));
        if (t && Array.isArray((t as { words?: TranscriptWord[] }).words)) {
          setWords((t as { words: TranscriptWord[] }).words);
        }
      }
    },
    [run],
  );

  const createProject = useCallback(
    async (videoPath: string) => {
      const r = await run("Creating project", () => sc.createProject(videoPath));
      if (!r) return;
      if (r.error) {
        setError(r.error);
        return;
      }
      await refreshProjects();
      if (r.project_id) await openProject(r.project_id);
    },
    [run, refreshProjects, openProject],
  );

  const transcribe = useCallback(async () => {
    if (!project) return;
    const r = await run("Transcribing (this can take a while)", () => sc.transcribe(project.project_id));
    if (!r) return;
    if (r.error) {
      setError(r.error);
      return;
    }
    setToast(`Transcribed: ${r.word_count} words, ${r.silence_count} silences`);
    await openProject(project.project_id);
  }, [project, run, openProject]);

  // --- EDL editing (all immutable) ----------------------------------------------------
  const mutate = useCallback((fn: (e: Edl) => Edl) => {
    setEdl((prev) => fn(prev));
  }, []);

  const addSegment = useCallback(
    (from: number, to: number, label?: string) =>
      mutate((e) => E.addSegment(e, from, to, label ?? `seg${e.segments.length + 1}`)),
    [mutate],
  );
  const removeSegment = useCallback(
    (i: number) =>
      mutate((e) => E.removeSegment(e, i)),
    [mutate],
  );
  const updateSegment = useCallback(
    (i: number, patch: Partial<{ from_word: number; to_word: number; label: string }>) =>
      mutate((e) => E.updateSegment(e, i, patch)),
    [mutate],
  );
  const moveSegment = useCallback((from: number, to: number) => mutate((e) => E.moveSegment(e, from, to)), [mutate]);
  const duplicateSegment = useCallback((i: number) => mutate((e) => E.duplicateSegment(e, i)), [mutate]);
  const applyStyling = useCallback((patch: E.StylingPatch) => mutate((e) => E.applyStyling(e, patch)), [mutate]);
  const addReframeShot = useCallback((shot: { start: number; mode: string; zoom?: number }) => mutate((e) => E.addReframeShot(e, shot)), [mutate]);
  const updateReframeShot = useCallback((i: number, patch: Record<string, unknown>) => mutate((e) => E.updateReframeShot(e, i, patch)), [mutate]);
  const removeReframeShot = useCallback((i: number) => mutate((e) => E.removeReframeShot(e, i)), [mutate]);

  // --- validate / render / verify -----------------------------------------------------
  const validate = useCallback(async () => {
    if (!project) return;
    const r = await run("Validating EDL", () => sc.validateEdl(project.project_id, edl));
    if (!r) return;
    if ((r as { error?: string }).error) {
      setError((r as { error?: string }).error!);
      setValidation(null);
      return;
    }
    setValidation(r);
    // adopt the saved edl_id so render/verify target the same saved EDL
    if (r.edl_id) setEdl((prev) => (prev.edl_id === r.edl_id ? prev : { ...prev, edl_id: r.edl_id! }));
  }, [project, edl, run]);

  const renderClip = useRef<(opts?: { aspect?: string }) => Promise<unknown>>();
  renderClip.current = async (opts) => {
    if (!project) return;
    const r = await run("Rendering", () => sc.render(project.project_id, edl, opts));
    if (r && (r as { error?: string }).error) setError((r as { error?: string }).error!);
    return r;
  };

  return {
    conn,
    projects,
    project,
    words,
    edl,
    validation,
    selectedSeg,
    busy,
    error,
    toast,
    setSelectedSeg,
    setError,
    setToast,
    refreshProjects,
    openProject,
    createProject,
    transcribe,
    addSegment,
    removeSegment,
    updateSegment,
    moveSegment,
    duplicateSegment,
    applyStyling,
    addReframeShot,
    updateReframeShot,
    removeReframeShot,
    validate,
    render: (opts?: { aspect?: string }) => renderClip.current?.(opts),
    run,
  };
}

export type EditorApi = ReturnType<typeof useEditor>;
