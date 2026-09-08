"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TopBar } from "@/components/top-bar";
import { SceneCanvas, type Pane } from "@/components/canvas/scene-canvas";
import { UploadPanel } from "@/components/upload/upload-panel";
import { QueryConsole } from "@/components/console/query-console";
import { ResultPanel } from "@/components/console/result-panel";
import { TracePanel } from "@/components/console/trace-panel";
import { streamQuery, uploadScene } from "@/lib/api";
import { MODE_ROLES } from "@/lib/tasks";
import type {
  QueryResult,
  SceneMeta,
  StagedFile,
  TraceStep,
  UploadMode,
  UploadResult,
} from "@/lib/types";

/** Formats a browser can decode; anything else falls back to the placeholder raster. */
const PREVIEWABLE = /\.(png|jpe?g|webp|gif)$/i;

function panesFor(upload: UploadResult, staged: StagedFile[]): Pane[] {
  const slots = MODE_ROLES[upload.mode];

  return upload.assets.map((asset, i) => {
    const meta: SceneMeta = {
      sceneId: asset.filename,
      role: asset.role,
      sizeBytes: asset.size_bytes,
      contentType: asset.content_type,
      // Nothing below is read out of the file yet — /upload stores it as bytes.
      sensor: null,
      acquired: null,
      lat: null,
      lon: null,
      gsd: null,
      epsg: null,
      bands: null,
    };
    const file = staged[i]?.file;
    return {
      meta,
      label: `${slots[i]?.label ?? asset.role} · ${asset.filename}`,
      previewUrl:
        file && PREVIEWABLE.test(asset.filename) ? URL.createObjectURL(file) : null,
      tint: asset.role === "sar" ? ("sar" as const) : ("optical" as const),
      seed: 11 + i * 7,
    };
  });
}

export default function Console() {
  const [mode, setMode] = useState<UploadMode>("single");
  const [panes, setPanes] = useState<Pane[]>([]);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [boundMode, setBoundMode] = useState<UploadMode | null>(null);
  const [binding, setBinding] = useState(false);
  const [bindError, setBindError] = useState<string | null>(null);

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [transportError, setTransportError] = useState<string | null>(null);
  const [steps, setSteps] = useState<TraceStep[]>([]);
  const [traceOpen, setTraceOpen] = useState(true);

  const inflight = useRef<AbortController | null>(null);
  const previewUrls = useRef<string[]>([]);

  // Object URLs outlive the render that made them, so they are revoked by hand.
  const replacePreviews = useCallback((next: Pane[]) => {
    previewUrls.current.forEach(URL.revokeObjectURL);
    previewUrls.current = next
      .map((pane) => pane.previewUrl)
      .filter((url): url is string => url !== null);
    return next;
  }, []);

  useEffect(
    () => () => {
      previewUrls.current.forEach(URL.revokeObjectURL);
      inflight.current?.abort();
    },
    [],
  );

  const bindScene = useCallback(
    async (nextMode: UploadMode, files: StagedFile[], benchmarkMode: boolean) => {
      inflight.current?.abort();
      setBinding(true);
      setBindError(null);

      try {
        const upload = await uploadScene(nextMode, files, benchmarkMode);
        setPanes(replacePreviews(panesFor(upload, files)));
        setUploadId(upload.upload_id);
        setBoundMode(upload.mode);
        setResult(null);
        setSteps([]);
        setTransportError(null);
      } catch (error) {
        setBindError(error instanceof Error ? error.message : String(error));
      } finally {
        setBinding(false);
      }
    },
    [replacePreviews],
  );

  const runQuery = useCallback(
    async (query: string) => {
      inflight.current?.abort();
      const controller = new AbortController();
      inflight.current = controller;

      setBusy(true);
      setResult(null);
      setSteps([]);
      setTransportError(null);
      setTraceOpen(true);

      try {
        const final = await streamQuery(
          { query, upload_id: uploadId },
          // Steps arrive in graph order, so appending blindly is enough.
          { onStep: (step) => setSteps((prev) => [...prev, step]) },
          controller.signal,
        );
        setResult(final);
        setSteps(final.steps);
      } catch (error) {
        if (controller.signal.aborted) return;
        setTransportError(error instanceof Error ? error.message : String(error));
      } finally {
        if (!controller.signal.aborted) setBusy(false);
      }
    },
    [uploadId],
  );

  const totalMs = steps.reduce((sum, s) => sum + (s.duration_ms ?? 0), 0);

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-void">
      <TopBar
        scene={panes[0]?.meta ?? null}
        intent={result?.intent ?? null}
        busy={busy}
      />

      <main className="flex min-h-0 flex-1 flex-col gap-3 p-3 lg:flex-row">
        <SceneCanvas panes={panes} mode={boundMode} overlays={result?.overlays ?? []}>
          <UploadPanel
            mode={mode}
            onModeChange={setMode}
            onBind={bindScene}
            binding={binding}
            bindError={bindError}
          />
        </SceneCanvas>

        <div className="flex min-h-0 w-full flex-col gap-3 lg:w-[440px] lg:shrink-0">
          <QueryConsole disabled={panes.length === 0} busy={busy} onSubmit={runQuery} />
          <ResultPanel
            result={result}
            panes={panes}
            busy={busy}
            hasScene={panes.length > 0}
            transportError={transportError}
          />
          <TracePanel
            steps={steps}
            open={traceOpen}
            onOpenChange={setTraceOpen}
            totalMs={totalMs}
            running={busy}
          />
        </div>
      </main>
    </div>
  );
}
