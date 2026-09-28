"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { RequireAuth } from "@/components/auth/require-auth";
import { useAuth } from "@/components/auth/auth-provider";
import {
  TourHelpButton,
  TourWelcome,
  useConsoleTour,
  type TourActions,
} from "@/components/tour/console-tour";
import { TopBar } from "@/components/top-bar";
import { SceneCanvas, type Pane } from "@/components/canvas/scene-canvas";
import { UploadPanel } from "@/components/upload/upload-panel";
import { Step, type StepState } from "@/components/console/step";
import { TaskPicker } from "@/components/console/task-picker";
import { SampleGallery } from "@/components/console/sample-gallery";
import { QuestionBox } from "@/components/console/question-box";
import { PipelineProgress } from "@/components/console/pipeline-progress";
import { ResultPanel } from "@/components/console/result-panel";
import { TracePanel } from "@/components/console/trace-panel";
import { fetchPreviewUrl, getStoredResult, getUpload, streamQuery, uploadScene } from "@/lib/api";
import { explainError, type Explained } from "@/lib/errors";
import {
  LANDING_SAMPLE_ID,
  loadSamples,
  needsBenchmarkMode,
  stageSample,
  type Sample,
} from "@/lib/samples";
import { SUGGESTED_QUESTIONS } from "@/lib/suggestions";
import { MODE_LABELS, MODE_ROLES, TASK_OPTIONS, type TaskChoice } from "@/lib/tasks";
import type { QueryResult, StagedFile, TraceStep, UploadMode, UploadResult } from "@/lib/types";

/** Formats a browser decodes itself; anything else is rendered by the server. */
const BROWSER_DECODES = /\.(png|jpe?g|webp|gif)$/i;

export default function ConsolePage() {
  return (
    <RequireAuth>
      <Suspense fallback={null}>
        <Console />
      </Suspense>
    </RequireAuth>
  );
}

function Console() {
  const router = useRouter();
  const params = useSearchParams();

  // -- step 1: imagery --------------------------------------------------------
  const [task, setTask] = useState<TaskChoice>("auto");
  const [mode, setMode] = useState<UploadMode>("single");
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [sampleLoading, setSampleLoading] = useState<string | null>(null);
  const [panes, setPanes] = useState<Pane[]>([]);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [boundMode, setBoundMode] = useState<UploadMode | null>(null);
  const [boundTitle, setBoundTitle] = useState<string | null>(null);
  const [binding, setBinding] = useState(false);
  const [bindError, setBindError] = useState<Explained | null>(null);
  const [imageryOpen, setImageryOpen] = useState(true);

  // -- step 2 and 3: question and answer -------------------------------------
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [answerStored, setAnswerStored] = useState(true);
  const [runError, setRunError] = useState<Explained | null>(null);
  const [steps, setSteps] = useState<TraceStep[]>([]);
  const [traceOpen, setTraceOpen] = useState(false);

  const inflight = useRef<AbortController | null>(null);
  /** Every blob: URL this page made, revoked when replaced or on unmount. */
  const blobUrls = useRef<string[]>([]);
  /** Bumped whenever the imagery changes, so late previews for old panes are dropped. */
  const generation = useRef(0);

  useEffect(
    () => () => {
      blobUrls.current.forEach(URL.revokeObjectURL);
      inflight.current?.abort();
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    loadSamples(controller.signal)
      .then(setSamples)
      .catch(() => setSamples([]));
    return () => controller.abort();
  }, []);

  /**
   * Show a bound upload: browser-decodable files straight from their bytes,
   * everything else (GeoTIFF) as the server's true-colour render.
   */
  const showUpload = useCallback((upload: UploadResult, staged: StagedFile[] = []) => {
    generation.current += 1;
    const gen = generation.current;
    blobUrls.current.forEach(URL.revokeObjectURL);
    blobUrls.current = [];

    const slots = MODE_ROLES[upload.mode];
    const next: Pane[] = upload.assets.map((asset, i) => {
      const file = staged[i]?.file;
      const local = file && BROWSER_DECODES.test(asset.filename) ? URL.createObjectURL(file) : null;
      if (local) blobUrls.current.push(local);
      return {
        meta: {
          sceneId: asset.filename,
          role: asset.role,
          sizeBytes: asset.size_bytes,
          contentType: asset.content_type,
          geo: asset.geo ?? null,
        },
        label: `${slots[i]?.label ?? asset.role}: ${asset.filename}`,
        preview: local ? { url: local, state: "ready" } : { url: null, state: "loading" },
      };
    });
    setPanes(next);

    upload.assets.forEach((asset, i) => {
      if (next[i].preview.state !== "loading") return;
      fetchPreviewUrl(upload.upload_id, asset.asset_id)
        .then((url) => {
          if (gen !== generation.current) return URL.revokeObjectURL(url);
          blobUrls.current.push(url);
          setPanes((prev) =>
            prev.map((p, j) => (j === i ? { ...p, preview: { url, state: "ready" } } : p)),
          );
        })
        .catch((error: unknown) => {
          if (gen !== generation.current) return;
          const message = explainError(error).detail;
          setPanes((prev) =>
            prev.map((p, j) =>
              j === i ? { ...p, preview: { url: null, state: "failed", message } } : p,
            ),
          );
        });
    });
  }, []);

  const resetRun = useCallback(() => {
    inflight.current?.abort();
    setBusy(false);
    setResult(null);
    setSteps([]);
    setRunError(null);
    setAnswerStored(true);
  }, []);

  const bindScene = useCallback(
    async (nextMode: UploadMode, files: StagedFile[], benchmarkMode: boolean, title?: string) => {
      resetRun();
      setBinding(true);
      setBindError(null);
      try {
        const upload = await uploadScene(nextMode, files, benchmarkMode);
        showUpload(upload, files);
        setUploadId(upload.upload_id);
        setBoundMode(upload.mode);
        setBoundTitle(title ?? files.map((f) => f.name).join(" and "));
        setImageryOpen(false);
      } catch (error) {
        setBindError(explainError(error));
      } finally {
        setBinding(false);
      }
    },
    [resetRun, showUpload],
  );

  const pickSample = useCallback(
    async (sample: Sample) => {
      setSampleLoading(sample.id);
      setBindError(null);
      const fixed = TASK_OPTIONS.find((o) => o.id === task)?.mode;
      if (fixed && fixed !== sample.mode) setTask("auto");
      setMode(sample.mode);
      try {
        const files = await stageSample(sample);
        await bindScene(
          sample.mode,
          files,
          needsBenchmarkMode(files.map((f) => f.name)),
          sample.title,
        );
        setQuestion(sample.question);
      } catch (error) {
        setBindError(explainError(error));
      } finally {
        setSampleLoading(null);
      }
    },
    [bindScene, task],
  );

  const reopen = useCallback(
    async (queryId: string) => {
      resetRun();
      try {
        const stored = await getStoredResult(queryId);
        setResult(stored.result);
        setSteps(stored.result.steps);
        setAnswerStored(stored.answerStored);
        if (stored.question) setQuestion(stored.question);
        if (stored.uploadId) {
          try {
            const upload = await getUpload(stored.uploadId);
            showUpload(upload);
            setUploadId(upload.upload_id);
            setBoundMode(upload.mode);
            setMode(upload.mode);
            setBoundTitle(upload.assets.map((a) => a.filename).join(" and "));
            setImageryOpen(false);
          } catch (error) {
            setBindError({
              ...explainError(error),
              title: "The imagery for this run is no longer stored",
            });
          }
        }
      } catch (error) {
        setRunError(explainError(error));
      }
    },
    [resetRun, showUpload],
  );

  // ?sample=<id> from the landing page, ?query=<id> from history. Handled once,
  // then dropped from the address bar so a refresh doesn't upload again.
  const handled = useRef(false);
  useEffect(() => {
    if (handled.current) return;
    const sampleId = params.get("sample");
    const queryId = params.get("query");
    let start: (() => void) | null = null;
    if (queryId) {
      start = () => void reopen(queryId);
    } else if (sampleId && samples) {
      const sample = samples.find((s) => s.id === sampleId && s.status === "ready");
      start = sample ? () => void pickSample(sample) : () => {};
    }
    if (!start) return;
    handled.current = true;
    router.replace("/console");
    // Deferred a tick: the work sets state, which belongs outside the effect body.
    window.setTimeout(start, 0);
  }, [params, samples, reopen, pickSample, router]);

  const runQuery = useCallback(
    async (text: string) => {
      inflight.current?.abort();
      const controller = new AbortController();
      inflight.current = controller;

      setBusy(true);
      setResult(null);
      setSteps([]);
      setRunError(null);
      setAnswerStored(true);

      const intent = TASK_OPTIONS.find((o) => o.id === task)?.intent ?? undefined;
      try {
        const final = await streamQuery(
          { query: text, upload_id: uploadId, ...(intent ? { intent } : {}) },
          // Steps arrive in graph order, so appending blindly is enough.
          { onStep: (step) => setSteps((prev) => [...prev, step]) },
          controller.signal,
        );
        setResult(final);
        setSteps(final.steps);
      } catch (error) {
        if (controller.signal.aborted) return;
        setRunError(explainError(error));
      } finally {
        if (!controller.signal.aborted) setBusy(false);
      }
    },
    [uploadId, task],
  );

  const bound = uploadId !== null && !imageryOpen;
  const step1: StepState = bound ? "done" : "current";
  const step2: StepState = !bound ? "upcoming" : result || busy ? "done" : "current";
  const step3: StepState = busy || result || runError ? "current" : "upcoming";
  const activeMode =
    task === "auto" ? mode : (TASK_OPTIONS.find((o) => o.id === task)?.mode ?? mode);
  const totalMs = steps.reduce((sum, s) => sum + (s.duration_ms ?? 0), 0);

  // The tour reads the console through this, fresh at every step.
  const { user } = useAuth();
  const tourActions = useRef<TourActions>({
    isBound: () => false,
    isBusy: () => false,
    hasResult: () => false,
    loadSample: async () => {},
    run: () => {},
  });
  useEffect(() => {
    tourActions.current = {
      isBound: () => bound,
      isBusy: () => busy,
      hasResult: () => result !== null || runError !== null,
      loadSample: async () => {
        const sample = samples?.find((s) => s.id === LANDING_SAMPLE_ID && s.status === "ready");
        if (sample) await pickSample(sample);
      },
      run: () => {
        const text = question.trim() || SUGGESTED_QUESTIONS[boundMode ?? activeMode][0];
        setQuestion(text);
        void runQuery(text);
      },
    };
  });
  const tour = useConsoleTour(user?.uid ?? null, tourActions);

  return (
    <div className="flex min-h-dvh flex-col bg-void lg:h-dvh lg:overflow-hidden">
      <TopBar busy={busy}>
        <TourHelpButton onClick={() => void tour.start(0)} />
      </TopBar>
      <TourWelcome
        open={tour.welcomeOpen}
        onStart={() => void tour.start(0)}
        onSkip={tour.skip}
      />

      <main className="flex min-h-0 flex-1 flex-col gap-3 p-3 lg:grid lg:grid-cols-[minmax(0,1fr)_minmax(380px,460px)]">
        <SceneCanvas
          panes={panes}
          mode={boundMode}
          overlays={result?.overlays ?? []}
          className="order-2 lg:order-none"
        />

        {/* The steps scroll as one column on desktop. On a phone the wrapper
            dissolves so the imagery can sit between step 1 and step 2. */}
        <div
          data-tour-scroll
          className="contents lg:flex lg:min-h-0 lg:flex-col lg:gap-3 lg:overflow-y-auto lg:pr-1"
        >
          <Step
            n={1}
            title="Choose imagery"
            state={step1}
            className="order-1 lg:order-none"
            data-tour="step-imagery"
            action={
              bound ? (
                <button
                  type="button"
                  onClick={() => setImageryOpen(true)}
                  className="rounded-sm text-[12.5px] text-signal underline-offset-4 hover:underline"
                >
                  Change imagery
                </button>
              ) : uploadId ? (
                <button
                  type="button"
                  onClick={() => setImageryOpen(false)}
                  className="rounded-sm text-[12.5px] text-text-dim underline-offset-4 hover:underline"
                >
                  Keep current imagery
                </button>
              ) : null
            }
            summary={
              bound ? (
                <p className="text-[13px] text-text-dim">
                  <span className="text-foreground">{boundTitle}</span>
                  {boundMode ? ` (${MODE_LABELS[boundMode].toLowerCase()})` : ""}
                </p>
              ) : undefined
            }
          >
            <div className="space-y-5">
              <div data-tour="task">
                <TaskPicker
                  task={task}
                  mode={mode}
                  onTaskChange={(next) => {
                    setTask(next);
                    const fixed = TASK_OPTIONS.find((o) => o.id === next)?.mode;
                    if (fixed) setMode(fixed);
                  }}
                  onModeChange={setMode}
                />
              </div>
              <div data-tour="upload" className="space-y-5">
                <SampleGallery
                  samples={samples}
                  mode={activeMode}
                  loadingId={sampleLoading}
                  onPick={pickSample}
                />
                <UploadPanel
                  key={activeMode}
                  mode={activeMode}
                  onBind={(m, files, benchmark) => bindScene(m, files, benchmark)}
                  binding={binding}
                />
              </div>
              {bindError ? (
                <div role="alert" className="rounded-sm border border-alarm/70 bg-raised p-3">
                  <p className="text-[13.5px] font-medium text-alarm">{bindError.title}</p>
                  <p className="mt-1 text-[13px] leading-relaxed text-text-dim">
                    {bindError.detail}
                  </p>
                </div>
              ) : null}
            </div>
          </Step>

          <Step
            n={2}
            title="Ask"
            state={step2}
            className="order-3 lg:order-none"
            data-tour="question"
          >
            <QuestionBox
              value={question}
              onChange={setQuestion}
              onRun={runQuery}
              mode={boundMode ?? activeMode}
              disabled={!bound}
              busy={busy}
            />
          </Step>

          <Step
            n={3}
            title="Read the answer"
            state={step3}
            className="order-4 lg:order-none"
            data-tour="answer"
          >
            <div className="space-y-4">
              {busy ? (
                <div data-tour="progress">
                  <PipelineProgress steps={steps} running={busy} />
                </div>
              ) : null}
              <div data-tour="result">
                <ResultPanel
                  result={result}
                  panes={panes}
                  busy={busy}
                  hasScene={bound}
                  error={runError}
                  answerStored={answerStored}
                />
              </div>
              {steps.length > 0 || busy ? (
                <div data-tour="trace">
                  <TracePanel
                    steps={steps}
                    open={traceOpen}
                    onOpenChange={setTraceOpen}
                    totalMs={totalMs}
                    running={busy}
                  />
                </div>
              ) : null}
            </div>
          </Step>
        </div>
      </main>
    </div>
  );
}
