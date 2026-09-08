"use client";

import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Frame, FieldRow } from "@/components/frame";
import { SyntheticRaster } from "@/components/canvas/synthetic-raster";
import type { Pane } from "@/components/canvas/scene-canvas";
import { INTENT_CODES, INTENT_LABELS } from "@/lib/suggestions";
import { TASK_LABELS } from "@/lib/tasks";
import { cn } from "@/lib/utils";
import type { MaskOverlay, QueryResult } from "@/lib/types";

/**
 * The fusion node reports the sentence it fed the VQA model under this label.
 * It gets its own block rather than a metrics row — it is a sentence, and the
 * row layout is built for short values.
 */
const SUMMARY_METRIC = "sensor summary";

/** Headline shown above the explanation when a run ends without an answer. */
const STATUS_HEADLINE: Record<Exclude<QueryResult["status"], "ok">, string> = {
  rejected: "The bound scene does not match this question",
  unavailable: "That model is not available here",
  failed: "The run did not finish",
};

export function ResultPanel({
  result,
  panes,
  busy,
  hasScene,
  transportError,
}: {
  result: QueryResult | null;
  /** The bound scenes, so fusion can show what it combined. */
  panes: Pane[];
  busy: boolean;
  hasScene: boolean;
  transportError: string | null;
}) {
  return (
    <Frame
      label="RESULT"
      aside={result ? result.queryId.slice(0, 8) : undefined}
      className="min-h-0 flex-1"
      bodyClassName="min-h-0"
    >
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-4 p-3">
          {transportError ? <TransportError message={transportError} /> : null}
          {busy && !transportError ? <RunningState /> : null}
          {!busy && !result && !transportError ? <EmptyState hasScene={hasScene} /> : null}
          {!busy && result && !transportError ? (
            <Answer result={result} panes={panes} />
          ) : null}
        </div>
      </ScrollArea>
    </Frame>
  );
}

function TransportError({ message }: { message: string }) {
  return (
    <div className="space-y-2 border border-alarm bg-raised p-3">
      <p className="flex items-center gap-2 text-[13px] text-alarm">
        <AlertTriangle className="size-3.5 shrink-0" />
        Could not reach the controller
      </p>
      <p className="font-mono text-[11px] leading-relaxed text-muted-foreground">{message}</p>
      <p className="font-mono text-[11px] leading-relaxed text-muted-foreground">
        Start the backend with{" "}
        <code className="text-text-dim">uvicorn app.main:app --reload</code> from{" "}
        <code className="text-text-dim">backend/</code>.
      </p>
    </div>
  );
}

function RunningState() {
  return (
    <p className="font-mono text-[11px] text-caution">
      Controller running — steps are landing in the trace below.
    </p>
  );
}

function EmptyState({ hasScene }: { hasScene: boolean }) {
  return (
    <p className="max-w-[42ch] text-[13px] leading-relaxed text-muted-foreground">
      {hasScene
        ? "Ask a question above. The controller classifies it, checks it against the bound scene, and runs the model that fits — every step shows up in the trace below."
        : "Bind a scene on the left to start asking questions."}
    </p>
  );
}

function Answer({ result, panes }: { result: QueryResult; panes: Pane[] }) {
  const failed = result.status !== "ok" ? result.status : null;
  const ok = failed === null;
  const confidencePct = Math.round(result.confidence * 100);
  const summary = result.metrics.find((m) => m.label === SUMMARY_METRIC)?.value ?? null;
  const rowMetrics = result.metrics.filter((m) => m.label !== SUMMARY_METRIC);
  const tone =
    result.confidence >= 0.8 ? "signal" : result.confidence >= 0.6 ? "caution" : "alarm";

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        {result.intent ? (
          <Badge
            variant="outline"
            className="gap-1.5 rounded-sm border-signal-deep bg-signal-deep/40 px-1.5 font-mono text-[10px] text-signal"
          >
            {INTENT_CODES[result.intent]}
            <span className="text-signal/60">{INTENT_LABELS[result.intent]}</span>
          </Badge>
        ) : null}
        <Badge
          variant="outline"
          className="rounded-sm border-rule bg-raised px-1.5 font-mono text-[10px] font-normal text-muted-foreground"
        >
          {TASK_LABELS[result.taskSelected]}
        </Badge>
        {result.modelsUsed.map((model) => (
          <Badge
            key={model}
            variant="outline"
            className="rounded-sm border-rule bg-raised px-1.5 font-mono text-[10px] font-normal text-muted-foreground"
          >
            {model}
          </Badge>
        ))}
      </div>

      {ok ? (
        <p className="max-w-[62ch] text-[13.5px] leading-[1.65] text-foreground">
          {result.answer}
        </p>
      ) : (
        <div className="space-y-2 border border-caution bg-raised p-3">
          <p className="flex items-start gap-2 text-[13px] leading-snug text-caution">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
            {failed ? STATUS_HEADLINE[failed] : null}
          </p>
          <p className="max-w-[58ch] font-mono text-[11px] leading-relaxed text-text-dim">
            {result.error ?? result.answer}
          </p>
        </div>
      )}

      {ok ? (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between font-mono text-[10px]">
            <span className="text-muted-foreground">
              confidence
              <span className="ml-1.5 text-rule-strong">mean token probability</span>
            </span>
            <span
              className={cn(
                tone === "signal" && "text-signal",
                tone === "caution" && "text-caution",
                tone === "alarm" && "text-alarm",
              )}
            >
              {result.confidence.toFixed(2)}
            </span>
          </div>
          <Progress
            value={confidencePct}
            className="h-1 rounded-none bg-raised"
            indicatorClassName={cn(
              "rounded-none",
              tone === "signal" && "bg-signal",
              tone === "caution" && "bg-caution",
              tone === "alarm" && "bg-alarm",
            )}
          />
        </div>
      ) : null}

      {rowMetrics.length > 0 ? (
        <div className="space-y-1 border-t border-rule pt-3">
          {rowMetrics.map((metric) => (
            <FieldRow key={metric.label} name={metric.label} value={metric.value} />
          ))}
        </div>
      ) : null}

      {ok ? <TaskReport result={result} panes={panes} summary={summary} /> : null}
    </>
  );
}

/**
 * Each task leaves behind a different kind of evidence, so each gets its own
 * readout: boxes for VQA, masks plus their share of the scene for change
 * detection, and the two inputs side by side for fusion, which produces a
 * described scene rather than geometry.
 */
function TaskReport({
  result,
  panes,
  summary,
}: {
  result: QueryResult;
  panes: Pane[];
  summary: string | null;
}) {
  if (result.taskSelected === "optical_sar_fusion") {
    return <FusionReport panes={panes} summary={summary} />;
  }
  if (result.taskSelected === "change_detection") {
    return <ChangeReport masks={result.overlays.filter((o) => o.kind === "mask")} />;
  }
  return <OverlayList result={result} />;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5 border-t border-rule pt-3">
      <p className="font-mono text-[10px] text-muted-foreground">{title}</p>
      {children}
    </div>
  );
}

function OverlayList({ result }: { result: QueryResult }) {
  if (result.overlays.length === 0) return null;
  return (
    <Section
      title={`${result.overlays.length} overlay${result.overlays.length > 1 ? "s" : ""} drawn on the scene`}
    >
      <ul className="divide-y divide-rule border border-rule">
        {result.overlays.map((overlay) => (
          <li
            key={overlay.id}
            className="flex items-baseline justify-between gap-3 px-2 py-1 font-mono text-[11px]"
          >
            <span className="truncate text-text-dim">{overlay.label}</span>
            <span className="shrink-0 text-muted-foreground">
              {overlay.kind === "box" ? "box" : "mask"} {overlay.confidence.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </Section>
  );
}

/** Changed regions, strongest first, each with the share of the scene it covers. */
function ChangeReport({ masks }: { masks: MaskOverlay[] }) {
  if (masks.length === 0) {
    return (
      <Section title="change mask">
        <p className="max-w-[52ch] text-[12.5px] leading-relaxed text-muted-foreground">
          No region passed the difference threshold — the model read the two captures as
          unchanged. Lower <code className="text-text-dim">diff_threshold</code> in
          model_config.yaml to report weaker differences.
        </p>
      </Section>
    );
  }

  const covered = masks.reduce((sum, mask) => sum + (mask.area ?? 0), 0);

  return (
    <Section
      title={`${masks.length} changed region${masks.length > 1 ? "s" : ""} masked on the scene${
        covered > 0 ? ` · ${(covered * 100).toFixed(1)}% of the frame` : ""
      }`}
    >
      <ul className="divide-y divide-rule border border-rule">
        {masks.map((mask) => (
          <li key={mask.id} className="flex items-baseline gap-3 px-2 py-1 font-mono text-[11px]">
            <span className="shrink-0 text-rule-strong">{mask.id}</span>
            <span className="truncate text-text-dim">{mask.label}</span>
            <span className="h-px min-w-3 flex-1 translate-y-[-3px] bg-rule" aria-hidden />
            {mask.area !== undefined ? (
              <span className="shrink-0 text-muted-foreground">
                {(mask.area * 100).toFixed(1)}%
              </span>
            ) : null}
            <span className="shrink-0 text-caution">{mask.confidence.toFixed(2)}</span>
          </li>
        ))}
      </ul>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Regions come from the vision tower&rsquo;s patch grid, so each mask is the bounding
        rectangle of a changed cluster rather than a traced outline.
      </p>
    </Section>
  );
}

/** The two inputs fusion combined, labelled by modality, with what it made of them. */
function FusionReport({ panes, summary }: { panes: Pane[]; summary: string | null }) {
  const optical = panes.find((pane) => pane.meta.role === "optical");
  const sar = panes.find((pane) => pane.meta.role === "sar");

  return (
    <Section title="fused inputs">
      {optical && sar ? (
        <div className="grid grid-cols-2 gap-px border border-rule bg-rule">
          {(
            [
              ["OPTICAL", optical],
              ["SAR", sar],
            ] as const
          ).map(([label, pane]) => (
            <figure key={label} className="relative bg-void">
              <div className="relative aspect-square w-full overflow-hidden">
                {pane.previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element -- a blob: URL from the user's own file
                  <img
                    src={pane.previewUrl}
                    alt={`${label} input: ${pane.meta.sceneId}`}
                    className="absolute inset-0 size-full object-cover"
                  />
                ) : (
                  <SyntheticRaster
                    seed={pane.seed}
                    tint={pane.tint}
                    className="absolute inset-0 size-full"
                  />
                )}
                <div className="texture-scanline absolute inset-0" />
                <span className="absolute left-1.5 top-1.5 bg-void/85 px-1 py-0.5 font-mono text-[9px] tracking-[0.14em] text-signal">
                  {label}
                </span>
              </div>
              <figcaption className="truncate px-1.5 py-1 font-mono text-[10px] text-muted-foreground">
                {pane.meta.sceneId}
              </figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <p className="text-[12.5px] leading-relaxed text-muted-foreground">
          The bound pair is no longer on screen, so the inputs cannot be shown alongside the
          answer.
        </p>
      )}

      {summary ? (
        <div className="border border-rule bg-raised p-2">
          <p className="font-mono text-[9px] tracking-[0.14em] text-muted-foreground">
            WHAT THE ENCODER TOLD THE MODEL
          </p>
          <p className="mt-1 max-w-[52ch] text-[12.5px] leading-relaxed text-text-dim">
            {summary}
          </p>
        </div>
      ) : null}
    </Section>
  );
}
