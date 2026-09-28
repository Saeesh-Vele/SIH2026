"use client";

import { AlertTriangle, Download, LogIn } from "lucide-react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { FieldRow } from "@/components/frame";
import { InfoPopover, Term } from "@/components/console/info-popover";
import type { Pane } from "@/components/canvas/scene-canvas";
import type { Explained } from "@/lib/errors";
import { signInHref } from "@/lib/safe-next";
import { INTENT_LABELS } from "@/lib/suggestions";
import { TASK_LABELS } from "@/lib/tasks";
import { cn } from "@/lib/utils";
import type { MaskOverlay, QueryResult } from "@/lib/types";

/**
 * The fusion node reports the sentence it fed the VQA model under this label.
 * It gets its own block rather than a metrics row — it is a sentence, and the
 * row layout is built for short values.
 */
const SUMMARY_METRIC = "sensor summary";

/** What happened and what to do, for each way a run can end without an answer. */
const STATUS_COPY: Record<
  Exclude<QueryResult["status"], "ok">,
  { title: string; next: string }
> = {
  rejected: {
    title: "The imagery can’t answer this question",
    next: "Check the reason below, then change the imagery in step 1 or rephrase the question.",
  },
  unavailable: {
    title: "The model for this task isn’t running on this server",
    next: "The server couldn’t load the model, most often because no GPU is available to it at the moment. Try again later; the exact reason is below.",
  },
  failed: {
    title: "The run stopped before it finished",
    next: "Run it again. If it fails the same way, the trace below shows which step stopped it.",
  },
};

export function ResultPanel({
  result,
  panes,
  busy,
  hasScene,
  error,
  answerStored = true,
}: {
  result: QueryResult | null;
  /** The bound scenes, so fusion can show what it combined. */
  panes: Pane[];
  busy: boolean;
  hasScene: boolean;
  error: Explained | null;
  /** False when reopening a run stored before answers were kept. */
  answerStored?: boolean;
}) {
  if (error) return <ErrorState error={error} />;
  if (busy) return null;
  if (!result) {
    return (
      <p className="text-[13px] leading-relaxed text-muted-foreground">
        {hasScene
          ? "Your answer will appear here, with how confident the model was and every step it took."
          : "Choose imagery and ask a question; the answer appears here."}
      </p>
    );
  }
  return <Answer result={result} panes={panes} answerStored={answerStored} />;
}

function ErrorState({ error }: { error: Explained }) {
  return (
    <div role="alert" className="space-y-2 rounded-sm border border-alarm/70 bg-raised p-3">
      <p className="flex items-start gap-2 text-[14px] font-medium text-alarm">
        <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
        {error.title}
      </p>
      <p className="text-[13px] leading-relaxed text-text-dim">{error.detail}</p>
      {error.signIn ? (
        <Link
          href={signInHref("/console")}
          className="inline-flex items-center gap-1.5 rounded-sm text-[13px] text-signal underline-offset-4 hover:underline"
        >
          <LogIn className="size-3.5" aria-hidden />
          Sign in again
        </Link>
      ) : null}
    </div>
  );
}

function DegradedBadge({ reason }: { reason: string | null }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <Badge
        variant="outline"
        className="gap-1 rounded-sm border-caution bg-caution/15 px-1.5 font-mono text-[11px] text-caution"
      >
        <AlertTriangle className="size-3" aria-hidden />
        Degraded mode — CPU fallback
      </Badge>
      <InfoPopover label="What does Degraded mode mean?">
        <p>
          The main model couldn&rsquo;t run on this server, so a smaller CPU model answered
          instead. Treat this answer as a rough guide: it is not from the fine-tuned model, and
          its confidence wasn&rsquo;t measured.
        </p>
        {reason ? <p className="mt-2 text-[12px] text-muted-foreground">Reason: {reason}</p> : null}
      </InfoPopover>
    </div>
  );
}

function Answer({
  result,
  panes,
  answerStored,
}: {
  result: QueryResult;
  panes: Pane[];
  answerStored: boolean;
}) {
  const failed = result.status !== "ok" ? result.status : null;
  const ok = failed === null;
  // Null only on a degraded (CPU-fallback) run: there is no measured value to
  // draw, so the bar is omitted rather than drawn at zero.
  const confidence = result.confidence;
  const summary = result.metrics.find((m) => m.label === SUMMARY_METRIC)?.value ?? null;
  const rowMetrics = result.metrics.filter((m) => m.label !== SUMMARY_METRIC);
  const tone =
    confidence === null
      ? null
      : confidence >= 0.8
        ? "signal"
        : confidence >= 0.6
          ? "caution"
          : "alarm";

  return (
    <div className="space-y-5">
      {/* 1. The answer — or, when there is none, what happened and what to do. */}
      {ok ? (
        <div className="space-y-2.5">
          {result.degraded ? <DegradedBadge reason={result.degradedReason} /> : null}
          {answerStored ? (
            <p className="text-[15.5px] leading-[1.65] text-foreground">{result.answer}</p>
          ) : (
            <p className="text-[13px] leading-relaxed text-muted-foreground">
              This run was saved before answers were stored, so only its trace is available.
            </p>
          )}
        </div>
      ) : (
        <div role="alert" className="space-y-2 rounded-sm border border-caution/70 bg-raised p-3">
          <p className="flex items-start gap-2 text-[14px] font-medium leading-snug text-caution">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
            {STATUS_COPY[failed].title}
          </p>
          <p className="text-[13px] leading-relaxed text-text">{STATUS_COPY[failed].next}</p>
          {result.error ?? result.answer ? (
            <p className="font-mono text-[11.5px] leading-relaxed text-muted-foreground">
              {result.error ?? result.answer}
            </p>
          ) : null}
        </div>
      )}

      {/* 2. How sure it was. */}
      {ok && answerStored && confidence === null ? (
        <div className="flex flex-wrap items-baseline justify-between gap-2 text-[12.5px]">
          <span className="text-muted-foreground">
            <Term term="Confidence">
              How sure the model was of its own words: the average probability it gave each token
              of the answer. Not a measure of whether the answer is correct.
            </Term>
          </span>
          <span className="text-caution">
            not measured — CPU fallback
            {result.degradedReason ? ` (${result.degradedReason})` : ""}
          </span>
        </div>
      ) : null}

      {ok && answerStored && confidence !== null ? (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between text-[12.5px]">
            <span className="text-muted-foreground">
              <Term term="Confidence">
                How sure the model was of its own words: the average probability it gave each
                token of the answer. Not a measure of whether the answer is correct.
              </Term>
            </span>
            <span
              className={cn(
                "font-mono",
                tone === "signal" && "text-signal",
                tone === "caution" && "text-caution",
                tone === "alarm" && "text-alarm",
              )}
            >
              {confidence.toFixed(2)}
            </span>
          </div>
          <Progress
            value={Math.round(confidence * 100)}
            aria-label={`Confidence ${confidence.toFixed(2)}`}
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

      {/* 3. The evidence behind it. */}
      {rowMetrics.length > 0 ? (
        <div className="space-y-1 border-t border-rule pt-3">
          {rowMetrics.map((metric) => (
            <FieldRow key={metric.label} name={metric.label} value={metric.value} />
          ))}
        </div>
      ) : null}

      {ok ? <TaskReport result={result} panes={panes} summary={summary} /> : null}

      <div className="flex flex-wrap gap-1.5 border-t border-rule pt-3">
        {result.intent ? <Chip>{INTENT_LABELS[result.intent]}</Chip> : null}
        <Chip>{TASK_LABELS[result.taskSelected]}</Chip>
        {/* A model is named only when it produced the answer shown. */}
        {(ok ? result.modelsUsed : []).map((model) => (
          <Chip key={model} mono>
            {model}
          </Chip>
        ))}
      </div>

      {ok && answerStored ? <DownloadReport result={result} panes={panes} /> : null}
    </div>
  );
}

function Chip({ children, mono = false }: { children: React.ReactNode; mono?: boolean }) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "max-w-full truncate rounded-sm border-rule bg-raised px-1.5 text-[11px] font-normal text-text-dim",
        mono && "font-mono",
      )}
    >
      {children}
    </Badge>
  );
}

function DownloadReport({ result, panes }: { result: QueryResult; panes: Pane[] }) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => {
        const reportData = {
          report_title: "SatQuery AI Remote-Sensing Analysis Report",
          generated_at: new Date().toISOString(),
          query_id: result.queryId,
          trace_id: result.traceId,
          status: result.status,
          task_selected: result.taskSelected,
          intent: result.intent,
          answer: result.answer,
          // null on a degraded run, alongside the flag that explains it.
          confidence_score: result.confidence,
          degraded: result.degraded,
          degraded_reason: result.degradedReason,
          models_used: result.modelsUsed,
          bound_scenes: panes.map((p) => ({
            scene_id: p.meta.sceneId,
            role: p.meta.role,
            size_bytes: p.meta.sizeBytes,
            content_type: p.meta.contentType,
          })),
          visual_evidence: result.overlays,
          metrics: result.metrics,
        };
        const blob = new Blob([JSON.stringify(reportData, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `satquery-report-${result.queryId.slice(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      }}
      className="h-8 w-full gap-2 rounded-sm border-rule-strong bg-raised text-[12.5px] text-foreground hover:border-signal hover:bg-raised hover:text-signal"
    >
      <Download className="size-3.5" aria-hidden />
      Download report (.json)
    </Button>
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
    <div className="space-y-2 border-t border-rule pt-3">
      <p className="text-[12.5px] text-text-dim">{title}</p>
      {children}
    </div>
  );
}

function OverlayList({ result }: { result: QueryResult }) {
  if (result.overlays.length === 0) return null;
  return (
    <Section
      title={`${result.overlays.length} region${result.overlays.length > 1 ? "s" : ""} marked on the image`}
    >
      <ul className="divide-y divide-rule border border-rule">
        {result.overlays.map((overlay) => (
          <li
            key={overlay.id}
            className="flex items-baseline justify-between gap-3 px-2 py-1 font-mono text-[11.5px]"
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
      <Section title="Changed areas">
        <p className="text-[13px] leading-relaxed text-muted-foreground">
          No area differed enough between the two dates to be marked, so the model read them as
          unchanged.
        </p>
      </Section>
    );
  }

  const covered = masks.reduce((sum, mask) => sum + (mask.area ?? 0), 0);

  return (
    <Section
      title={`${masks.length} changed area${masks.length > 1 ? "s" : ""} marked on the image${
        covered > 0 ? `, ${(covered * 100).toFixed(1)}% of the frame` : ""
      }`}
    >
      <ul className="divide-y divide-rule border border-rule">
        {masks.map((mask) => (
          <li key={mask.id} className="flex items-baseline gap-3 px-2 py-1 font-mono text-[11.5px]">
            <span className="shrink-0 text-muted-foreground">{mask.id}</span>
            <span className="truncate text-text-dim">{mask.label}</span>
            <span className="h-px min-w-3 flex-1 translate-y-[-3px] bg-rule" aria-hidden />
            {mask.area !== undefined ? (
              <span className="shrink-0 text-muted-foreground">
                {(mask.area * 100).toFixed(1)}%
              </span>
            ) : null}
            <span className="shrink-0 text-text-dim">{mask.confidence.toFixed(2)}</span>
          </li>
        ))}
      </ul>
      <p className="text-[12px] leading-relaxed text-muted-foreground">
        Each area is the rectangle around a cluster of image patches that changed, not a traced
        outline.
      </p>
    </Section>
  );
}

/** The two inputs fusion combined, labelled by modality, with what it made of them. */
function FusionReport({ panes, summary }: { panes: Pane[]; summary: string | null }) {
  const optical = panes.find((pane) => pane.meta.role === "optical");
  const sar = panes.find((pane) => pane.meta.role === "sar");

  return (
    <Section title="What was combined">
      {optical && sar ? (
        <div className="grid grid-cols-2 gap-px border border-rule bg-rule">
          {(
            [
              ["Optical", optical],
              ["SAR", sar],
            ] as const
          ).map(([label, pane]) => (
            <figure key={label} className="bg-void">
              <div className="relative aspect-square w-full overflow-hidden">
                {pane.preview.url ? (
                  // eslint-disable-next-line @next/next/no-img-element -- a blob: URL of the user's own file
                  <img
                    src={pane.preview.url}
                    alt={`${label} input: ${pane.meta.sceneId}`}
                    className="absolute inset-0 size-full object-cover [image-rendering:pixelated]"
                  />
                ) : (
                  <span className="texture-graticule absolute inset-0 grid place-items-center px-2 text-center text-[11px] text-muted-foreground">
                    No preview
                  </span>
                )}
                <span className="absolute left-1.5 top-1.5 rounded-sm bg-void/85 px-1 py-0.5 text-[10.5px] text-signal">
                  {label}
                </span>
              </div>
              <figcaption className="truncate px-1.5 py-1 font-mono text-[10.5px] text-muted-foreground">
                {pane.meta.sceneId}
              </figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <p className="text-[13px] leading-relaxed text-muted-foreground">
          The two images are no longer on screen, so they can&rsquo;t be shown beside the answer.
        </p>
      )}

      {summary ? (
        <div className="rounded-sm border border-rule bg-raised p-2.5">
          <p className="text-[12px] text-muted-foreground">What the encoders told the model</p>
          <p className="mt-1 text-[13px] leading-relaxed text-text-dim">{summary}</p>
        </div>
      ) : null}
    </Section>
  );
}
