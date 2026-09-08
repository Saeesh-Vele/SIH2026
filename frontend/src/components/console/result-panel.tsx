"use client";

import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Frame, FieldRow } from "@/components/frame";
import { INTENT_CODES, INTENT_LABELS } from "@/lib/suggestions";
import { TASK_LABELS } from "@/lib/tasks";
import { cn } from "@/lib/utils";
import type { QueryResult } from "@/lib/types";

/** Headline shown above the explanation when a run ends without an answer. */
const STATUS_HEADLINE: Record<Exclude<QueryResult["status"], "ok">, string> = {
  rejected: "The bound scene does not match this question",
  unavailable: "That model is not available here",
  failed: "The run did not finish",
};

export function ResultPanel({
  result,
  busy,
  hasScene,
  transportError,
}: {
  result: QueryResult | null;
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
          {!busy && result && !transportError ? <Answer result={result} /> : null}
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

function Answer({ result }: { result: QueryResult }) {
  const failed = result.status !== "ok" ? result.status : null;
  const ok = failed === null;
  const confidencePct = Math.round(result.confidence * 100);
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

      {result.metrics.length > 0 ? (
        <div className="space-y-1 border-t border-rule pt-3">
          {result.metrics.map((metric) => (
            <FieldRow key={metric.label} name={metric.label} value={metric.value} />
          ))}
        </div>
      ) : null}

      {result.overlays.length > 0 ? (
        <div className="space-y-1.5 border-t border-rule pt-3">
          <p className="font-mono text-[10px] text-muted-foreground">
            {result.overlays.length} overlay
            {result.overlays.length > 1 ? "s" : ""} drawn on the scene
          </p>
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
        </div>
      ) : null}
    </>
  );
}
