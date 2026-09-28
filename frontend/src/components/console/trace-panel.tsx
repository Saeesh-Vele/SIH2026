"use client";

import { ChevronDown } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { StatusDot } from "@/components/status-dot";
import { formatDuration } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TraceStep, TraceStepStatus } from "@/lib/types";

const TONE: Record<TraceStepStatus, "signal" | "caution" | "muted" | "alarm"> = {
  complete: "signal",
  running: "caution",
  pending: "muted",
  failed: "alarm",
};

/**
 * Docked, collapsible feed of what the agent did. Steps append as they land, so
 * the live stream from /query can push into the same list unchanged.
 */
export function TracePanel({
  steps,
  open,
  onOpenChange,
  totalMs,
  running = false,
}: {
  steps: TraceStep[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  totalMs: number;
  /** True while the controller is still working, between steps. */
  running?: boolean;
}) {
  const active = running || steps.some((s) => s.status === "running");

  return (
    <section className="flex flex-col rounded-sm border border-rule bg-panel">
      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
        className="flex h-10 shrink-0 items-center gap-2 rounded-sm px-3 text-left transition-colors hover:bg-raised"
      >
        <ChevronDown
          className={cn(
            "size-3.5 shrink-0 text-muted-foreground transition-transform",
            !open && "-rotate-90",
          )}
        />
        <span className="text-[13px] text-foreground">Trace: every step it took</span>
        <span className="ml-auto flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
          {steps.length > 0 ? (
            <>
              <StatusDot tone={active ? "caution" : "signal"} pulse={active} />
              {steps.length} steps
              {totalMs > 0 ? (
                <>
                  <span className="text-rule-strong" aria-hidden>│</span>
                  {formatDuration(totalMs)}
                </>
              ) : null}
            </>
          ) : (
            "none yet"
          )}
        </span>
      </button>

      {open ? (
        <ScrollArea className="max-h-80 border-t border-rule">
          {steps.length === 0 ? (
            <p className="p-3 text-[12.5px] leading-relaxed text-muted-foreground">
              Steps appear here as SatQuery works: what it decided you were asking, which model it
              ran, and how long each stage took. Every run is saved with this record.
            </p>
          ) : (
            <ol className="p-3">
              {steps.map((step, i) => (
                <li key={`${i}-${step.label}`} className="relative flex gap-3 pb-3 last:pb-0">
                  <div className="flex flex-col items-center pt-1">
                    <StatusDot
                      tone={TONE[step.status]}
                      pulse={step.status === "running"}
                    />
                    {i < steps.length - 1 ? (
                      <span className="mt-1 w-px flex-1 bg-rule" aria-hidden />
                    ) : null}
                  </div>
                  <div className="min-w-0 flex-1 space-y-0.5">
                    <div className="flex items-baseline gap-2">
                      <span className="font-mono text-[12px] text-foreground">
                        {step.label}
                      </span>
                      {step.duration_ms ? (
                        <span className="ml-auto shrink-0 font-mono text-[11px] text-muted-foreground">
                          {formatDuration(step.duration_ms)}
                        </span>
                      ) : null}
                    </div>
                    {step.detail ? (
                      <p className="break-words font-mono text-[11.5px] leading-relaxed text-muted-foreground">
                        {step.detail}
                      </p>
                    ) : null}
                  </div>
                </li>
              ))}
            </ol>
          )}
        </ScrollArea>
      ) : null}
    </section>
  );
}
