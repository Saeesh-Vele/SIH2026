import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export type StepState = "current" | "done" | "upcoming";

/**
 * One of the console's three numbered steps. The number is real sequence —
 * imagery, then question, then answer — so it is shown; a finished step says
 * so with a check and can collapse to a one-line summary.
 */
export function Step({
  n,
  title,
  state,
  summary,
  action,
  children,
  className,
  ...props
}: {
  n: number;
  title: string;
  state: StepState;
  /** Shown instead of the body when the step is done and collapsed. */
  summary?: React.ReactNode;
  /** Top-right control, e.g. "Change imagery". */
  action?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
} & Omit<React.ComponentProps<"section">, "title">) {
  const headingId = `step-${n}-title`;
  return (
    <section
      aria-labelledby={headingId}
      aria-current={state === "current" ? "step" : undefined}
      className={cn(
        "rounded-(--radius-surface) border bg-panel",
        state === "current" ? "border-rule-strong" : "border-rule",
        className,
      )}
      {...props}
    >
      <header className="flex items-center gap-3 px-4 pt-4">
        <span
          aria-hidden
          className={cn(
            "grid size-6 shrink-0 place-items-center rounded-full border font-mono text-[11px]",
            state === "done" && "border-signal bg-signal text-void",
            state === "current" && "border-signal text-signal",
            state === "upcoming" && "border-rule-strong text-muted-foreground",
          )}
        >
          {state === "done" ? <Check className="size-3.5" strokeWidth={3} /> : n}
        </span>
        <h2
          id={headingId}
          className={cn(
            "text-[15px] font-semibold",
            state === "upcoming" ? "text-muted-foreground" : "text-foreground",
          )}
        >
          <span className="sr-only">
            Step {n}
            {state === "done" ? ", done" : state === "current" ? ", current" : ""}:{" "}
          </span>
          {title}
        </h2>
        {action ? <div className="ml-auto">{action}</div> : null}
      </header>
      <div className="p-4 pt-3">{summary ?? children}</div>
    </section>
  );
}
