"use client";

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

/**
 * A short explanation behind a small, keyboard-reachable trigger. Used for the
 * handful of terms a first-time visitor will not know, and for "what does
 * Degraded mean?" — one or two sentences, never a manual.
 */
export function InfoPopover({
  label,
  children,
  className,
  triggerText = "What does this mean?",
}: {
  /** Accessible name, e.g. "What is SAR?" */
  label: string;
  children: React.ReactNode;
  className?: string;
  triggerText?: string;
}) {
  return (
    <Popover>
      <PopoverTrigger
        aria-label={label}
        className={cn(
          "rounded-sm text-[12px] text-text-dim underline decoration-dotted underline-offset-4 transition-colors hover:text-signal",
          className,
        )}
      >
        {triggerText}
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-72 rounded-sm border-rule-strong bg-raised p-3 text-[13px] leading-relaxed text-text"
      >
        {children}
      </PopoverContent>
    </Popover>
  );
}

/** A term with a dotted underline that opens its definition. */
export function Term({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <InfoPopover label={`What is ${term}?`} triggerText={term} className="text-[length:inherit] text-inherit">
      <p className="font-medium text-foreground">{term}</p>
      <p className="mt-1 text-text-dim">{children}</p>
    </InfoPopover>
  );
}
