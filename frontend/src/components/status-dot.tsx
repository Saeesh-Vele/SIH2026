import { cn } from "@/lib/utils";

type Tone = "signal" | "caution" | "muted" | "alarm";

const TONE_CLASS: Record<Tone, string> = {
  signal: "bg-signal",
  caution: "bg-caution",
  muted: "bg-rule-strong",
  alarm: "bg-alarm",
};

export function StatusDot({
  tone = "muted",
  pulse = false,
  className,
}: {
  tone?: Tone;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block size-1.5 shrink-0 rounded-full",
        TONE_CLASS[tone],
        pulse && "animate-signal-pulse",
        className,
      )}
    />
  );
}
