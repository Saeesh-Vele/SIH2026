export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={`inline-flex items-baseline gap-1.5 ${className ?? ""}`}>
      <span className="text-[15px] font-semibold tracking-tight text-foreground">SatQuery</span>
      <span className="font-mono text-[10px] text-signal">AI</span>
    </span>
  );
}
