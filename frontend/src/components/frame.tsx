import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * A schematic frame: hairline rectangle with its label notched into the top
 * border, the way a component is labelled on an engineering drawing. Used
 * instead of a drop-shadowed card so panels read as instrument bays.
 */
export function Frame({
  label,
  aside,
  className,
  bodyClassName,
  children,
  ...props
}: React.ComponentProps<"section"> & {
  label: string;
  aside?: React.ReactNode;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cn("relative flex min-h-0 flex-col border border-rule bg-panel", className)}
      {...props}
    >
      <span className="pointer-events-none absolute -top-[7px] left-3 z-10 bg-panel px-1.5 font-mono text-[10px] leading-none tracking-[0.14em] text-text-dim">
        {label}
      </span>
      {aside ? (
        <span className="pointer-events-none absolute -top-[7px] right-3 z-10 bg-panel px-1.5 font-mono text-[10px] leading-none text-muted-foreground">
          {aside}
        </span>
      ) : null}
      <div className={cn("flex min-h-0 flex-1 flex-col pt-3", bodyClassName)}>{children}</div>
    </section>
  );
}

/** Key/value row in mono, for scene metadata and trace parameters. */
export function FieldRow({
  name,
  value,
  className,
}: {
  name: string;
  value: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-baseline gap-3 font-mono text-[11px]", className)}>
      <span className="shrink-0 text-muted-foreground">{name}</span>
      <span className="h-px min-w-3 flex-1 translate-y-[-3px] bg-rule" aria-hidden />
      <span className="shrink-0 text-text-dim">{value}</span>
    </div>
  );
}
