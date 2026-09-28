"use client";

import { cn } from "@/lib/utils";
import type { Overlay } from "@/lib/types";

/**
 * Overlays live in normalised (0–1) scene space so geometry survives any canvas
 * size. Masks are drawn in SVG, which polygons need; boxes are HTML, so their
 * reticle ticks stay square whatever the panel's aspect ratio.
 */
export function OverlayLayer({
  overlays,
  selectedId,
  onSelect,
}: {
  overlays: Overlay[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const masks = overlays.filter((o) => o.kind === "mask");
  const boxes = overlays.filter((o) => o.kind === "box");

  return (
    <>
      {masks.length > 0 ? (
        <svg
          className="absolute inset-0 size-full"
          viewBox="0 0 1000 1000"
          preserveAspectRatio="none"
        >
          {masks.map((mask) => {
            const active = mask.id === selectedId;
            return (
              <polygon
                key={mask.id}
                points={mask.polygon.map(([x, y]) => `${x * 1000},${y * 1000}`).join(" ")}
                fill={active ? "rgba(34, 201, 192, 0.22)" : "rgba(34, 201, 192, 0.12)"}
                stroke={active ? "var(--signal)" : "rgba(34, 201, 192, 0.6)"}
                strokeWidth={active ? 2.5 : 1.5}
                vectorEffect="non-scaling-stroke"
                className="cursor-pointer"
                onClick={() => onSelect(active ? null : mask.id)}
              />
            );
          })}
        </svg>
      ) : null}

      {boxes.map((box) => {
        const active = box.id === selectedId;
        return (
          <button
            key={box.id}
            type="button"
            onClick={() => onSelect(active ? null : box.id)}
            style={{
              left: `${box.x * 100}%`,
              top: `${box.y * 100}%`,
              width: `${box.w * 100}%`,
              height: `${box.h * 100}%`,
            }}
            className={cn(
              "absolute cursor-pointer border",
              active
                ? "border-signal bg-signal/15"
                : "border-signal/60 hover:border-signal",
            )}
          >
            {/* Corner ticks read as a reticle rather than a plain rectangle. */}
            {(
              [
                "-top-px -left-px border-l-2 border-t-2",
                "-top-px -right-px border-r-2 border-t-2",
                "-bottom-px -left-px border-b-2 border-l-2",
                "-bottom-px -right-px border-b-2 border-r-2",
              ] as const
            ).map((position) => (
              <span
                key={position}
                aria-hidden
                className={cn(
                  "absolute size-2",
                  position,
                  active ? "border-signal" : "border-signal/80",
                )}
              />
            ))}
            <span className="sr-only">
              {box.label}, confidence {box.confidence.toFixed(2)}
            </span>
          </button>
        );
      })}
    </>
  );
}

/** Labels sit above each overlay's top-left corner, outside any scaled space. */
export function OverlayLabels({
  overlays,
  selectedId,
}: {
  overlays: Overlay[];
  selectedId: string | null;
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      {overlays.map((overlay) => {
        const active = overlay.id === selectedId;
        const [left, top] =
          overlay.kind === "box"
            ? [overlay.x, overlay.y]
            : [
                Math.min(...overlay.polygon.map((p) => p[0])),
                Math.min(...overlay.polygon.map((p) => p[1])),
              ];

        return (
          <span
            key={overlay.id}
            style={{ left: `${left * 100}%`, top: `${top * 100}%` }}
            className={cn(
              "absolute -translate-y-full whitespace-nowrap px-1 py-0.5 font-mono text-[9px] leading-none",
              active ? "bg-signal text-void" : "bg-void/90 text-signal",
            )}
          >
            {overlay.label} {overlay.confidence.toFixed(2)}
          </span>
        );
      })}
    </div>
  );
}
