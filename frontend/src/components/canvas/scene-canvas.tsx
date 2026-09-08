"use client";

import { useState } from "react";
import { Layers, Maximize2, SplitSquareHorizontal } from "lucide-react";
import { Frame } from "@/components/frame";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { OverlayLabels, OverlayLayer } from "@/components/canvas/overlay-layer";
import { SyntheticRaster } from "@/components/canvas/synthetic-raster";
import { formatBytes } from "@/lib/validate-upload";
import { formatDms } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Overlay, SceneMeta, UploadMode } from "@/lib/types";

export interface Pane {
  meta: SceneMeta;
  label: string;
  /** Object URL when the browser can decode the file; null for GeoTIFF. */
  previewUrl: string | null;
  tint: "optical" | "sar";
  seed: number;
}

export function SceneCanvas({
  panes,
  mode,
  overlays,
  children,
}: {
  panes: Pane[];
  mode: UploadMode | null;
  overlays: Overlay[];
  /** Rendered in place of the raster when no scene is bound. */
  children?: React.ReactNode;
}) {
  const [showOverlays, setShowOverlays] = useState(true);
  const [split, setSplit] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const empty = panes.length === 0;
  const visiblePanes = split ? panes : panes.slice(0, 1);
  const primary = panes[0]?.meta ?? null;

  return (
    <Frame
      label="SCENE"
      aside={primary ? formatBytes(primary.sizeBytes) : undefined}
      className="min-w-0 flex-1"
      bodyClassName="p-3 pt-4 gap-3"
    >
      <div className="relative min-h-0 flex-1 border border-rule bg-void">
        {empty ? (
          <div className="texture-graticule absolute inset-0 grid place-items-center overflow-auto">
            {children}
          </div>
        ) : (
          <>
            <div className="absolute inset-0 flex">
              {visiblePanes.map((pane, i) => (
                <div
                  key={pane.label}
                  className={cn(
                    "relative min-w-0 flex-1 overflow-hidden",
                    i > 0 && "border-l border-rule-strong",
                  )}
                >
                  {pane.previewUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element -- a blob: URL from the user's own file
                    <img
                      src={pane.previewUrl}
                      alt={pane.meta.sceneId}
                      className="absolute inset-0 size-full object-cover"
                    />
                  ) : (
                    <SyntheticRaster
                      seed={pane.seed}
                      tint={pane.tint}
                      className="absolute inset-0 size-full"
                    />
                  )}
                  <div className="texture-graticule texture-scanline absolute inset-0" />

                  <span className="absolute left-2 top-2 max-w-[85%] truncate bg-void/85 px-1.5 py-0.5 font-mono text-[10px] text-text-dim">
                    {pane.label}
                  </span>
                  {!pane.previewUrl ? (
                    <span className="absolute bottom-2 left-2 bg-void/85 px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">
                      no browser preview for this format
                    </span>
                  ) : null}

                  {showOverlays && i === visiblePanes.length - 1 ? (
                    <>
                      <OverlayLayer
                        overlays={overlays}
                        selectedId={selectedId}
                        onSelect={setSelectedId}
                      />
                      <OverlayLabels overlays={overlays} selectedId={selectedId} />
                    </>
                  ) : null}
                </div>
              ))}
            </div>

            <CanvasControls
              showOverlays={showOverlays}
              onToggleOverlays={() => setShowOverlays((v) => !v)}
              canSplit={panes.length > 1}
              split={split}
              onToggleSplit={() => setSplit((v) => !v)}
              overlayCount={overlays.length}
            />
          </>
        )}
      </div>

      <ReadoutStrip meta={primary} mode={mode} overlayCount={overlays.length} />
    </Frame>
  );
}

function CanvasControls({
  showOverlays,
  onToggleOverlays,
  canSplit,
  split,
  onToggleSplit,
  overlayCount,
}: {
  showOverlays: boolean;
  onToggleOverlays: () => void;
  canSplit: boolean;
  split: boolean;
  onToggleSplit: () => void;
  overlayCount: number;
}) {
  return (
    <div className="absolute right-2 top-2 flex flex-col gap-px border border-rule bg-panel">
      <CanvasButton
        active={showOverlays}
        disabled={overlayCount === 0}
        onClick={onToggleOverlays}
        label={overlayCount === 0 ? "No overlays yet" : "Overlays"}
      >
        <Layers className="size-3.5" />
      </CanvasButton>
      {canSplit ? (
        <CanvasButton active={split} onClick={onToggleSplit} label="Side by side">
          <SplitSquareHorizontal className="size-3.5" />
        </CanvasButton>
      ) : null}
      <CanvasButton active={false} onClick={() => {}} label="Fit to view">
        <Maximize2 className="size-3.5" />
      </CanvasButton>
    </div>
  );
}

function CanvasButton({
  active,
  disabled,
  onClick,
  label,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          onClick={onClick}
          aria-pressed={active}
          className={cn(
            "grid size-7 place-items-center transition-colors",
            active ? "bg-signal-deep text-signal" : "text-muted-foreground hover:text-foreground",
            disabled && "cursor-not-allowed opacity-40 hover:text-muted-foreground",
          )}
        >
          {children}
          <span className="sr-only">{label}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="left" className="font-mono text-[10px]">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * Georeferencing shows a dash until /upload parses GeoTIFF headers. A plausible
 * coordinate here would be a fabricated one, which is worse than an empty field.
 */
function ReadoutStrip({
  meta,
  mode,
  overlayCount,
}: {
  meta: SceneMeta | null;
  mode: UploadMode | null;
  overlayCount: number;
}) {
  const dash = "—";
  const cells: [string, string][] = [
    ["lat", meta?.lat != null ? formatDms(meta.lat, "lat") : dash],
    ["lon", meta?.lon != null ? formatDms(meta.lon, "lon") : dash],
    ["gsd", meta?.gsd != null ? `${meta.gsd} m` : dash],
    ["crs", meta?.epsg ?? dash],
    ["bands", meta?.bands?.join(" ") ?? dash],
    ["mode", mode ?? dash],
    ["overlays", String(overlayCount)],
  ];

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-1 border-t border-rule pt-2 font-mono text-[11px]">
      {cells.map(([name, value]) => (
        <span key={name} className="flex items-baseline gap-1.5">
          <span className="text-muted-foreground">{name}</span>
          <span className={cn(value === dash ? "text-rule-strong" : "text-text-dim")}>
            {value}
          </span>
        </span>
      ))}
      {meta && meta.lat == null ? (
        <span className="text-muted-foreground">header not parsed yet</span>
      ) : null}
    </div>
  );
}
