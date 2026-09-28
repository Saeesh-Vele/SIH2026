"use client";

import { useState } from "react";
import { ImageOff, Layers, Maximize2, SplitSquareHorizontal, ZoomIn, ZoomOut } from "lucide-react";
import { Frame } from "@/components/frame";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { OverlayLabels, OverlayLayer } from "@/components/canvas/overlay-layer";
import { formatBytes } from "@/lib/validate-upload";
import { formatDms } from "@/lib/format";
import { MODE_LABELS } from "@/lib/tasks";
import { cn } from "@/lib/utils";
import type { Overlay, SceneMeta, UploadMode } from "@/lib/types";

export interface PanePreview {
  /** blob: URL — the file itself for PNG/JPEG, the server's render for GeoTIFF. */
  url: string | null;
  state: "loading" | "ready" | "failed";
  /** Why there is no picture, in plain words. */
  message?: string;
}

export interface Pane {
  meta: SceneMeta;
  label: string;
  preview: PanePreview;
}

const ZOOMS = [1, 2, 4, 8] as const;

export function SceneCanvas({
  panes,
  mode,
  overlays,
  className,
}: {
  panes: Pane[];
  mode: UploadMode | null;
  overlays: Overlay[];
  className?: string;
}) {
  const [showOverlays, setShowOverlays] = useState(true);
  const [split, setSplit] = useState(true);
  const [zoom, setZoom] = useState<(typeof ZOOMS)[number]>(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const empty = panes.length === 0;
  const visiblePanes = split ? panes : panes.slice(-1);
  const primary = panes[0]?.meta ?? null;
  const zoomIndex = ZOOMS.indexOf(zoom);

  return (
    <Frame
      label="IMAGERY"
      aside={primary ? formatBytes(primary.sizeBytes) : undefined}
      className={cn("min-w-0", className)}
      bodyClassName="p-3 pt-4 gap-3"
    >
      <div
        data-tour="canvas"
        className="relative h-[min(85vw,26rem)] overflow-hidden rounded-sm border border-rule bg-void lg:h-auto lg:min-h-[18rem] lg:flex-1"
      >
        {empty ? (
          <div className="texture-graticule absolute inset-0 grid place-items-center p-6 text-center">
            <p className="max-w-xs text-[13px] leading-relaxed text-muted-foreground">
              Your imagery appears here once it&rsquo;s uploaded, with anything the answer marks
              drawn on top.
            </p>
          </div>
        ) : (
          <>
            <div className="absolute inset-0 flex">
              {visiblePanes.map((pane, i) => (
                <PaneView
                  key={pane.label}
                  pane={pane}
                  zoom={zoom}
                  divider={i > 0}
                  overlays={
                    showOverlays && i === visiblePanes.length - 1 ? (
                      <>
                        <OverlayLayer
                          overlays={overlays}
                          selectedId={selectedId}
                          onSelect={setSelectedId}
                        />
                        <OverlayLabels overlays={overlays} selectedId={selectedId} />
                      </>
                    ) : null
                  }
                />
              ))}
            </div>

            <div className="absolute right-2 top-2 flex flex-col gap-px rounded-sm border border-rule bg-panel">
              <CanvasButton
                active={showOverlays && overlays.length > 0}
                disabled={overlays.length === 0}
                onClick={() => setShowOverlays((v) => !v)}
                label={
                  overlays.length === 0
                    ? "Nothing marked yet"
                    : showOverlays
                      ? "Hide marked areas"
                      : "Show marked areas"
                }
              >
                <Layers className="size-3.5" />
              </CanvasButton>
              {panes.length > 1 ? (
                <CanvasButton
                  active={split}
                  onClick={() => setSplit((v) => !v)}
                  label={split ? "Show one image" : "Show side by side"}
                >
                  <SplitSquareHorizontal className="size-3.5" />
                </CanvasButton>
              ) : null}
              <CanvasButton
                active={false}
                disabled={zoomIndex === ZOOMS.length - 1}
                onClick={() => setZoom(ZOOMS[Math.min(zoomIndex + 1, ZOOMS.length - 1)])}
                label="Zoom in"
              >
                <ZoomIn className="size-3.5" />
              </CanvasButton>
              <CanvasButton
                active={false}
                disabled={zoomIndex === 0}
                onClick={() => setZoom(ZOOMS[Math.max(zoomIndex - 1, 0)])}
                label="Zoom out"
              >
                <ZoomOut className="size-3.5" />
              </CanvasButton>
              <CanvasButton
                active={zoom === 1}
                disabled={zoom === 1}
                onClick={() => setZoom(1)}
                label="Fit to view"
              >
                <Maximize2 className="size-3.5" />
              </CanvasButton>
            </div>
            {zoom > 1 ? (
              <span className="absolute bottom-2 right-2 rounded-sm bg-void/85 px-1.5 py-0.5 font-mono text-[11px] text-text-dim">
                {zoom}×, scroll to pan
              </span>
            ) : null}
          </>
        )}
      </div>

      <ReadoutStrip meta={primary} mode={mode} overlayCount={overlays.length} />
    </Frame>
  );
}

/**
 * One image, fitted inside its pane at its own aspect ratio. Overlays sit in
 * the same box as the picture, so their 0–1 coordinates land on the pixels
 * they describe at any pane shape and any zoom.
 */
function PaneView({
  pane,
  zoom,
  divider,
  overlays,
}: {
  pane: Pane;
  zoom: number;
  divider: boolean;
  overlays: React.ReactNode;
}) {
  const geo = pane.meta.geo;
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const w = natural?.w ?? geo?.width ?? 1;
  const h = natural?.h ?? geo?.height ?? 1;
  const ratio = w / h;
  // Small tiles (EuroSAT is 64 px) are shown as their pixels, not a blur.
  const pixelated = w <= 512;

  return (
    <div className={cn("relative min-w-0 flex-1", divider && "border-l border-rule-strong")}>
      <div className="absolute inset-0 flex overflow-auto [container-type:size]">
        <div
          className="relative m-auto shrink-0"
          style={{
            width: `calc(min(100cqw, 100cqh * ${ratio}) * ${zoom})`,
            aspectRatio: `${w} / ${h}`,
          }}
        >
          {pane.preview.url ? (
            // eslint-disable-next-line @next/next/no-img-element -- a blob: URL of the user's own file
            <img
              src={pane.preview.url}
              alt={`${pane.label}, as uploaded`}
              onLoad={(e) =>
                setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
              }
              className={cn(
                "absolute inset-0 size-full",
                pixelated && "[image-rendering:pixelated]",
              )}
            />
          ) : (
            <div className="texture-graticule absolute inset-0 grid place-items-center p-3 text-center">
              {pane.preview.state === "loading" ? (
                <p className="text-[12.5px] text-muted-foreground" role="status">
                  Rendering a preview…
                </p>
              ) : (
                <p className="flex max-w-[24ch] flex-col items-center gap-1.5 text-[12.5px] leading-snug text-muted-foreground">
                  <ImageOff className="size-4" aria-hidden />
                  {pane.preview.message ?? "No preview for this file. It can still be queried."}
                </p>
              )}
            </div>
          )}
          {overlays}
        </div>
      </div>

      <span className="pointer-events-none absolute left-2 top-2 z-10 max-w-[70%] truncate rounded-sm bg-void/85 px-1.5 py-0.5 text-[11.5px] text-text">
        {pane.label}
      </span>
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
          aria-label={label}
          className={cn(
            "grid size-8 place-items-center transition-colors",
            active ? "bg-signal-deep text-signal" : "text-text-dim hover:text-foreground",
            disabled && "cursor-not-allowed opacity-40 hover:text-text-dim",
          )}
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="left" className="text-[11.5px]">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

/**
 * Coordinates come from the file's own header, read at upload. A PNG or JPEG
 * has none to give; a file we could not read says so. Nothing is inferred.
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
  if (!meta) {
    return (
      <p className="shrink-0 border-t border-rule pt-2 text-[12px] text-muted-foreground">
        Location and resolution appear here when the file carries them.
      </p>
    );
  }

  const geo = meta.geo;
  const cells: [string, string][] = [];
  if (geo?.width && geo.height) cells.push(["size", `${geo.width} × ${geo.height} px`]);
  if (geo?.bands) cells.push(["bands", String(geo.bands)]);
  if (geo?.status === "georeferenced") {
    if (geo.lat != null) cells.push(["lat", formatDms(geo.lat, "lat")]);
    if (geo.lon != null) cells.push(["lon", formatDms(geo.lon, "lon")]);
    if (geo.gsd_m != null) cells.push(["pixel", `${geo.gsd_m} m`]);
    if (geo.crs) cells.push(["crs", geo.crs]);
  }
  if (mode) cells.push(["layout", MODE_LABELS[mode]]);
  cells.push(["marked", String(overlayCount)]);

  const note =
    geo?.status === "georeferenced"
      ? null
      : geo?.status === "none"
        ? "No georeferencing in this file"
        : "Georeferencing: not read";

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-1 border-t border-rule pt-2 font-mono text-[11.5px]">
      {cells.map(([name, value]) => (
        <span key={name} className="flex items-baseline gap-1.5">
          <span className="text-muted-foreground">{name}</span>
          <span className="text-text-dim">{value}</span>
        </span>
      ))}
      {note ? <span className="font-sans text-[12px] text-text-dim">{note}</span> : null}
    </div>
  );
}
