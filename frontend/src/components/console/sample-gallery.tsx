"use client";

import { ImagePlus } from "lucide-react";
import { MODE_LABELS } from "@/lib/tasks";
import type { Sample } from "@/lib/samples";
import { cn } from "@/lib/utils";
import type { UploadMode } from "@/lib/types";

/**
 * One-click sample imagery for the current layout, from
 * public/samples/manifest.json. A sample not yet added keeps a labelled empty
 * slot, so its absence is visible rather than silently skipped.
 */
export function SampleGallery({
  samples,
  mode,
  loadingId,
  onPick,
}: {
  samples: Sample[] | null;
  mode: UploadMode;
  loadingId: string | null;
  onPick: (sample: Sample) => void;
}) {
  if (!samples) return null;
  const forMode = samples.filter((sample) => sample.mode === mode);
  if (forMode.length === 0) return null;

  return (
    <div className="space-y-2">
      <p className="text-[13px] text-text-dim">Or try a sample</p>
      <ul className="grid gap-2 sm:grid-cols-2">
        {forMode.map((sample) =>
          sample.status === "ready" ? (
            <li key={sample.id}>
              <button
                type="button"
                disabled={loadingId !== null}
                onClick={() => onPick(sample)}
                className={cn(
                  "group flex w-full items-center gap-3 rounded-sm border border-rule bg-raised p-2 text-left transition-colors hover:border-signal disabled:cursor-wait disabled:opacity-60",
                  loadingId === sample.id && "border-signal",
                )}
              >
                <SampleThumb sample={sample} />
                <span className="min-w-0">
                  <span className="block truncate text-[13px] font-medium text-foreground group-hover:text-signal">
                    {loadingId === sample.id ? "Loading…" : sample.title}
                  </span>
                  <span className="block truncate text-[11.5px] text-muted-foreground">
                    {sample.files.map((f) => f.label).join(" and ")}
                  </span>
                </span>
              </button>
            </li>
          ) : (
            <li
              key={sample.id}
              className="flex items-center gap-3 rounded-sm border border-dashed border-rule-strong p-2"
            >
              <span className="grid size-11 shrink-0 place-items-center rounded-sm bg-raised">
                <ImagePlus className="size-4 text-muted-foreground" aria-hidden />
              </span>
              <span className="min-w-0 text-[11.5px] leading-snug text-muted-foreground">
                <span className="block text-[12.5px] text-text-dim">
                  {MODE_LABELS[sample.mode]} sample not added yet
                </span>
                {sample.needs}
              </span>
            </li>
          ),
        )}
      </ul>
      {forMode.some((s) => s.status === "ready") ? (
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          {Array.from(new Set(forMode.map((s) => s.attribution).filter(Boolean))).join(" ")}
        </p>
      ) : null}
    </div>
  );
}

function SampleThumb({ sample }: { sample: Sample }) {
  return (
    <span className="relative block size-11 shrink-0 overflow-hidden rounded-sm border border-rule bg-void">
      {sample.thumbnail ? (
        // eslint-disable-next-line @next/next/no-img-element -- 64 px static tile, pixelated on purpose
        <img
          src={sample.thumbnail}
          alt=""
          className="size-full object-cover [image-rendering:pixelated]"
        />
      ) : null}
    </span>
  );
}
