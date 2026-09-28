"use client";

import { useRef, useState } from "react";
import { FileWarning, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MODE_HINTS, MODE_ROLES } from "@/lib/tasks";
import {
  BENCHMARK_EXTENSIONS,
  acceptAttribute,
  formatBytes,
  validateFile,
} from "@/lib/validate-upload";
import { cn } from "@/lib/utils";
import type { AssetRole, StagedFile, UploadMode } from "@/lib/types";

const isPlainImage = (name: string) =>
  (BENCHMARK_EXTENSIONS as readonly string[]).some((ext) => name.toLowerCase().endsWith(ext));

/**
 * The upload slots for one imagery layout. The layout itself is chosen in the
 * step above; this only knows how many slots to show and what goes in each.
 */
export function UploadPanel({
  mode,
  onBind,
  binding,
}: {
  mode: UploadMode;
  onBind: (mode: UploadMode, files: StagedFile[], benchmarkMode: boolean) => void;
  binding: boolean;
}) {
  const [benchmarkMode, setBenchmarkMode] = useState(false);
  const [staged, setStaged] = useState<Partial<Record<AssetRole, StagedFile>>>({});
  const [errors, setErrors] = useState<Partial<Record<AssetRole, string>>>({});
  /** A PNG/JPEG dropped while benchmark mode was off, waiting on one click. */
  const [offer, setOffer] = useState<{ role: AssetRole; file: File } | null>(null);

  const slots = MODE_ROLES[mode];
  const ready = slots.every((slot) => staged[slot.role as AssetRole]);
  const remaining = slots.filter((slot) => !staged[slot.role as AssetRole]).length;

  function stage(role: AssetRole, file: File, benchmark: boolean) {
    const result = validateFile(file, benchmark);
    if (!result.ok) {
      setErrors((prev) => ({ ...prev, [role]: result.reason }));
      setStaged((prev) => {
        const next = { ...prev };
        delete next[role];
        return next;
      });
      return;
    }
    setErrors((prev) => ({ ...prev, [role]: undefined }));
    setStaged((prev) => ({ ...prev, [role]: { role, name: file.name, sizeBytes: file.size, file } }));
  }

  function accept(role: AssetRole, file: File | undefined) {
    if (!file) return;
    setOffer(null);
    if (!benchmarkMode && isPlainImage(file.name)) {
      setErrors((prev) => ({ ...prev, [role]: undefined }));
      setOffer({ role, file });
      return;
    }
    stage(role, file, benchmarkMode);
  }

  function clear(role: AssetRole) {
    setStaged((prev) => {
      const next = { ...prev };
      delete next[role];
      return next;
    });
    setErrors((prev) => ({ ...prev, [role]: undefined }));
  }

  function setBenchmark(on: boolean) {
    setBenchmarkMode(on);
    setOffer(null);
    // Turning it off un-stages only what it had let in.
    if (!on) {
      setStaged((prev) =>
        Object.fromEntries(Object.entries(prev).filter(([, f]) => f && !isPlainImage(f.name))),
      );
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-[13px] leading-relaxed text-text-dim">{MODE_HINTS[mode]}</p>

      <div className={cn("grid gap-3", slots.length > 1 && "sm:grid-cols-2")}>
        {slots.map((slot) => (
          <Dropzone
            key={slot.role}
            label={slot.label}
            noun={slot.noun}
            hint={slot.hint}
            staged={staged[slot.role as AssetRole]}
            error={errors[slot.role as AssetRole]}
            benchmarkMode={benchmarkMode}
            onFile={(file) => accept(slot.role as AssetRole, file)}
            onClear={() => clear(slot.role as AssetRole)}
          />
        ))}
      </div>

      {offer ? (
        <div role="status" className="space-y-2 border border-rule-strong bg-raised p-3">
          <p className="text-[13px] leading-relaxed text-text">
            <span className="font-medium text-foreground">{offer.file.name}</span> is a{" "}
            {offer.file.name.split(".").pop()?.toUpperCase()}. GeoTIFF is the default because it
            keeps every spectral band and the map coordinates; PNG and JPEG hold colour only.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              onClick={() => {
                setBenchmarkMode(true);
                stage(offer.role, offer.file, true);
                setOffer(null);
              }}
              className="h-8 rounded-sm bg-signal text-[12.5px] text-void hover:bg-signal/85"
            >
              Switch to benchmark mode and use this file
            </Button>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() => setOffer(null)}
              className="h-8 rounded-sm text-[12.5px] text-text-dim hover:bg-transparent hover:text-foreground"
            >
              Choose a different file
            </Button>
          </div>
        </div>
      ) : null}

      <label className="flex cursor-pointer items-start gap-3 text-[13px]">
        <input
          type="checkbox"
          checked={benchmarkMode}
          onChange={(e) => setBenchmark(e.target.checked)}
          className="mt-1 size-3.5 shrink-0 accent-[var(--signal)]"
        />
        <span>
          <span className="text-foreground">Benchmark mode</span>
          <span className="block text-[12px] leading-relaxed text-muted-foreground">
            Also accept PNG and JPEG, for datasets that ship as plain images. Answers stay in
            pixel space: there are no map coordinates.
          </span>
        </span>
      </label>

      <div className="flex flex-wrap items-center gap-3">
        <Button
          disabled={!ready || binding}
          onClick={() =>
            onBind(
              mode,
              slots.map((slot) => staged[slot.role as AssetRole]!),
              benchmarkMode,
            )
          }
          className="h-9 rounded-sm bg-signal px-4 text-[13px] font-medium text-void hover:bg-signal/85 disabled:bg-raised disabled:text-muted-foreground disabled:opacity-100"
        >
          {binding ? "Uploading…" : "Upload and continue"}
        </Button>
        <span className="text-[12px] text-muted-foreground" aria-live="polite">
          {binding
            ? "Sending your files to the server."
            : ready
              ? `${slots.length === 1 ? "Image" : "Both images"} ready.`
              : `Add ${remaining} more ${remaining === 1 ? "image" : "images"}.`}
        </span>
      </div>
    </div>
  );
}

function Dropzone({
  label,
  noun,
  hint,
  staged,
  error,
  benchmarkMode,
  onFile,
  onClear,
}: {
  label: string;
  noun: string;
  hint: string;
  staged?: StagedFile;
  error?: string;
  benchmarkMode: boolean;
  onFile: (file: File | undefined) => void;
  onClear: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div className="space-y-1.5">
      <p className="flex items-baseline justify-between text-[12.5px]">
        <span className="font-medium text-foreground">{label}</span>
        <span className="text-muted-foreground">{hint}</span>
      </p>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          onFile(e.dataTransfer.files[0]);
        }}
        className={cn(
          "flex h-24 flex-col items-center justify-center gap-1.5 rounded-sm border border-dashed bg-raised px-3 text-center transition-colors",
          dragging ? "border-signal bg-signal-deep/30" : "border-rule-strong",
          staged && "border-solid border-signal/60",
          error && "border-alarm",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          className="sr-only"
          tabIndex={-1}
          aria-hidden
          // Accept both, so a PNG can be picked and offered benchmark mode.
          accept={acceptAttribute(true)}
          onChange={(e) => {
            onFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />

        {staged ? (
          <>
            <p className="max-w-full truncate text-[12.5px] text-foreground">{staged.name}</p>
            <p className="font-mono text-[11px] text-muted-foreground">
              {formatBytes(staged.sizeBytes)}
            </p>
            <button
              type="button"
              onClick={onClear}
              className="flex items-center gap-1 rounded-sm text-[12px] text-text-dim hover:text-foreground"
            >
              <X className="size-3" aria-hidden />
              Remove {noun.replace(/^an? /, "")}
            </button>
          </>
        ) : (
          <>
            <Upload className="size-4 text-muted-foreground" aria-hidden />
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="rounded-sm text-[12.5px] text-text-dim underline-offset-4 hover:text-signal hover:underline"
            >
              Choose {noun}
            </button>
            <p className="text-[11.5px] text-muted-foreground">
              or drop a {benchmarkMode ? ".tif, .png or .jpg" : ".tif"} file here
            </p>
          </>
        )}
      </div>

      {error ? (
        <p className="flex items-start gap-1.5 text-[12px] text-alarm" role="alert">
          <FileWarning className="mt-px size-3.5 shrink-0" aria-hidden />
          {error}
        </p>
      ) : null}
    </div>
  );
}
