"use client";

import { useRef, useState } from "react";
import { FileWarning, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MODE_HINTS, MODE_LABELS, MODE_ROLES } from "@/lib/tasks";
import { acceptAttribute, formatBytes, validateFile } from "@/lib/validate-upload";
import { cn } from "@/lib/utils";
import type { AssetRole, StagedFile, UploadMode } from "@/lib/types";

const MODES: UploadMode[] = ["single", "cross_modal", "bi_temporal"];

export function UploadPanel({
  mode,
  onModeChange,
  onBind,
  binding,
  bindError,
}: {
  mode: UploadMode;
  onModeChange: (mode: UploadMode) => void;
  onBind: (mode: UploadMode, files: StagedFile[], benchmarkMode: boolean) => void;
  binding: boolean;
  bindError: string | null;
}) {
  const [benchmarkMode, setBenchmarkMode] = useState(false);
  const [staged, setStaged] = useState<Partial<Record<AssetRole, StagedFile>>>({});
  const [errors, setErrors] = useState<Partial<Record<AssetRole, string>>>({});

  const slots = MODE_ROLES[mode];
  const ready = slots.every((slot) => staged[slot.role as AssetRole]);
  const remaining = slots.filter((slot) => !staged[slot.role as AssetRole]).length;

  function reset() {
    setStaged({});
    setErrors({});
  }

  function accept(role: AssetRole, file: File | undefined) {
    if (!file) return;
    const result = validateFile(file, benchmarkMode);
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
    setStaged((prev) => ({
      ...prev,
      [role]: { role, name: file.name, sizeBytes: file.size, file },
    }));
  }

  function clear(role: AssetRole) {
    setStaged((prev) => {
      const next = { ...prev };
      delete next[role];
      return next;
    });
    setErrors((prev) => ({ ...prev, [role]: undefined }));
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-5 p-6">
      <div className="space-y-1.5">
        <h2 className="text-base font-medium text-foreground">Bind a scene</h2>
        <p className="max-w-md text-[13px] leading-relaxed text-muted-foreground">
          Pick how many captures the question needs, then load them. GeoTIFF keeps
          the georeferencing; benchmark corpora ship as RGB and need the switch below.
        </p>
      </div>

      <Tabs
        value={mode}
        onValueChange={(value) => {
          onModeChange(value as UploadMode);
          reset();
        }}
      >
        <TabsList className="h-8 w-full rounded-none border border-rule bg-raised p-0.5">
          {MODES.map((m) => (
            <TabsTrigger
              key={m}
              value={m}
              className="h-7 flex-1 rounded-none font-mono text-[11px] data-[state=active]:bg-signal-deep data-[state=active]:text-signal data-[state=active]:shadow-none"
            >
              {MODE_LABELS[m]}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <p className="font-mono text-[11px] text-muted-foreground">{MODE_HINTS[mode]}</p>

      <div className={cn("grid gap-3", slots.length > 1 && "sm:grid-cols-2")}>
        {slots.map((slot) => (
          <Dropzone
            key={slot.role}
            label={slot.label}
            staged={staged[slot.role as AssetRole]}
            error={errors[slot.role as AssetRole]}
            benchmarkMode={benchmarkMode}
            onFile={(file) => accept(slot.role as AssetRole, file)}
            onClear={() => clear(slot.role as AssetRole)}
          />
        ))}
      </div>

      <label className="flex cursor-pointer items-start gap-3 border border-rule bg-raised p-3">
        <input
          type="checkbox"
          checked={benchmarkMode}
          onChange={(e) => {
            setBenchmarkMode(e.target.checked);
            reset();
          }}
          className="mt-0.5 size-3.5 shrink-0 accent-[var(--signal)]"
        />
        <span className="space-y-0.5">
          <span className="block text-[13px] text-foreground">Benchmark dataset mode</span>
          <span className="block font-mono text-[11px] leading-relaxed text-muted-foreground">
            Accepts PNG and JPEG alongside GeoTIFF. Answers stay in pixel space —
            nothing is projected to a coordinate system.
          </span>
        </span>
      </label>

      <div className="flex items-center gap-3">
        <Button
          disabled={!ready || binding}
          onClick={() =>
            onBind(
              mode,
              slots.map((slot) => staged[slot.role as AssetRole]!),
              benchmarkMode,
            )
          }
          className="h-8 rounded-sm bg-signal font-mono text-[11px] text-void hover:bg-signal/85 disabled:bg-raised disabled:text-muted-foreground disabled:opacity-100"
        >
          {binding ? "Binding" : "Bind scene"}
        </Button>
        <span className="font-mono text-[11px] text-muted-foreground">
          {binding
            ? "uploading…"
            : ready
              ? `${slots.length} file${slots.length > 1 ? "s" : ""} staged`
              : `waiting on ${remaining} more ${remaining === 1 ? "file" : "files"}`}
        </span>
      </div>

      {bindError ? (
        <p className="flex items-start gap-2 border border-alarm bg-raised p-2.5 font-mono text-[11px] leading-relaxed text-alarm">
          <FileWarning className="mt-px size-3.5 shrink-0" />
          {bindError}
        </p>
      ) : null}
    </div>
  );
}

function Dropzone({
  label,
  staged,
  error,
  benchmarkMode,
  onFile,
  onClear,
}: {
  label: string;
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
          "flex h-28 flex-col items-center justify-center gap-2 border border-dashed bg-raised px-3 text-center transition-colors",
          dragging ? "border-signal bg-signal-deep/30" : "border-rule-strong",
          error && "border-alarm",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          className="sr-only"
          accept={acceptAttribute(benchmarkMode)}
          onChange={(e) => onFile(e.target.files?.[0])}
        />

        {staged ? (
          <>
            <p className="max-w-full truncate font-mono text-[11px] text-foreground">
              {staged.name}
            </p>
            <p className="font-mono text-[10px] text-muted-foreground">
              {formatBytes(staged.sizeBytes)}
            </p>
            <button
              type="button"
              onClick={() => {
                onClear();
                if (inputRef.current) inputRef.current.value = "";
              }}
              className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground hover:text-foreground"
            >
              <X className="size-3" />
              Remove
            </button>
          </>
        ) : (
          <>
            <Upload className="size-4 text-muted-foreground" />
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="font-mono text-[11px] text-text-dim underline-offset-4 hover:text-signal hover:underline"
            >
              Choose {label.toLowerCase()}
            </button>
            <p className="font-mono text-[10px] text-muted-foreground">
              or drop {benchmarkMode ? ".tif .png .jpg" : ".tif .tiff"}
            </p>
          </>
        )}
      </div>

      <p className="flex items-baseline gap-1.5 font-mono text-[10px]">
        <span className="text-muted-foreground">{label}</span>
        {error ? (
          <span className="flex items-start gap-1 text-alarm">
            <FileWarning className="mt-px size-3 shrink-0" />
            {error}
          </span>
        ) : null}
      </p>
    </div>
  );
}
