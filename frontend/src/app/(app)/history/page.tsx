"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AlertTriangle, ImageOff } from "lucide-react";
import { RequireAuth } from "@/components/auth/require-auth";
import { TopBar } from "@/components/top-bar";
import { fetchThumbnailUrl, getHistory } from "@/lib/api";
import { explainError, type Explained } from "@/lib/errors";
import { TASK_LABELS } from "@/lib/tasks";
import type { HistoryItem } from "@/lib/types";

export default function HistoryPage() {
  return (
    <RequireAuth>
      <History />
    </RequireAuth>
  );
}

const STATUS_NOTE: Record<string, string> = {
  rejected: "Imagery didn’t fit the question",
  unavailable: "Model unavailable",
  failed: "Didn’t finish",
};

function History() {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<Explained | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getHistory(50, controller.signal)
      .then(setItems)
      .catch((err) => {
        if (!controller.signal.aborted) setError(explainError(err));
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="flex min-h-dvh flex-col bg-void">
      <TopBar />
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 sm:px-6">
        <h1 className="text-xl font-semibold text-foreground">Your history</h1>
        <p className="mt-1 text-[13.5px] text-text-dim">
          Every question you&rsquo;ve asked, newest first. Open one to see its answer and trace
          again.
        </p>

        <div className="mt-6">
          {error ? (
            <div role="alert" className="rounded-sm border border-alarm/70 bg-panel p-4">
              <p className="flex items-center gap-2 text-[14px] font-medium text-alarm">
                <AlertTriangle className="size-4" aria-hidden />
                {error.title}
              </p>
              <p className="mt-1 text-[13px] leading-relaxed text-text-dim">{error.detail}</p>
            </div>
          ) : items === null ? (
            <p className="text-[13px] text-muted-foreground" role="status">
              Loading your history…
            </p>
          ) : items.length === 0 ? (
            <div className="rounded-sm border border-dashed border-rule-strong p-6 text-center">
              <p className="text-[14px] text-foreground">No questions yet</p>
              <p className="mt-1 text-[13px] text-muted-foreground">
                Runs you make in the console are saved here, visible only to you.
              </p>
              <Link
                href="/console"
                className="mt-4 inline-block rounded-sm bg-signal px-4 py-2 text-[13px] font-medium text-void hover:bg-signal/85"
              >
                Open the console
              </Link>
            </div>
          ) : (
            <ul className="divide-y divide-rule rounded-sm border border-rule bg-panel">
              {items.map((item) => (
                <li key={item._id}>
                  <Link
                    href={`/console?query=${encodeURIComponent(item._id)}`}
                    className="flex items-center gap-4 p-3 transition-colors hover:bg-raised"
                  >
                    <Thumbnail uploadId={item.upload_id} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[14px] text-foreground">
                        {item.query}
                      </span>
                      <span className="mt-0.5 block text-[12px] text-muted-foreground">
                        {item.task_type ? TASK_LABELS[item.task_type] : "Task not recorded"}
                        {item.status && item.status !== "ok" ? (
                          <span className="text-caution">, {STATUS_NOTE[item.status]}</span>
                        ) : null}
                      </span>
                    </span>
                    <time
                      dateTime={item.timestamp}
                      className="shrink-0 text-right text-[12px] text-muted-foreground"
                    >
                      {formatWhen(item.timestamp)}
                    </time>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </main>
    </div>
  );
}

function formatWhen(iso: string): string {
  const date = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return date.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Fetched only once the row scrolls into view, then kept until unmount. */
function Thumbnail({ uploadId }: { uploadId: string | null }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!uploadId || !node) return;
    const controller = new AbortController();
    let made: string | null = null;
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      observer.disconnect();
      fetchThumbnailUrl(uploadId, controller.signal)
        .then((u) => {
          made = u;
          setUrl(u);
        })
        .catch(() => {
          if (!controller.signal.aborted) setFailed(true);
        });
    });
    observer.observe(node);
    return () => {
      observer.disconnect();
      controller.abort();
      if (made) URL.revokeObjectURL(made);
    };
  }, [uploadId]);

  return (
    <span
      ref={ref}
      className="texture-graticule relative grid size-12 shrink-0 place-items-center overflow-hidden rounded-sm border border-rule bg-void"
    >
      {url ? (
        // eslint-disable-next-line @next/next/no-img-element -- blob: URL of the user's own upload
        <img src={url} alt="" className="size-full object-cover [image-rendering:pixelated]" />
      ) : !uploadId || failed ? (
        <ImageOff className="size-4 text-muted-foreground" aria-label="Image not available" />
      ) : null}
    </span>
  );
}
