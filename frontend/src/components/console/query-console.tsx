"use client";

import { useRef, useState } from "react";
import { CornerDownLeft } from "lucide-react";
import { Frame } from "@/components/frame";
import { INTENT_CODES, SUGGESTED_QUERIES } from "@/lib/suggestions";
import { cn } from "@/lib/utils";

/**
 * Command-console query entry. Styled after shadcn's Command palette — a
 * prompt marker, an inline list, keyboard-first selection — but docked rather
 * than opened in a dialog, since querying is the panel's whole job.
 */
export function QueryConsole({
  disabled,
  busy,
  onSubmit,
}: {
  disabled: boolean;
  busy: boolean;
  onSubmit: (query: string) => void;
}) {
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const matches = value.trim()
    ? SUGGESTED_QUERIES.filter((s) => s.text.toLowerCase().includes(value.toLowerCase()))
    : SUGGESTED_QUERIES;
  const showList = focused && matches.length > 0;

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || disabled || busy) return;
    onSubmit(trimmed);
    setValue("");
    setFocused(false);
    inputRef.current?.blur();
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!showList) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlight((h) => (h + 1) % matches.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight((h) => (h - 1 + matches.length) % matches.length);
    } else if (e.key === "Enter" && value.trim() === "") {
      e.preventDefault();
      submit(matches[highlight].text);
    } else if (e.key === "Escape") {
      setFocused(false);
      inputRef.current?.blur();
    }
  }

  return (
    <Frame
      label="QUERY"
      aside={disabled ? "bind a scene first" : "↩ to run"}
      bodyClassName="p-2 pt-3"
    >
      <div
        className={cn(
          "flex items-center gap-2 border bg-raised px-2.5",
          focused ? "border-signal" : "border-rule",
          disabled && "opacity-50",
        )}
      >
        <span
          className={cn(
            "font-mono text-[13px] leading-none",
            busy ? "text-caution" : "text-signal",
          )}
          aria-hidden
        >
          {busy ? "··" : "▸"}
        </span>
        <input
          ref={inputRef}
          value={value}
          disabled={disabled || busy}
          placeholder={busy ? "running…" : "Ask about the scene"}
          onChange={(e) => {
            setValue(e.target.value);
            setHighlight(0);
          }}
          onFocus={() => setFocused(true)}
          onBlur={() => window.setTimeout(() => setFocused(false), 120)}
          onKeyDown={(e) => {
            onKeyDown(e);
            if (e.key === "Enter" && value.trim()) submit(value);
          }}
          className="h-9 flex-1 bg-transparent font-mono text-[12px] text-foreground placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed"
        />
        <button
          type="button"
          disabled={disabled || busy || !value.trim()}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => submit(value)}
          className="text-muted-foreground transition-colors hover:text-signal disabled:opacity-30 disabled:hover:text-muted-foreground"
        >
          <CornerDownLeft className="size-3.5" />
          <span className="sr-only">Run query</span>
        </button>
      </div>

      {showList ? (
        <ul className="mt-1 border border-rule bg-panel">
          {matches.map((s, i) => (
            <li key={s.text}>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onMouseEnter={() => setHighlight(i)}
                onClick={() => submit(s.text)}
                className={cn(
                  "flex w-full items-center gap-2 px-2.5 py-1.5 text-left font-mono text-[11px] transition-colors",
                  i === highlight ? "bg-raised text-foreground" : "text-text-dim",
                )}
              >
                <span
                  className={cn(
                    "shrink-0 text-[9px]",
                    i === highlight ? "text-signal" : "text-muted-foreground",
                  )}
                >
                  {INTENT_CODES[s.intent]}
                </span>
                <span className="truncate">{s.text}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </Frame>
  );
}
