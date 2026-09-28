"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CircleHelp } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import type { Driver, DriveStep, PopoverDOM } from "./load-driver";

/**
 * The first-visit tour of the console.
 *
 * Eight steps, anchored to `data-tour` attributes rather than class names so
 * restyling never breaks it. It does the work alongside the person: step 2
 * loads a sample if nothing is loaded, step 4 presses Run, so the tour ends
 * with a real result on screen. Progress is kept per user in localStorage;
 * a refresh resumes where it left off, and once finished or skipped it never
 * appears on its own again. The "?" button starts it over.
 */

/** What the tour needs from the console, read fresh at each step. */
export interface TourActions {
  isBound: () => boolean;
  isBusy: () => boolean;
  hasResult: () => boolean;
  /** Load the default sample and wait until it is bound. */
  loadSample: () => Promise<void>;
  /** Run whatever is in the question box. Resolves when the run starts. */
  run: () => void;
}

type Stored = { state: "in_progress" | "done" | "skipped"; step: number };

const STORAGE_PREFIX = "satquery.tour.v1:";

function read(uid: string): Stored | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_PREFIX + uid);
    return raw ? (JSON.parse(raw) as Stored) : null;
  } catch {
    return null;
  }
}

function write(uid: string, value: Stored) {
  try {
    window.localStorage.setItem(STORAGE_PREFIX + uid, JSON.stringify(value));
  } catch {
    // Private mode or storage off: the tour still works, it just won't resume.
  }
}

const sel = (key: string) => `[data-tour="${key}"]`;
const find = (key: string) => document.querySelector(sel(key));

/** Resolves when `ready()` is true, or false after `ms`. */
function until(ready: () => boolean, ms: number): Promise<boolean> {
  return new Promise((resolve) => {
    const started = Date.now();
    const tick = () => {
      if (ready()) return resolve(true);
      if (Date.now() - started > ms) return resolve(false);
      window.setTimeout(tick, 120);
    };
    tick();
  });
}

/** Indexes that need the console to be in a given state before they make sense. */
const NEEDS_IMAGERY_FROM = 2;
const NEEDS_RESULT_FROM = 5;
const RUN_STEP = 3;

export function useConsoleTour(uid: string | null, actions: React.RefObject<TourActions>) {
  const [welcomeOpen, setWelcomeOpen] = useState(false);
  const driverRef = useRef<Driver | null>(null);

  const start = useCallback(
    async (from = 0) => {
      if (!uid) return;
      setWelcomeOpen(false);
      driverRef.current?.destroy();

      const { driver } = await import("./load-driver");
      const act = () => actions.current;
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      let destroyed = false;
      // The outcome is written here rather than in onDestroyed, which
      // driver.js skips when no element is active at the moment it closes.
      const end = (state: "done" | "skipped") => {
        if (destroyed) return;
        destroyed = true;
        write(uid, { state, step: 0 });
        driverRef.current = null;
        d.destroy();
      };

      const busyButton = (button: HTMLButtonElement, text: string) => {
        button.disabled = true;
        button.textContent = text;
      };

      const steps: DriveStep[] = [
        {
          element: sel("task"),
          popover: {
            title: "Say what you want to do",
            description:
              "Pick a kind of question, or leave it on <b>Let SatQuery decide</b> and just ask. Each choice shows the imagery it needs.",
          },
        },
        {
          element: sel("upload"),
          popover: {
            title: "Add imagery",
            description:
              "Upload your own satellite image, or pick a sample. If nothing is loaded yet, the tour will load a sample for you.",
            onPopoverRender: (popover) => {
              popover.nextButton.textContent = act().isBound() ? "Next" : "Load a sample";
            },
            onNextClick: async (_el, _step, { driver: d }) => {
              if (!act().isBound()) {
                const button = d.getState("popover")?.nextButton as HTMLButtonElement | undefined;
                if (button) busyButton(button, "Loading…");
                await act().loadSample();
                await until(() => act().isBound(), 20000);
              }
              d.moveNext();
            },
          },
        },
        {
          element: sel("question"),
          popover: {
            title: "Ask in plain English",
            description:
              "Type a question, or tap a suggestion to fill the box. A sample comes with a question already in it.",
          },
        },
        {
          element: sel("run"),
          popover: {
            title: "Run it",
            description:
              "Run sends your question with the imagery. The tour will press it for you.",
            nextBtnText: "Run it",
            onPopoverRender: (popover) => {
              if (act().hasResult()) popover.nextButton.textContent = "Next";
            },
            onNextClick: async (_el, _step, { driver: d }) => {
              if (!act().hasResult() && !act().isBusy()) {
                const button = d.getState("popover")?.nextButton as HTMLButtonElement | undefined;
                if (button) busyButton(button, "Starting…");
                act().run();
                await until(() => Boolean(find("progress")) || act().hasResult(), 8000);
              }
              d.moveNext();
            },
          },
        },
        {
          element: () => find("progress") ?? (find("answer") as Element),
          popover: {
            title: "Watch it work",
            description:
              "This bar shows which stage is running: reading the files, understanding the question, checking the imagery, running the model. The model step takes longest.",
            onPopoverRender: (popover) => {
              if (!act().hasResult()) {
                popover.nextButton.textContent = "Wait for the answer";
              } else if (!find("progress")) {
                // Fast runs (or a model that couldn't start) finish before this step shows.
                popover.description.textContent =
                  "This run finished before the tour got here. On a longer run, a bar here shows which stage is working: reading the files, understanding the question, checking the imagery, running the model.";
              }
            },
            onNextClick: async (_el, _step, { driver: d }) => {
              if (!act().hasResult()) {
                const button = d.getState("popover")?.nextButton as HTMLButtonElement | undefined;
                if (button) busyButton(button, "Waiting…");
                await until(() => act().hasResult() || !act().isBusy(), 180000);
              }
              d.moveNext();
            },
          },
        },
        {
          element: sel("result"),
          popover: {
            title: "The answer, then how sure it was",
            description:
              "The answer comes first. <b>Confidence</b> is how sure the model was of its own words, not a guarantee it’s right. If the imagery couldn’t answer, this says what happened and what to do.",
          },
        },
        {
          element: sel("trace"),
          popover: {
            title: "The trace",
            description:
              "Every run keeps a record of each step: what SatQuery decided you were asking, which model ran, and how long it took. Open it to check how the answer was reached.",
          },
        },
        {
          element: () => find("degraded") ?? (find("result") as Element),
          popover: {
            title: "If you see “Degraded”",
            description:
              "When the main model can’t run on the server, a smaller CPU model answers instead. That answer is labelled <b>Degraded</b> and its confidence says “not measured”. It’s never hidden.",
            onNextClick: () => end("done"),
          },
        },
      ];

      // Every popover gets the same frame: dialog semantics, a Skip button and
      // focus on the main action. driver.js lets a step's own render hook
      // replace the global one, so the two are composed here instead.
      const frame = (popover: PopoverDOM) => {
        popover.wrapper.setAttribute("role", "dialog");
        popover.wrapper.setAttribute("aria-label", popover.title.textContent ?? "Tour");
        const skip = document.createElement("button");
        skip.type = "button";
        skip.className = "satquery-tour-skip";
        skip.textContent = "Skip tour";
        skip.addEventListener("click", () => end("skipped"));
        popover.footer.prepend(skip);
        // Keyboard users land on the main action, not the page behind.
        window.setTimeout(() => popover.nextButton.focus(), 0);
      };
      for (const step of steps) {
        const own = step.popover?.onPopoverRender;
        if (step.popover) {
          step.popover.onPopoverRender = (popover, opts) => {
            frame(popover);
            own?.(popover, opts);
          };
        }
      }

      // Saved as each step begins, so a refresh mid-transition still resumes
      // there; a tour closed in the meantime stays closed.
      const saveStep = (index: number | undefined) => {
        if (!destroyed && index !== undefined) write(uid, { state: "in_progress", step: index });
      };

      const column = document.querySelector("[data-tour-scroll]");
      let settled = false;

      const d = driver({
        steps,
        animate: !reduceMotion,
        smoothScroll: !reduceMotion,
        allowClose: true,
        allowKeyboardControl: true,
        showProgress: true,
        progressText: "{{current}} of {{total}}",
        nextBtnText: "Next",
        prevBtnText: "Back",
        doneBtnText: "Finish",
        popoverClass: "satquery-tour",
        overlayOpacity: 0.72,
        stagePadding: 6,
        stageRadius: 4,
        waitForElement: 1500,
        skipMissingElement: true,
        onHighlightStarted: (_el, _step, { index }) => {
          settled = false;
          saveStep(index);
        },
        onHighlighted: () => {
          settled = true;
          d.refresh();
        },
        // Escape, the close button and a click on the overlay all land here.
        onDestroyStarted: () => end("skipped"),
      });
      driverRef.current = d;

      // driver.js tracks window scrolling only. The steps live in their own
      // scrolling column on desktop, so the highlight is re-measured when that
      // column scrolls. Re-measuring mid-transition would fight driver.js's
      // own animation, so it waits for a step to finish highlighting first.
      const remeasure = () => {
        if (!d.isActive()) return column?.removeEventListener("scroll", remeasure);
        if (settled) d.refresh();
      };
      column?.addEventListener("scroll", remeasure, { passive: true });

      write(uid, { state: "in_progress", step: from });
      d.drive(from);
    },
    [uid, actions],
  );

  // First visit: offer the tour. Mid-tour refresh: resume at a step that
  // makes sense for what is (not) on screen now.
  useEffect(() => {
    if (!uid) return;
    const stored = read(uid);
    const timer = window.setTimeout(() => {
      if (!stored) return setWelcomeOpen(true);
      if (stored.state !== "in_progress") return;
      let step = stored.step;
      const a = actions.current;
      if (step >= NEEDS_IMAGERY_FROM && !a.isBound()) step = 1;
      else if (step >= NEEDS_RESULT_FROM && !a.hasResult()) step = RUN_STEP;
      void start(step);
    }, 600);
    return () => window.clearTimeout(timer);
  }, [uid, start, actions]);

  useEffect(() => () => driverRef.current?.destroy(), []);

  const skip = useCallback(() => {
    if (uid) write(uid, { state: "skipped", step: 0 });
    setWelcomeOpen(false);
  }, [uid]);

  return { welcomeOpen, start, skip };
}

export function TourWelcome({
  open,
  onStart,
  onSkip,
}: {
  open: boolean;
  onStart: () => void;
  onSkip: () => void;
}) {
  return (
    <Dialog open={open} onOpenChange={(next) => (next ? undefined : onSkip())}>
      <DialogContent className="max-w-md rounded-(--radius-surface) border-rule-strong bg-panel">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold">Welcome to SatQuery</DialogTitle>
          <DialogDescription className="text-[14px] leading-relaxed text-text-dim">
            Ask questions about satellite imagery in plain English. The tour loads a sample,
            asks a question and walks you through the answer, in about a minute.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter className="gap-2 sm:justify-start">
          <Button
            onClick={onStart}
            autoFocus
            className="bg-accent-gradient h-9 rounded-sm px-4 text-[13px] font-semibold text-void hover:brightness-110"
          >
            Take a 60-second tour
          </Button>
          <Button
            variant="ghost"
            onClick={onSkip}
            className="h-9 rounded-sm text-[13px] text-text-dim hover:bg-raised hover:text-foreground"
          >
            Skip
          </Button>
        </DialogFooter>
        <p className="text-[12px] text-muted-foreground">
          You can start it again any time from the ? button at the top.
        </p>
      </DialogContent>
    </Dialog>
  );
}

export function TourHelpButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Take the tour"
      title="Take the tour"
      className="grid size-8 place-items-center rounded-sm text-text-dim transition-colors hover:bg-raised hover:text-foreground"
    >
      <CircleHelp className="size-4" aria-hidden />
    </button>
  );
}
