"use client";

import { UserMenu } from "@/components/auth/user-menu";
import { Badge } from "@/components/ui/badge";
import { StatusDot } from "@/components/status-dot";
import { INTENT_CODES, INTENT_LABELS } from "@/lib/suggestions";
import type { Intent, SceneMeta } from "@/lib/types";

export function TopBar({
  scene,
  intent,
  busy,
}: {
  scene: SceneMeta | null;
  intent: Intent | null;
  busy: boolean;
}) {
  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-rule bg-panel px-4">
      <div className="flex items-baseline gap-2">
        <span className="text-[15px] font-semibold tracking-tight text-foreground">
          SatQuery
        </span>
        <span className="font-mono text-[10px] text-signal">AI</span>
      </div>

      <span className="h-4 w-px bg-rule" aria-hidden />

      <p className="truncate font-mono text-[11px] text-muted-foreground">
        {scene ? scene.sceneId : "no scene bound"}
      </p>

      <div className="ml-auto flex items-center gap-3">
        {intent ? (
          <Badge
            variant="outline"
            className="gap-1.5 rounded-sm border-signal-deep bg-signal-deep/40 px-2 font-mono text-[10px] font-medium text-signal"
          >
            {INTENT_CODES[intent]}
            <span className="text-signal/60">{INTENT_LABELS[intent]}</span>
          </Badge>
        ) : null}

        <div className="flex items-center gap-2 font-mono text-[10px] tracking-[0.12em] text-muted-foreground">
          <StatusDot tone={busy ? "signal" : "muted"} pulse={busy} />
          {busy ? "RUNNING" : "IDLE"}
        </div>

        <span className="h-4 w-px bg-rule" aria-hidden />
        <UserMenu />
      </div>
    </header>
  );
}
