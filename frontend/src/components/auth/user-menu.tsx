"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { LogOut, UserPlus } from "lucide-react";
import { useAuth } from "@/components/auth/auth-provider";
import { UpgradeGuestDialog } from "@/components/auth/upgrade-guest-dialog";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

function initials(name: string | null, email: string | null): string {
  const source = name?.trim() || email?.split("@")[0] || "";
  const parts = source.split(/[\s._-]+/).filter(Boolean);
  const letters = parts.length > 1 ? parts[0][0] + parts[1][0] : source.slice(0, 2);
  return letters.toUpperCase() || "?";
}

export function UserMenu() {
  const { user, signOut } = useAuth();
  const router = useRouter();
  const [upgradeOpen, setUpgradeOpen] = useState(false);

  if (!user) return null;

  const guest = user.isAnonymous;
  const who = guest ? "Guest" : (user.email ?? user.displayName ?? "Signed in");

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger className="flex items-center gap-2 rounded-sm px-1 py-0.5 text-left outline-none transition-colors hover:bg-raised focus-visible:outline-1 focus-visible:outline-signal">
          <Avatar className="size-7 rounded-sm">
            {!guest && user.photoURL ? (
              <AvatarImage src={user.photoURL} alt="" referrerPolicy="no-referrer" />
            ) : null}
            <AvatarFallback className="rounded-sm bg-raised font-mono text-[10px] text-text-dim">
              {guest ? "G" : initials(user.displayName, user.email)}
            </AvatarFallback>
          </Avatar>
          <span className="hidden max-w-[16ch] truncate font-mono text-[11px] text-text-dim sm:inline">
            {who}
          </span>
          <span className="sr-only">Account menu</span>
        </DropdownMenuTrigger>

        <DropdownMenuContent
          align="end"
          className="w-64 rounded-sm border-rule bg-panel font-mono text-[11px]"
        >
          <DropdownMenuLabel className="space-y-0.5 font-normal">
            <span className="block truncate text-[12px] text-foreground">{who}</span>
            <span className="block text-[10px] text-muted-foreground">
              {guest ? "Guest session on this browser" : "Signed in"}
            </span>
          </DropdownMenuLabel>
          <DropdownMenuSeparator className="bg-rule" />

          {guest ? (
            <DropdownMenuItem
              onSelect={() => setUpgradeOpen(true)}
              className="gap-2 rounded-sm text-[11px] text-text-dim focus:bg-raised focus:text-signal"
            >
              <UserPlus className="size-3.5" aria-hidden />
              Save your history to an account
            </DropdownMenuItem>
          ) : null}

          <DropdownMenuItem
            onSelect={async () => {
              await signOut();
              router.replace("/");
            }}
            className="items-start gap-2 rounded-sm text-[11px] text-text-dim focus:bg-raised focus:text-foreground"
          >
            <LogOut className="mt-px size-3.5" aria-hidden />
            <span>
              Sign out
              {guest ? (
                <span className="mt-0.5 block text-[10px] leading-snug text-muted-foreground">
                  A guest session can&rsquo;t be reopened after signing out.
                </span>
              ) : null}
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {guest ? <UpgradeGuestDialog open={upgradeOpen} onOpenChange={setUpgradeOpen} /> : null}
    </>
  );
}
