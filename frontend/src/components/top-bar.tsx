"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { UserMenu } from "@/components/auth/user-menu";
import { StatusDot } from "@/components/status-dot";
import { Wordmark } from "@/components/landing/wordmark";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/console", label: "Console" },
  { href: "/history", label: "History" },
];

export function TopBar({ busy = false, children }: { busy?: boolean; children?: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-rule bg-panel px-3 sm:gap-4 sm:px-4">
      <Link href="/" className="rounded-sm" aria-label="SatQuery AI home">
        <Wordmark />
      </Link>

      <nav aria-label="Console" className="flex items-center gap-1">
        {NAV.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            aria-current={pathname === item.href ? "page" : undefined}
            className={cn(
              "rounded-sm px-2 py-1 text-[13px] transition-colors",
              pathname === item.href ? "text-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="ml-auto flex items-center gap-3">
        {busy ? (
          <span className="hidden items-center gap-2 text-[12px] text-text-dim sm:flex" role="status">
            <StatusDot tone="signal" pulse />
            Running
          </span>
        ) : null}
        {children}
        <UserMenu />
      </div>
    </header>
  );
}
