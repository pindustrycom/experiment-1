"use client";
import { Activity, BarChart3, FlaskConical, Gauge, LogOut, Menu, Moon, Newspaper, Settings, Sun, X } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";
import { StatusDot } from "@/components/domain";
import { Button } from "@/components/ui/button";
import { apiSend, useApi } from "@/lib/api";
import { ago } from "@/lib/format";
import type { SystemStatus, User } from "@/lib/types";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Dashboard", icon: Gauge },
  { href: "/news", label: "News intelligence", icon: Newspaper },
  { href: "/performance", label: "Model performance", icon: BarChart3 },
  { href: "/bot", label: "Bot activity", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
];

const UserCtx = createContext<User | null>(null);
export const useUser = () => useContext(UserCtx);
export const useIsAdmin = () => useContext(UserCtx)?.role === "admin";

function ThemeToggle() {
  const [dark, setDark] = useState(true);
  useEffect(() => setDark(document.documentElement.classList.contains("dark")), []);
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => {
        const next = !dark;
        document.documentElement.classList.toggle("dark", next);
        try {
          localStorage.setItem("sm-theme", next ? "dark" : "light");
        } catch {}
        setDark(next);
      }}
    >
      {dark ? <Sun /> : <Moon />}
    </Button>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const { data: user } = useApi<User>("/auth/me");
  const { data: status } = useApi<SystemStatus>("/system/status", { refreshInterval: 30000 });

  useEffect(() => setOpen(false), [pathname]);

  async function logout() {
    await apiSend("/auth/logout", "POST").catch(() => null);
    router.replace("/login");
  }

  const demo = status?.mode === "demo";
  return (
    <UserCtx.Provider value={user ?? null}>
      <div className="flex min-h-screen">
        <aside
          className={cn(
            "fixed inset-y-0 left-0 z-40 w-52 border-r border-border bg-panel transition-transform lg:static lg:translate-x-0",
            open ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <div className="flex h-11 items-center gap-2 border-b border-border px-3">
            <svg viewBox="0 0 32 32" className="size-5" aria-hidden>
              <path d="M5 22 L12 14 L17 18 L27 8" stroke="var(--accent)" strokeWidth="2.5" fill="none" />
              <path d="M5 24 L12 19 L17 22 L27 20" stroke="var(--risk)" strokeWidth="2.5" fill="none" />
            </svg>
            <span className="text-xs font-semibold tracking-wide">SPLITTABILITY</span>
            <button className="ml-auto lg:hidden" onClick={() => setOpen(false)} aria-label="Close menu">
              <X className="size-4" />
            </button>
          </div>
          <nav className="space-y-0.5 p-2">
            {NAV.map(({ href, label, icon: Icon }) => {
              const active = href === "/" ? pathname === "/" || pathname.startsWith("/pairs") : pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs",
                    active ? "bg-panel-2 text-fg" : "text-muted hover:bg-panel-2 hover:text-fg",
                  )}
                >
                  <Icon className="size-3.5" />
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="absolute inset-x-0 bottom-0 space-y-1 border-t border-border p-3 text-[11px] text-muted">
            <div className="flex items-center gap-1.5">
              <StatusDot state={status?.bot.state ?? "unknown"} /> Bot {status?.bot.state ?? "…"}
            </div>
            <div>News {ago(status?.last_news_update)}</div>
            <div>Prices {ago(status?.last_price_update)}</div>
            <div className="truncate">{user?.email}</div>
          </div>
        </aside>
        {open ? <div className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setOpen(false)} /> : null}
        <div className="flex min-w-0 flex-1 flex-col">
          {demo ? (
            <div className="flex items-center justify-center gap-2 border-b border-sim/40 bg-sim/10 px-3 py-1 text-[11px] font-medium text-sim" role="status">
              <FlaskConical className="size-3" /> DEMO MODE — all prices, news, scores and predictions shown are SIMULATED. Configure API keys and set APP_MODE=production for real data.
            </div>
          ) : null}
          <header className="flex h-11 items-center gap-2 border-b border-border bg-panel px-3">
            <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu">
              <Menu />
            </Button>
            <div className="text-xs text-muted">{NAV.find((n) => (n.href === "/" ? pathname === "/" : pathname.startsWith(n.href)))?.label ?? "Pair detail"}</div>
            <div className="ml-auto flex items-center gap-1">
              {status ? (
                <span className="hidden items-center gap-1.5 text-[11px] text-muted sm:flex">
                  <StatusDot state={status.bot.state} />
                  {status.bot.paused ? "monitoring paused" : status.bot.running_jobs.length ? `running ${status.bot.running_jobs.join(", ")}` : "monitoring"}
                </span>
              ) : null}
              <ThemeToggle />
              <Button variant="ghost" size="icon" onClick={logout} aria-label="Sign out">
                <LogOut />
              </Button>
            </div>
          </header>
          <main className="min-w-0 flex-1 p-3 sm:p-4">{children}</main>
        </div>
      </div>
    </UserCtx.Provider>
  );
}
