"use client";
import { AlertTriangle, FlaskConical } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ago, directionLabel, fmtNum, fmtPct, fmtSigned, riskLevel } from "@/lib/format";
import type { Direction } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ScoreCell({ value, status, threshold = 70, size = "sm" }: { value: number | null | undefined; status?: string; threshold?: number; size?: "sm" | "lg" }) {
  if (value === null || value === undefined) {
    return (
      <span className="text-muted" title="Unavailable: not enough data to compute a defensible score">
        n/a
      </span>
    );
  }
  const lvl = riskLevel(value, threshold);
  return (
    <span className={cn("num inline-flex items-center gap-1", size === "lg" ? "text-2xl font-semibold" : "text-xs")}>
      <span className={cn(lvl === "high" && "text-risk", lvl === "elevated" && "text-warn")}>{fmtNum(value, 1)}</span>
      {status === "provisional" ? (
        <span className="text-[9px] font-sans uppercase text-warn" title="Provisional: some components unavailable; score re-normalised over available weight">
          prov
        </span>
      ) : null}
    </span>
  );
}

export function ScoreBar({ value, threshold = 70 }: { value: number | null | undefined; threshold?: number }) {
  const lvl = riskLevel(value, threshold);
  return (
    <div className="h-1 w-16 overflow-hidden rounded bg-panel-2">
      {value !== null && value !== undefined ? (
        <div
          className={cn("h-full", lvl === "high" ? "bg-risk" : lvl === "elevated" ? "bg-warn" : "bg-accent-2")}
          style={{ width: `${Math.max(2, Math.min(100, value))}%` }}
        />
      ) : null}
    </div>
  );
}

export function Delta({ value, digits = 1 }: { value: number | null | undefined; digits?: number }) {
  if (value === null || value === undefined) return <span className="text-muted">—</span>;
  return <span className={cn("num", value > 0.05 ? "text-risk" : value < -0.05 ? "text-ok" : "text-muted")}>{fmtSigned(value, digits)}</span>;
}

export function DirectionBadge({ d, a, b }: { d: Direction | string | null | undefined; a: string; b: string }) {
  if (!d || d === "not_applicable") return <span className="text-muted">—</span>;
  const variant = d === "uncertain" ? "warn" : d === "neutral" ? "default" : "blue";
  return <Badge variant={variant as "warn" | "default" | "blue"}>{directionLabel(d, a, b)}</Badge>;
}

export function ConfidenceMeter({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return <span className="text-muted">—</span>;
  const bars = Math.round(value * 5);
  return (
    <span className="inline-flex items-center gap-1.5" title="Evidence confidence (source quality, coverage, agreement, completeness, AI certainty, freshness). Not a probability.">
      <span className="flex gap-0.5">
        {[0, 1, 2, 3, 4].map((i) => (
          <span key={i} className={cn("h-2.5 w-1 rounded-sm", i < bars ? "bg-accent" : "bg-panel-2")} />
        ))}
      </span>
      <span className="num text-[11px] text-muted">{fmtPct(value)}</span>
    </span>
  );
}

export function StatusDot({ state }: { state: string }) {
  const color =
    state === "running" ? "bg-accent animate-pulse" : state === "idle" ? "bg-ok" : state === "paused" ? "bg-warn" : state === "failed" ? "bg-risk" : "bg-muted";
  return <span className={cn("inline-block size-2 rounded-full", color)} />;
}

export function AgeBadge({ ts, staleHours, label }: { ts: string | null | undefined; staleHours: number; label?: string }) {
  if (!ts) return <Badge variant="warn">{label ? `${label}: ` : ""}never</Badge>;
  const hrs = (Date.now() - new Date(ts).getTime()) / 3.6e6;
  return (
    <Badge variant={hrs > staleHours ? "warn" : "default"} title={new Date(ts).toISOString()}>
      {label ? `${label}: ` : ""}
      {ago(ts)}
      {hrs > staleHours ? " · stale" : ""}
    </Badge>
  );
}

export function SimulatedTag({ show = true }: { show?: boolean }) {
  if (!show) return null;
  return (
    <Badge variant="sim" title="Simulated demo data, not real market or news data">
      <FlaskConical className="size-3" /> Simulated
    </Badge>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const v = status === "success" || status === "analyzed" || status === "complete" ? "ok" : status === "failed" || status === "interrupted" ? "risk" : status === "running" ? "accent" : status === "queued" || status === "provisional" || status === "budget_blocked" || status === "not_configured" ? "warn" : "default";
  return <Badge variant={v as "ok"}>{status.replace(/_/g, " ")}</Badge>;
}

export function Notice({ kind = "warn", children }: { kind?: "warn" | "info" | "risk"; children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border px-3 py-2 text-xs",
        kind === "warn" && "border-warn/40 bg-warn/5 text-warn",
        kind === "risk" && "border-risk/40 bg-risk/5 text-risk",
        kind === "info" && "border-border bg-panel-2 text-muted",
      )}
    >
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
      <div>{children}</div>
    </div>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: "risk" | "warn" | "ok" }) {
  return (
    <div className="rounded-lg border border-border bg-panel px-3 py-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted">{label}</div>
      <div className={cn("num mt-1 text-xl font-semibold", tone === "risk" && "text-risk", tone === "warn" && "text-warn", tone === "ok" && "text-ok")}>{value}</div>
      {sub ? <div className="mt-0.5 truncate text-[11px] text-muted">{sub}</div> : null}
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="px-3 py-8 text-center text-xs text-muted">{children}</div>;
}
