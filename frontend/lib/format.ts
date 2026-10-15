import type { Direction } from "./types";

export function fmtNum(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtSigned(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
}

export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtUsd(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v < 1 ? `$${v.toFixed(3)}` : `$${v.toFixed(2)}`;
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return new Intl.NumberFormat("en-US").format(v);
}

export function fmtDate(s: string | null | undefined): string {
  if (!s) return "—";
  return new Date(s).toLocaleDateString("en-CA");
}

export function fmtDateTime(s: string | null | undefined): string {
  if (!s) return "—";
  const d = new Date(s);
  return `${d.toLocaleDateString("en-CA")} ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`;
}

export function ago(s: string | null | undefined, now: Date = new Date()): string {
  if (!s) return "never";
  const sec = Math.max(0, (now.getTime() - new Date(s).getTime()) / 1000);
  if (sec < 90) return "just now";
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400 * 2) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

export function directionLabel(d: Direction | string | null | undefined, a: string, b: string): string {
  switch (d) {
    case "a_advantaged":
      return `${a} advantaged`;
    case "b_advantaged":
      return `${b} advantaged`;
    case "neutral":
      return "Neutral";
    case "uncertain":
      return "Uncertain";
    default:
      return "—";
  }
}

export type RiskLevel = "high" | "elevated" | "normal" | "none";

export function riskLevel(v: number | null | undefined, threshold = 70): RiskLevel {
  if (v === null || v === undefined) return "none";
  if (v >= threshold) return "high";
  if (v >= threshold - 15) return "elevated";
  return "normal";
}

export const COMPONENT_NAMES: Record<string, string> = {
  N: "Asymmetric news risk",
  F: "Fundamental divergence",
  M: "Market relationship deterioration",
  P: "Event persistence",
  S: "Structural exposure differences",
  C: "Correlation weakness",
  D: "Correlation deterioration",
  R: "Relative-price divergence",
};

export function eventLabel(t: string): string {
  return t.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** Format a timestamp in a named IANA zone (e.g. scheduler times in America/New_York). */
export function fmtInZone(s: string | null | undefined, timeZone: string): string {
  if (!s) return "—";
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZoneName: "short" }).formatToParts(new Date(s));
  const g = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  return `${g("year")}-${g("month")}-${g("day")} ${g("hour")}:${g("minute")} ${g("timeZoneName")}`;
}
