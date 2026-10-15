"use client";
import { RefreshCw } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AddPairDialog } from "@/components/add-pair-dialog";
import { AgeBadge, Delta, Empty, Notice, Stat, StatusBadge } from "@/components/domain";
import { PairTable } from "@/components/pair-table";
import { useIsAdmin } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useApi } from "@/lib/api";
import { ago, fmtInt, fmtNum, fmtUsd } from "@/lib/format";
import type { Page, Pair, SystemStatus, Watchlist } from "@/lib/types";

interface Distribution {
  histogram: { bin: string; si_v1: number; si_v2: number }[];
  monthly_mean: { month: string; si_v1: number | null; si_v2: number | null }[];
  n: number;
}
interface Mover { slug: string; si_v2: number | null; change_7d: number; direction: string }
interface AlertRow { id: number; pair: string | null; kind: string; severity: string; message: string; created_at: string }

const tooltipStyle = { background: "var(--panel)", border: "1px solid var(--border)", fontSize: 11, color: "var(--fg)" };

export default function Dashboard() {
  const admin = useIsAdmin();
  const [status, setStatus] = useState<"monitored" | "active" | "disabled" | "archived" | "all">("monitored");
  const [watchlist, setWatchlist] = useState<string>("");
  const [industry, setIndustry] = useState<string>("");
  const [q, setQ] = useState("");
  const qs = new URLSearchParams({ status, page_size: "500" });
  if (watchlist) qs.set("watchlist_id", watchlist);
  const pairs = useApi<Page<Pair>>(`/pairs?${qs}`, { refreshInterval: 60000 });
  const sys = useApi<SystemStatus>("/system/status", { refreshInterval: 30000 });
  const dist = useApi<Distribution>("/system/score-distribution?days=730");
  const movers = useApi<Mover[]>("/system/top-movers?limit=5");
  const wls = useApi<Watchlist[]>("/watchlists");
  const alerts = useApi<AlertRow[]>("/alerts?limit=6");

  const industries = useMemo(() => Array.from(new Set((pairs.data?.items ?? []).map((p) => p.industry).filter(Boolean))) as string[], [pairs.data]);
  const filtered = useMemo(() => {
    const term = q.trim().toUpperCase();
    return (pairs.data?.items ?? []).filter(
      (p) =>
        (!industry || p.industry === industry) &&
        (!term || p.slug.includes(term) || p.company_a.name.toUpperCase().includes(term) || p.company_b.name.toUpperCase().includes(term)),
    );
  }, [pairs.data, industry, q]);

  const s = sys.data;
  return (
    <div className="space-y-3">
      {s?.demo_data_present ? (
        <Notice kind="risk">
          Production mode, but this database still contains SIMULATED demo rows. The worker will not run until they are removed with <code>python -m app.cli wipe-demo --yes</code>.
        </Notice>
      ) : null}
      {s && s.mode === "production" && (!s.providers.anthropic || (!s.providers.tiingo && !s.providers.stooq)) ? (
        <Notice>
          Some integrations are not configured ({Object.entries(s.providers).filter(([, v]) => !v).map(([k]) => k).join(", ")}). Affected components are reported as unavailable rather than scored.
        </Notice>
      ) : null}
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-8">
        {s ? (
          <>
            <Stat label="Monitored pairs" value={s.pairs.monitored} sub={`${s.pairs.scored_si_v2} with SI-v2`} />
            <Stat label="Avg SI-v2" value={fmtNum(s.pairs.avg_si_v2)} sub={`SI-v1 avg ${fmtNum(s.pairs.avg_si_v1)}`} />
            <Stat label="High-risk pairs" value={s.pairs.high_risk} tone={s.pairs.high_risk ? "risk" : undefined} sub="SI-v2 ≥ pair threshold" />
            <Stat label="Significant events" value={s.significant_events_7d} sub="material, severity ≥ 0.4, 7d" tone={s.significant_events_7d ? "warn" : undefined} />
            <Stat label="Bot" value={<span className="capitalize">{s.bot.state}</span>} sub={s.bot.queued ? `${s.bot.queued} queued` : `active ${ago(s.bot.last_activity)}`} tone={s.bot.paused ? "warn" : undefined} />
            <Stat label="News update" value={ago(s.last_news_update)} sub={s.pending_analysis ? `${s.pending_analysis} stories awaiting analysis` : "all analysed"} tone={(s.news_age_hours ?? 99) > 3 ? "warn" : undefined} />
            <Stat label="Price update" value={ago(s.last_price_update)} sub="after US close" tone={(s.price_age_hours ?? 999) > 80 ? "warn" : undefined} />
            <Stat label="AI cost (month)" value={fmtUsd(s.ai.month_cost_usd)} sub={`limit ${fmtUsd(s.ai.monthly_limit_usd)} · ${fmtInt(s.ai.input_tokens + s.ai.output_tokens)} tok`} tone={s.ai.month_cost_usd > s.ai.monthly_limit_usd * 0.8 ? "warn" : undefined} />
          </>
        ) : (
          Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-[72px]" />)
        )}
      </div>

      <div className="grid gap-3 xl:grid-cols-[1fr_320px]">
        <Card>
          <CardHeader className="flex-wrap">
            <CardTitle>Watchlist</CardTitle>
            <div className="flex flex-wrap items-center gap-1.5">
              <Input placeholder="Search pair or company" value={q} onChange={(e) => setQ(e.target.value)} className="h-7 w-44" aria-label="Search" />
              <Select value={watchlist} onChange={(e) => setWatchlist(e.target.value)} className="h-7" aria-label="Watchlist">
                <option value="">All watchlists</option>
                {(wls.data ?? []).map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.name}
                  </option>
                ))}
              </Select>
              <Select value={industry} onChange={(e) => setIndustry(e.target.value)} className="h-7" aria-label="Industry">
                <option value="">All industries</option>
                {industries.map((i) => (
                  <option key={i}>{i}</option>
                ))}
              </Select>
              <Select value={status} onChange={(e) => setStatus(e.target.value as typeof status)} className="h-7" aria-label="Status">
                <option value="monitored">Active + disabled</option>
                <option value="active">Active</option>
                <option value="disabled">Disabled</option>
                <option value="archived">Archived</option>
                <option value="all">All</option>
              </Select>
              <Button variant="ghost" size="icon" onClick={() => pairs.mutate()} aria-label="Refresh">
                <RefreshCw />
              </Button>
              {admin ? <AddPairDialog watchlists={wls.data ?? []} onCreated={() => pairs.mutate()} /> : null}
            </div>
          </CardHeader>
          {pairs.error ? <Empty>Failed to load pairs: {String(pairs.error.message)}</Empty> : pairs.data ? <PairTable pairs={filtered} /> : <Skeleton className="m-3 h-64" />}
          <div className="border-t border-border px-3 py-1.5 text-[10px] text-muted">
            SI values are relative risk scores (0–100), not probabilities. “prov” = provisional (some components unavailable). n/a = not enough data for a defensible score.
          </div>
        </Card>

        <div className="space-y-3">
          <Card>
            <CardHeader>
              <CardTitle>Largest 7-day increases</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1.5 py-2">
              {(movers.data ?? []).length === 0 ? (
                <p className="text-xs text-muted">No score history yet.</p>
              ) : (
                movers.data!.map((m) => (
                  <Link key={m.slug} href={`/pairs/${m.slug}`} className="flex items-center justify-between text-xs hover:text-accent">
                    <span className="font-mono">{m.slug.replace("-", " / ")}</span>
                    <span className="flex items-center gap-2">
                      <span className="num text-muted">{fmtNum(m.si_v2)}</span>
                      <Delta value={m.change_7d} />
                    </span>
                  </Link>
                ))
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Open alerts</CardTitle>
              <Badge>{s?.open_alerts ?? 0}</Badge>
            </CardHeader>
            <CardContent className="space-y-2 py-2">
              {(alerts.data ?? []).length === 0 ? (
                <p className="text-xs text-muted">No open alerts.</p>
              ) : (
                alerts.data!.map((a) => (
                  <div key={a.id} className="text-xs">
                    <div className="flex items-center gap-1.5">
                      <Badge variant={a.severity === "critical" ? "risk" : a.severity === "warning" ? "warn" : "default"}>{a.severity}</Badge>
                      <span className="text-muted">{ago(a.created_at)}</span>
                    </div>
                    <div className="mt-0.5">{a.message}</div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Data freshness</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-1.5 py-2">
              <AgeBadge ts={s?.last_news_update} staleHours={3} label="news" />
              <AgeBadge ts={s?.last_price_update} staleHours={80} label="prices" />
              {s ? <StatusBadge status={s.ai.configured ? "success" : "not_configured"} /> : null}
              <span className="text-[11px] text-muted">Claude: {s?.ai.model}</span>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Score distribution (stored scores, 2y)</CardTitle>
            <span className="text-[10px] text-muted">n = {fmtInt(dist.data?.n)}</span>
          </CardHeader>
          <CardContent className="h-56">
            {dist.data && dist.data.n > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dist.data.histogram} margin={{ left: -20, right: 4, top: 4 }}>
                  <CartesianGrid strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="bin" tickLine={false} />
                  <YAxis tickLine={false} allowDecimals={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--panel-2)" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="si_v1" name="SI-v1" fill="var(--muted)" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="si_v2" name="SI-v2" fill="var(--accent)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Empty>No scores stored yet.</Empty>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Average score by month</CardTitle>
          </CardHeader>
          <CardContent className="h-56">
            {dist.data && dist.data.monthly_mean.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dist.data.monthly_mean} margin={{ left: -20, right: 8, top: 4 }}>
                  <CartesianGrid strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="month" tickLine={false} minTickGap={24} />
                  <YAxis tickLine={false} domain={[0, 100]} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="si_v1" name="SI-v1" stroke="var(--muted)" dot={false} strokeWidth={1.5} connectNulls />
                  <Line dataKey="si_v2" name="SI-v2" stroke="var(--accent)" dot={false} strokeWidth={1.5} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Empty>No scores stored yet.</Empty>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
