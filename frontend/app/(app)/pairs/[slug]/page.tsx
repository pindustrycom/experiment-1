"use client";
import { Archive, ArrowLeft, Pencil } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useMemo, useState } from "react";
import { ArticleCard } from "@/components/article-card";
import { ConfidenceMeter, Delta, DirectionBadge, Empty, Notice, ScoreCell, SimulatedTag, StatusBadge } from "@/components/domain";
import { NormalizedPrices, RollingCorr, ScoreHistory, SeriesArea, ZScore, type HistoryPoint } from "@/components/pair-charts";
import { useIsAdmin } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Input, Label, Select, Textarea } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiSend, useApi } from "@/lib/api";
import { COMPONENT_NAMES, fmtDateTime, fmtNum, fmtPct } from "@/lib/format";
import type { Article, Page, Pair, Prediction, Score } from "@/lib/types";
import { cn } from "@/lib/utils";

interface PriceData {
  available: boolean;
  reason?: string;
  provider: string;
  is_total_return: boolean;
  is_demo: boolean;
  ticker_a: string;
  ticker_b: string;
  last_date: string;
  expected_last_session: string;
  stale: boolean;
  sessions_behind: number;
  normalized: { date: string; a: number; b: number }[];
  relative_return: { date: string; value: number | null }[];
  rolling_corr_63: { date: string; value: number | null }[];
  rolling_corr_252: { date: string; value: number | null }[];
  zscore: { date: string; z: number; spread: number; beta: number }[];
  cointegration: Record<string, { p_value: number | null; hedge_ratio: number | null; window_start: string; window_end: string }>;
  method: Record<string, string>;
}
interface EventMarker { id: number; date: string; event_type: string; direction: string; severity: number; headline: string }

function ComponentRows({ comps }: { comps: Score["components"] }) {
  if (!comps) return null;
  return (
    <div className="space-y-2">
      {comps.map((c) => (
        <div key={c.component} className="text-xs">
          <div className="flex items-center justify-between gap-2">
            <span>
              <span className="font-mono font-semibold">{c.component}</span> <span className="text-muted">{COMPONENT_NAMES[c.component]}</span>
            </span>
            <span className="num flex items-center gap-2">
              <span className="text-muted">w {c.weight < 1 ? fmtPct(c.weight) : c.weight}</span>
              {c.available ? fmtNum(c.value, 3) : <Badge variant="warn">unavailable</Badge>}
            </span>
          </div>
          <div className="mt-1 h-1 rounded bg-panel-2">
            {c.available && c.value !== null ? <div className="h-full rounded bg-accent" style={{ width: `${Math.max(1, c.value * 100)}%` }} /> : null}
          </div>
          <div className="mt-0.5 text-[11px] text-muted">{c.explanation}</div>
        </div>
      ))}
    </div>
  );
}

function ScoreInspector({ slug, scoreId, onClose }: { slug: string; scoreId: number | null; onClose: () => void }) {
  const { data } = useApi<Score>(scoreId ? `/pairs/${slug}/scores/${scoreId}` : null);
  return (
    <Dialog open={scoreId !== null} onOpenChange={(o) => !o && onClose()}>
      <DialogContent title={data ? `${data.index_name.toUpperCase()} · ${fmtDateTime(data.as_of)}` : "Score"} description={data ? `Model ${data.model_version} · trigger ${data.trigger}` : undefined} className="max-w-2xl">
        {data ? (
          <div className="space-y-3 text-xs">
            <div className="flex flex-wrap items-center gap-3">
              <ScoreCell value={data.value} status={data.status} size="lg" />
              <StatusBadge status={data.status} />
              <span className="text-muted">coverage {fmtPct(data.coverage)}</span>
              {data.confidence !== null ? <ConfidenceMeter value={data.confidence} /> : null}
              <SimulatedTag show={data.is_demo} />
            </div>
            <p>{data.explanation}</p>
            <ComponentRows comps={data.components} />
            {data.confidence_breakdown?.parts ? (
              <div>
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Confidence breakdown</div>
                <div className="grid grid-cols-2 gap-1 sm:grid-cols-3">
                  {Object.entries(data.confidence_breakdown.parts).map(([k, v]) => (
                    <div key={k}>
                      <span className="text-muted">{k.replace(/_/g, " ")}</span> <span className="num">{v === null ? "n/a" : fmtNum(v, 2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            {data.supporting_event_ids.length ? <div className="text-muted">Supporting event IDs: {data.supporting_event_ids.join(", ")}</div> : null}
            <pre className="max-h-48 overflow-auto rounded bg-panel-2 p-2 text-[10px] text-muted">{JSON.stringify(data.components?.map((c) => ({ [c.component]: c.inputs })), null, 1)}</pre>
          </div>
        ) : (
          <Skeleton className="h-40" />
        )}
      </DialogContent>
    </Dialog>
  );
}

function EditPair({ pair, onSaved }: { pair: Pair; onSaved: () => void }) {
  const [open, setOpen] = useState(false);
  const [f, setF] = useState({ industry: pair.industry ?? "", status: pair.status, alert_threshold: String(pair.alert_threshold), alert_change_threshold: String(pair.alert_change_threshold), monitoring_interval_minutes: String(pair.monitoring_interval_minutes), notes: pair.notes ?? "" });
  const [err, setErr] = useState<string | null>(null);
  async function save() {
    try {
      await apiSend(`/pairs/${pair.id}`, "PATCH", {
        industry: f.industry || null,
        status: f.status,
        alert_threshold: Number(f.alert_threshold),
        alert_change_threshold: Number(f.alert_change_threshold),
        monitoring_interval_minutes: Number(f.monitoring_interval_minutes),
        notes: f.notes || null,
      });
      setOpen(false);
      onSaved();
    } catch (e) {
      setErr((e as Error).message);
    }
  }
  return (
    <>
      <Button variant="secondary" size="sm" onClick={() => setOpen(true)}>
        <Pencil /> Edit
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent title={`Edit ${pair.slug}`} description="Archiving keeps every historical score and prediction.">
          <div className="grid grid-cols-2 gap-3">
            <div className="col-span-2 space-y-1">
              <Label>Industry</Label>
              <Input value={f.industry} onChange={(e) => setF({ ...f, industry: e.target.value })} />
            </div>
            <div className="space-y-1">
              <Label>Status</Label>
              <Select value={f.status} onChange={(e) => setF({ ...f, status: e.target.value as Pair["status"] })} className="w-full">
                <option value="active">active</option>
                <option value="disabled">disabled (kept, not monitored)</option>
                <option value="archived">archived</option>
              </Select>
            </div>
            <div className="space-y-1">
              <Label>News interval (min)</Label>
              <Input type="number" min={15} value={f.monitoring_interval_minutes} onChange={(e) => setF({ ...f, monitoring_interval_minutes: e.target.value })} />
            </div>
            <div className="space-y-1">
              <Label>Alert threshold</Label>
              <Input type="number" min={0} max={100} value={f.alert_threshold} onChange={(e) => setF({ ...f, alert_threshold: e.target.value })} />
            </div>
            <div className="space-y-1">
              <Label>7-day change alert</Label>
              <Input type="number" min={0} max={100} value={f.alert_change_threshold} onChange={(e) => setF({ ...f, alert_change_threshold: e.target.value })} />
            </div>
            <div className="col-span-2 space-y-1">
              <Label>Notes</Label>
              <Textarea rows={3} value={f.notes} onChange={(e) => setF({ ...f, notes: e.target.value })} />
            </div>
            {err ? <p className="col-span-2 text-xs text-risk">{err}</p> : null}
            <div className="col-span-2 flex justify-end">
              <Button onClick={save}>Save</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

function Calibration({ preds }: { preds: Prediction[] }) {
  const resolved = preds.filter((p) => p.outcome && p.si_v2 !== null);
  if (resolved.length < 20) return <p className="text-xs text-muted">Calibration needs ≥ 20 resolved predictions for this pair (have {resolved.length}).</p>;
  const bins = [0, 20, 40, 60, 80, 100];
  const rows = bins.slice(0, -1).map((lo, i) => {
    const hi = bins[i + 1];
    const inBin = resolved.filter((p) => p.si_v2! >= lo && (i === bins.length - 2 ? p.si_v2! <= hi : p.si_v2! < hi));
    return { bin: `${lo}–${hi}`, n: inBin.length, rate: inBin.length ? inBin.filter((p) => p.outcome!.diverged).length / inBin.length : null };
  });
  return (
    <Table>
      <THead>
        <TR>
          <TH>SI-v2 bin</TH>
          <TH className="text-right">n</TH>
          <TH className="text-right">Observed divergence rate</TH>
        </TR>
      </THead>
      <TBody>
        {rows.map((r) => (
          <TR key={r.bin}>
            <TD className="num">{r.bin}</TD>
            <TD className="num text-right">{r.n}</TD>
            <TD className="num text-right">{r.rate === null ? "—" : fmtPct(r.rate)}</TD>
          </TR>
        ))}
      </TBody>
    </Table>
  );
}

export default function PairPage() {
  const { slug } = useParams<{ slug: string }>();
  const admin = useIsAdmin();
  const pair = useApi<Pair>(`/pairs/${slug}`);
  const [years, setYears] = useState(3);
  const prices = useApi<PriceData>(`/pairs/${slug}/prices?years=${years}`);
  const scores = useApi<Score[]>(`/pairs/${slug}/scores?days=1500&limit=5000`);
  const latestV2 = useApi<Score[]>(`/pairs/${slug}/scores?index=si_v2&days=30&detail=true`);
  const latestV1 = useApi<Score[]>(`/pairs/${slug}/scores?index=si_v1&days=30&detail=true`);
  const events = useApi<EventMarker[]>(`/pairs/${slug}/events?days=1500`);
  const [newsPage, setNewsPage] = useState(1);
  const news = useApi<Page<Article>>(`/pairs/${slug}/news?page=${newsPage}&page_size=15`);
  const [horizon, setHorizon] = useState<string>("");
  const [predState, setPredState] = useState("all");
  const preds = useApi<Page<Prediction>>(`/pairs/${slug}/predictions?page_size=500&state=${predState}${horizon ? `&horizon=${horizon}` : ""}`);
  const [inspect, setInspect] = useState<number | null>(null);

  const history: HistoryPoint[] = useMemo(() => {
    const m = new Map<number, HistoryPoint>();
    for (const s of scores.data ?? []) {
      const t = new Date(s.as_of).getTime();
      const p = m.get(t) ?? { t };
      if (s.index_name === "si_v1") {
        p.si_v1 = s.value;
        p.v1id = s.id;
      } else {
        p.si_v2 = s.value;
        p.v2id = s.id;
      }
      m.set(t, p);
    }
    return Array.from(m.values()).sort((a, b) => a.t - b.t);
  }, [scores.data]);
  const markers = (events.data ?? []).map((e) => ({ t: new Date(e.date).getTime(), y: 4 + e.severity * 10, label: e.headline }));

  if (pair.error) return <Empty>Pair not found.</Empty>;
  const p = pair.data;
  const v2 = latestV2.data?.at(-1);
  const v1 = latestV1.data?.at(-1);
  const a = p?.company_a.ticker ?? "A";
  const b = p?.company_b.ticker ?? "B";

  async function archive() {
    if (!p) return;
    await apiSend(`/pairs/${p.id}`, "DELETE");
    pair.mutate();
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Link href="/" className="text-muted hover:text-fg" aria-label="Back">
          <ArrowLeft className="size-4" />
        </Link>
        {p ? (
          <>
            <h1 className="font-mono text-base font-semibold">
              {a} / {b}
            </h1>
            <span className="text-xs text-muted">
              {p.company_a.name} · {p.company_b.name}
            </span>
            <Badge>{p.industry ?? "Unclassified"}</Badge>
            {p.status !== "active" ? <Badge variant="warn">{p.status}</Badge> : null}
            <SimulatedTag show={p.is_demo} />
            <span className="ml-auto flex gap-1.5">
              {admin ? <EditPair pair={p} onSaved={() => pair.mutate()} /> : null}
              {admin && p.status !== "archived" ? (
                <Button variant="ghost" size="sm" onClick={archive} title="Archive (history is kept)">
                  <Archive /> Archive
                </Button>
              ) : null}
            </span>
          </>
        ) : (
          <Skeleton className="h-6 w-64" />
        )}
      </div>

      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="prices">Price relationship</TabsTrigger>
          <TabsTrigger value="history">Splittability history</TabsTrigger>
          <TabsTrigger value="research">AI research</TabsTrigger>
          <TabsTrigger value="predictions">Prediction tracking</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Card>
              <CardHeader>
                <CardTitle>Current indices</CardTitle>
                <span className="text-[10px] text-muted">as of {fmtDateTime(v2?.as_of ?? v1?.as_of)}</span>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-muted">SI-v2</div>
                    <ScoreCell value={v2?.value} status={v2?.status} threshold={p?.alert_threshold} size="lg" />
                    <div className="text-[10px] text-muted">
                      7d <Delta value={p?.si_v2?.change_7d} /> · {v2?.status ?? "—"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-muted">SI-v1 (benchmark)</div>
                    <ScoreCell value={v1?.value} threshold={p?.alert_threshold} size="lg" />
                    <div className="text-[10px] text-muted">
                      7d <Delta value={p?.si_v1?.change_7d} />
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-muted">Directional bias</div>
                    <div className="mt-1.5">
                      <DirectionBadge d={v2?.direction} a={a} b={b} />
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-muted">Evidence confidence</div>
                    <div className="mt-1.5">
                      <ConfidenceMeter value={v2?.confidence} />
                    </div>
                  </div>
                </div>
                <p className="text-xs leading-relaxed">{v2?.explanation ?? "No SI-v2 score yet."}</p>
                <p className="text-[10px] text-muted">
                  Explanation generated deterministically from the stored components and Claude's structured event interpretations. SI values are relative risk scores, not probabilities.
                  Model {v2?.model_version ?? "—"}.
                </p>
                {v2?.direction_detail?.reason ? <Notice kind="info">Bias reasoning: {String(v2.direction_detail.reason)}</Notice> : null}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>SI-v2 components</CardTitle>
                {v2 ? (
                  <Button variant="ghost" size="sm" onClick={() => setInspect(v2.id)}>
                    Inspect evidence
                  </Button>
                ) : null}
              </CardHeader>
              <CardContent>{v2 ? <ComponentRows comps={v2.components} /> : <Skeleton className="h-40" />}</CardContent>
            </Card>
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>SI-v1 components (market-only benchmark)</CardTitle>
              </CardHeader>
              <CardContent>{v1 ? <ComponentRows comps={v1.components} /> : <Empty>No SI-v1 score.</Empty>}</CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="prices">
          {prices.data && !prices.data.available ? (
            <Notice>{prices.data.reason}</Notice>
          ) : prices.data ? (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <Select value={years} onChange={(e) => setYears(Number(e.target.value))} aria-label="Range">
                  {[1, 2, 3, 5, 10].map((y) => (
                    <option key={y} value={y}>
                      {y}y
                    </option>
                  ))}
                </Select>
                <Badge>{prices.data.provider}</Badge>
                <Badge variant={prices.data.is_total_return ? "ok" : "warn"}>{prices.data.is_total_return ? "total return (dividend-adjusted)" : "price return only"}</Badge>
                <span className="text-muted">last close {prices.data.last_date}</span>
                {prices.data.stale ? <Badge variant="warn">stale: {prices.data.sessions_behind} session(s) behind — showing last known data</Badge> : null}
                <SimulatedTag show={prices.data.is_demo} />
              </div>
              <div className="grid gap-3 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Normalized price (start = 100)</CardTitle>
                  </CardHeader>
                  <CardContent className="h-60">
                    <NormalizedPrices data={prices.data.normalized} a={a} b={b} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Relative total return ln(A/A₀) − ln(B/B₀)</CardTitle>
                  </CardHeader>
                  <CardContent className="h-60">
                    <SeriesArea data={prices.data.relative_return} name="log gap" threshold={0.2} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Rolling return correlation</CardTitle>
                  </CardHeader>
                  <CardContent className="h-60">
                    <RollingCorr
                      data={prices.data.rolling_corr_252.map((r, i) => ({ date: r.date, c252: r.value, c63: prices.data!.rolling_corr_63.find((x) => x.date === r.date)?.value ?? null, i }))}
                    />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Spread z-score (walk-forward hedge ratio)</CardTitle>
                  </CardHeader>
                  <CardContent className="h-60">{prices.data.zscore.length ? <ZScore data={prices.data.zscore} /> : <Empty>Not enough history.</Empty>}</CardContent>
                </Card>
              </div>
              <Card>
                <CardHeader>
                  <CardTitle>Cointegration (Engle–Granger on log prices) & hedge ratio</CardTitle>
                </CardHeader>
                <Table>
                  <THead>
                    <TR>
                      <TH>Window</TH>
                      <TH>Period</TH>
                      <TH className="text-right">Hedge ratio β</TH>
                      <TH className="text-right">p-value</TH>
                      <TH>Reading</TH>
                    </TR>
                  </THead>
                  <TBody>
                    {Object.entries(prices.data.cointegration).map(([w, c]) => (
                      <TR key={w}>
                        <TD className="num">{w} sessions</TD>
                        <TD className="num text-muted">
                          {c.window_start} → {c.window_end}
                        </TD>
                        <TD className="num text-right">{fmtNum(c.hedge_ratio, 3)}</TD>
                        <TD className="num text-right">{fmtNum(c.p_value, 3)}</TD>
                        <TD>{c.p_value === null ? "—" : c.p_value < 0.05 ? <Badge variant="ok">cointegrated at 5%</Badge> : <Badge variant="warn">no evidence of cointegration</Badge>}</TD>
                      </TR>
                    ))}
                  </TBody>
                </Table>
                <div className="space-y-0.5 border-t border-border px-3 py-2 text-[10px] text-muted">
                  {Object.entries(prices.data.method).map(([k, v]) => (
                    <div key={k}>
                      <span className="font-semibold">{k}:</span> {v}
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          ) : (
            <Skeleton className="h-96" />
          )}
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>SI-v1 and SI-v2 over time · ◆ material events · click a point to inspect evidence</CardTitle>
            </CardHeader>
            <CardContent className="h-80">
              {history.length ? <ScoreHistory data={history} events={markers} onSelect={setInspect} threshold={p?.alert_threshold ?? 70} /> : <Empty>No score history.</Empty>}
            </CardContent>
          </Card>
          <Card className="mt-3">
            <CardHeader>
              <CardTitle>Recent scores</CardTitle>
            </CardHeader>
            <Table>
              <THead>
                <TR>
                  <TH>As of</TH>
                  <TH>Index</TH>
                  <TH className="text-right">Value</TH>
                  <TH>Status</TH>
                  <TH>Trigger</TH>
                  <TH>Model</TH>
                  <TH></TH>
                </TR>
              </THead>
              <TBody>
                {(scores.data ?? [])
                  .slice(-40)
                  .reverse()
                  .map((s) => (
                    <TR key={s.id}>
                      <TD className="num">{fmtDateTime(s.as_of)}</TD>
                      <TD className="font-mono">{s.index_name}</TD>
                      <TD className="text-right">
                        <ScoreCell value={s.value} status={s.status} />
                      </TD>
                      <TD>
                        <StatusBadge status={s.status} />
                      </TD>
                      <TD className="text-muted">{s.trigger}</TD>
                      <TD className="font-mono text-[10px] text-muted">{s.model_version}</TD>
                      <TD>
                        <Button variant="ghost" size="sm" onClick={() => setInspect(s.id)}>
                          Inspect
                        </Button>
                      </TD>
                    </TR>
                  ))}
              </TBody>
            </Table>
          </Card>
        </TabsContent>

        <TabsContent value="research">
          <Card>
            <CardHeader>
              <CardTitle>Articles, filings and announcements</CardTitle>
              <span className="text-[10px] text-muted">{news.data?.total ?? 0} stories (deduplicated)</span>
            </CardHeader>
            {news.data?.items.length ? news.data.items.map((a) => <ArticleCard key={a.id} a={a} />) : <Empty>No stories collected for this pair yet.</Empty>}
            {news.data && news.data.total > news.data.page_size ? (
              <div className="flex justify-end gap-2 border-t border-border p-2">
                <Button variant="secondary" size="sm" disabled={newsPage <= 1} onClick={() => setNewsPage(newsPage - 1)}>
                  Previous
                </Button>
                <Button variant="secondary" size="sm" disabled={newsPage * news.data.page_size >= news.data.total} onClick={() => setNewsPage(newsPage + 1)}>
                  Next
                </Button>
              </div>
            ) : null}
          </Card>
        </TabsContent>

        <TabsContent value="predictions">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
            <Select value={horizon} onChange={(e) => setHorizon(e.target.value)} aria-label="Horizon">
              <option value="">All horizons</option>
              {[30, 90, 180, 252, 504].map((h) => (
                <option key={h} value={h}>
                  {h} trading days
                </option>
              ))}
            </Select>
            <Select value={predState} onChange={(e) => setPredState(e.target.value)} aria-label="State">
              <option value="all">All</option>
              <option value="pending">Pending (not matured)</option>
              <option value="resolved">Resolved</option>
            </Select>
            <span className="text-muted">Predictions are immutable once recorded. Pending predictions are never scored as right or wrong.</span>
          </div>
          <div className="grid gap-3 xl:grid-cols-[1fr_300px]">
            <Card>
              <Table>
                <THead>
                  <TR>
                    <TH>Date</TH>
                    <TH className="text-right">H</TH>
                    <TH>Matures</TH>
                    <TH className="text-right">SI-v1</TH>
                    <TH className="text-right">SI-v2</TH>
                    <TH>Bias</TH>
                    <TH className="text-right">Threshold</TH>
                    <TH className="text-right">Realised D</TH>
                    <TH>Outcome</TH>
                    <TH className="text-right">Error*</TH>
                  </TR>
                </THead>
                <TBody>
                  {(preds.data?.items ?? []).slice(0, 300).map((r) => (
                    <TR key={r.id}>
                      <TD className="num">{r.prediction_date}</TD>
                      <TD className="num text-right">{r.horizon_days}</TD>
                      <TD className="num text-muted">{r.maturity_date}</TD>
                      <TD className="text-right">
                        <ScoreCell value={r.si_v1} />
                      </TD>
                      <TD className="text-right">
                        <ScoreCell value={r.si_v2} status={r.si_v2_status} />
                      </TD>
                      <TD>
                        <DirectionBadge d={r.direction} a={a} b={b} />
                      </TD>
                      <TD className="num text-right">{fmtNum(r.target_definition.threshold, 3)}</TD>
                      <TD className="num text-right">{r.outcome ? fmtNum(r.outcome.divergence, 3) : "—"}</TD>
                      <TD>
                        {r.outcome ? (
                          <Badge variant={r.outcome.diverged ? "risk" : "ok"}>{r.outcome.diverged ? "diverged" : "held together"}</Badge>
                        ) : (
                          <Badge>pending</Badge>
                        )}
                      </TD>
                      <TD className={cn("num text-right", r.error !== null && Math.abs(r.error) > 0.5 && "text-warn")}>{r.error === null ? "—" : fmtNum(r.error, 2)}</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
              {!preds.data?.items.length ? <Empty>No predictions recorded yet. Snapshots are taken daily after the US close.</Empty> : null}
              <div className="border-t border-border px-3 py-1.5 text-[10px] text-muted">
                * Error = SI-v2/100 − outcome (pseudo-probability; the index is not calibrated). Divergence D = |ln(PA(t+H)/PA(t)) − ln(PB(t+H)/PB(t))| on adjusted closes.
              </div>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Calibration (this pair)</CardTitle>
              </CardHeader>
              <CardContent>{preds.data ? <Calibration preds={preds.data.items} /> : <Skeleton className="h-24" />}</CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
      <ScoreInspector slug={slug} scoreId={inspect} onClose={() => setInspect(null)} />
    </div>
  );
}
