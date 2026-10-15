"use client";
import { Download, FileText, Play } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Empty, Notice, SimulatedTag, Stat } from "@/components/domain";
import { useIsAdmin } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { API, apiSend, useApi } from "@/lib/api";
import { fmtDateTime, fmtInt, fmtNum, fmtPct } from "@/lib/format";

interface Confusion { threshold: number; tp: number; fp: number; fn: number; tn: number; precision: number | null; recall: number | null }
interface Metrics {
  n: number;
  positives?: number;
  base_rate?: number;
  status: string;
  roc_auc?: number | null;
  roc_auc_ci?: { low: number | null; high: number | null; n_boot: number; cluster: string } | null;
  pr_auc?: number | null;
  brier?: number | null;
  brier_note?: string;
  calibration?: { bin: string; mean_predicted: number; observed_rate: number; n: number }[];
  quartiles?: { quartile: string; n: number; divergence_rate: number; score_range: [number, number] }[];
  confusion?: Confusion[];
  magnitude?: { spearman_rho: number | null; p_value: number | null };
}
interface HorizonBlock {
  pending: number;
  resolved: number;
  target: Record<string, string | number>;
  indices: Record<string, Metrics>;
  by_sector: Record<string, { n: number; divergence_rate: number; si_v1_auc: number | null; si_v2_auc: number | null }>;
  sensitivity: { multiplier: number; base_rate: number; si_v1_auc: number | null; si_v2_auc: number | null; correlation_only_auc: number | null }[];
  paper_trades: Record<string, { n: number; mean_return: number | null; hit_rate: number | null }>;
}
interface Backtest {
  id: number;
  kind: string;
  label: string;
  created_at: string;
  params: Record<string, unknown>;
  metrics: Record<string, any>;
  limitations: string[];
  n_observations: number;
  is_demo: boolean;
}
interface PerfReport {
  live: { horizons: Record<string, HorizonBlock>; contains_simulated: boolean; note: string };
  backtests: Backtest[];
}

const INDEX_LABEL: Record<string, string> = {
  si_v1: "SI-v1",
  si_v2: "SI-v2",
  correlation_only: "Correlation only",
  previous_return_gap: "Previous return gap",
  si_v2_partial_default: "SI-v2 (F,M,S) default weights",
  si_v2_partial_train_optimised: "SI-v2 (F,M,S) train-optimised",
};
const label = (k: string) => INDEX_LABEL[k] ?? k.replace(/_/g, " ");
const tooltipStyle = { background: "var(--panel)", border: "1px solid var(--border)", fontSize: 11, color: "var(--fg)" };
const COLORS = ["#60a5fa", "#f59e0b", "#a78bfa", "#94a3b8", "#34d399", "#f472b6"];

function MetricsTable({ indices, testId }: { indices: Record<string, Metrics>; testId?: string }) {
  const keys = Object.keys(indices);
  if (!keys.length) return <Empty>No metrics.</Empty>;
  return (
    <Table data-testid={testId}>
      <THead>
        <TR>
          <TH>Index</TH>
          <TH className="text-right">n</TH>
          <TH className="text-right">Base rate</TH>
          <TH className="text-right">ROC-AUC</TH>
          <TH className="text-right">95% CI (industry bootstrap)</TH>
          <TH className="text-right">PR-AUC</TH>
          <TH className="text-right">Brier*</TH>
          <TH className="text-right">Spearman ρ (magnitude)</TH>
        </TR>
      </THead>
      <TBody>
        {keys.map((k) => {
          const m = indices[k];
          return (
            <TR key={k}>
              <TD className="font-medium">{label(k)}</TD>
              <TD className="num text-right">{fmtInt(m.n)}</TD>
              <TD className="num text-right">{fmtPct(m.base_rate, 1)}</TD>
              <TD className="num text-right">{m.status === "ok" ? fmtNum(m.roc_auc, 3) : <span className="text-muted">{m.status.replace(/_/g, " ")}</span>}</TD>
              <TD className="num text-right">{m.roc_auc_ci && m.roc_auc_ci.low !== null ? `${fmtNum(m.roc_auc_ci.low, 2)} – ${fmtNum(m.roc_auc_ci.high, 2)}` : "—"}</TD>
              <TD className="num text-right">{fmtNum(m.pr_auc, 3)}</TD>
              <TD className="num text-right">{fmtNum(m.brier, 3)}</TD>
              <TD className="num text-right">
                {fmtNum(m.magnitude?.spearman_rho, 2)}
                {m.magnitude?.p_value !== undefined && m.magnitude?.p_value !== null ? <span className="text-muted"> (p {fmtNum(m.magnitude.p_value, 3)})</span> : null}
              </TD>
            </TR>
          );
        })}
      </TBody>
    </Table>
  );
}

function CalibrationChart({ indices }: { indices: Record<string, Metrics> }) {
  const bins = new Map<string, Record<string, number | string>>();
  const keys = Object.keys(indices).filter((k) => indices[k].calibration?.length);
  for (const k of keys) {
    for (const c of indices[k].calibration ?? []) {
      const row = bins.get(c.bin) ?? { bin: c.bin, ideal: (Number(c.bin.split("-")[0]) + Number(c.bin.split("-")[1])) / 2 };
      row[k] = c.observed_rate;
      bins.set(c.bin, row);
    }
  }
  const data = Array.from(bins.values()).sort((a, b) => String(a.bin).localeCompare(String(b.bin)));
  if (!data.length) return <Empty>Not enough resolved predictions for calibration.</Empty>;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ left: -15, right: 8, top: 5 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
        <XAxis dataKey="bin" tick={{ fontSize: 10, fill: "var(--muted)" }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "var(--muted)" }} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtPct(Number(v))} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="ideal" name="Perfect calibration" stroke="var(--muted)" strokeDasharray="4 4" dot={false} />
        {keys.map((k, i) => (
          <Line key={k} dataKey={k} name={label(k)} stroke={COLORS[i % COLORS.length]} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function QuartileChart({ indices }: { indices: Record<string, Metrics> }) {
  const keys = Object.keys(indices).filter((k) => indices[k].quartiles?.length);
  const data = ["Q1", "Q2", "Q3", "Q4"].map((q) => {
    const row: Record<string, number | string> = { quartile: q };
    for (const k of keys) {
      const r = indices[k].quartiles?.find((x) => x.quartile === q);
      if (r) row[k] = r.divergence_rate;
    }
    return row;
  });
  if (!keys.length) return <Empty>Not enough resolved predictions.</Empty>;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: -15, right: 8, top: 5 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
        <XAxis dataKey="quartile" tick={{ fontSize: 10, fill: "var(--muted)" }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "var(--muted)" }} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtPct(Number(v))} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {keys.map((k, i) => (
          <Bar key={k} dataKey={k} name={label(k)} fill={COLORS[i % COLORS.length]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function ConfusionTables({ indices }: { indices: Record<string, Metrics> }) {
  const keys = Object.keys(indices).filter((k) => indices[k].confusion?.length);
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {keys.map((k) => (
        <div key={k}>
          <div className="mb-1 text-xs font-medium">{label(k)}</div>
          <Table>
            <THead>
              <TR>
                <TH>Alert at ≥</TH>
                <TH className="text-right">TP</TH>
                <TH className="text-right">FP</TH>
                <TH className="text-right">FN</TH>
                <TH className="text-right">TN</TH>
                <TH className="text-right">Precision</TH>
                <TH className="text-right">Recall</TH>
              </TR>
            </THead>
            <TBody>
              {indices[k].confusion!.map((c) => (
                <TR key={c.threshold}>
                  <TD className="num">{c.threshold}</TD>
                  <TD className="num text-right">{c.tp}</TD>
                  <TD className="num text-right">{c.fp}</TD>
                  <TD className="num text-right">{c.fn}</TD>
                  <TD className="num text-right">{c.tn}</TD>
                  <TD className="num text-right">{fmtPct(c.precision)}</TD>
                  <TD className="num text-right">{fmtPct(c.recall)}</TD>
                </TR>
              ))}
            </TBody>
          </Table>
        </div>
      ))}
    </div>
  );
}

function LiveHorizon({ h, block }: { h: string; block: HorizonBlock }) {
  const v1 = block.indices.si_v1;
  const v2 = block.indices.si_v2;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <Stat label="Resolved" value={fmtInt(block.resolved)} sub={`${fmtInt(block.pending)} still pending (never counted)`} />
        <Stat label="Threshold" value={fmtNum(Number(block.target.threshold), 3)} sub={`H = ${h} trading days`} />
        <Stat label="Divergence base rate" value={fmtPct(v2?.base_rate ?? v1?.base_rate, 1)} />
        <Stat label="SI-v1 ROC-AUC" value={fmtNum(v1?.roc_auc, 3)} />
        <Stat label="SI-v2 ROC-AUC" value={fmtNum(v2?.roc_auc, 3)} tone={v2?.roc_auc !== undefined && v2?.roc_auc !== null && v1?.roc_auc && v2.roc_auc > v1.roc_auc ? "ok" : undefined} />
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Target definition</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-1 text-xs md:grid-cols-2">
          {Object.entries(block.target).map(([k, v]) => (
            <div key={k}>
              <span className="text-muted">{k.replace(/_/g, " ")}:</span> <span className="num">{String(v)}</span>
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Discrimination</CardTitle>
        </CardHeader>
        <CardContent>
          <MetricsTable indices={block.indices} testId={`live-metrics-${h}`} />
          <p className="mt-1 text-[11px] text-muted">*Brier uses score/100 as a pseudo-probability; the indices are relative risk rankings and are not calibrated probabilities.</p>
        </CardContent>
      </Card>
      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Calibration (observed divergence rate by score decile)</CardTitle>
          </CardHeader>
          <CardContent>
            <CalibrationChart indices={block.indices} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Divergence rate by score quartile</CardTitle>
          </CardHeader>
          <CardContent>
            <QuartileChart indices={block.indices} />
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Precision and recall at alert thresholds</CardTitle>
        </CardHeader>
        <CardContent>
          <ConfusionTables indices={block.indices} />
        </CardContent>
      </Card>
      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>By industry</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <THead>
                <TR>
                  <TH>Industry</TH>
                  <TH className="text-right">n</TH>
                  <TH className="text-right">Divergence rate</TH>
                  <TH className="text-right">SI-v1 AUC</TH>
                  <TH className="text-right">SI-v2 AUC</TH>
                </TR>
              </THead>
              <TBody>
                {Object.entries(block.by_sector).map(([k, r]) => (
                  <TR key={k}>
                    <TD>{k}</TD>
                    <TD className="num text-right">{r.n}</TD>
                    <TD className="num text-right">{fmtPct(r.divergence_rate)}</TD>
                    <TD className="num text-right">{fmtNum(r.si_v1_auc, 2)}</TD>
                    <TD className="num text-right">{fmtNum(r.si_v2_auc, 2)}</TD>
                  </TR>
                ))}
              </TBody>
            </Table>
            <p className="mt-1 text-[11px] text-muted">AUC is undefined (—) when every observation in an industry had the same outcome.</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Threshold sensitivity</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <THead>
                <TR>
                  <TH>Threshold ×</TH>
                  <TH className="text-right">Base rate</TH>
                  <TH className="text-right">SI-v1 AUC</TH>
                  <TH className="text-right">SI-v2 AUC</TH>
                  <TH className="text-right">Corr-only AUC</TH>
                </TR>
              </THead>
              <TBody>
                {block.sensitivity.map((r) => (
                  <TR key={r.multiplier}>
                    <TD className="num">{r.multiplier}</TD>
                    <TD className="num text-right">{fmtPct(r.base_rate)}</TD>
                    <TD className="num text-right">{fmtNum(r.si_v1_auc, 3)}</TD>
                    <TD className="num text-right">{fmtNum(r.si_v2_auc, 3)}</TD>
                    <TD className="num text-right">{fmtNum(r.correlation_only_auc, 3)}</TD>
                  </TR>
                ))}
              </TBody>
            </Table>
            <div className="mt-3 text-xs">
              <div className="mb-1 font-medium">Paper strategies (research only, no orders are placed)</div>
              {Object.entries(block.paper_trades).map(([k, r]) => (
                <div key={k} className="flex justify-between border-b border-border py-1 last:border-0">
                  <span className="capitalize">{k}</span>
                  <span className="num">
                    n {r.n} · mean {fmtPct(r.mean_return, 2)} · hit rate {fmtPct(r.hit_rate)}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function BacktestCard({ b }: { b: Backtest }) {
  const m = b.metrics;
  if (b.kind === "reported_historical") {
    return (
      <Card data-testid="reported-benchmark">
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>{b.label}</CardTitle>
          <Badge variant="default">Reported, not reproduced</Badge>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div className="grid gap-1 md:grid-cols-3">
            {Object.entries(b.params).map(([k, v]) => (
              <div key={k}>
                <span className="text-muted">{k}:</span> {String(v)}
              </div>
            ))}
          </div>
          <Table>
            <THead>
              <TR>
                <TH>Index</TH>
                <TH className="text-right">ROC-AUC</TH>
                <TH className="text-right">Top-quartile divergence</TH>
                <TH className="text-right">Bottom-quartile divergence</TH>
              </TR>
            </THead>
            <TBody>
              {Object.entries(m).map(([k, r]: [string, any]) => (
                <TR key={k}>
                  <TD>{label(k)}</TD>
                  <TD className="num text-right">{fmtNum(r.roc_auc, 4)}</TD>
                  <TD className="num text-right">{fmtPct(r.top_quartile_divergence_rate, 1)}</TD>
                  <TD className="num text-right">{fmtPct(r.bottom_quartile_divergence_rate, 1)}</TD>
                </TR>
              ))}
            </TBody>
          </Table>
          <ul className="list-disc pl-4 text-muted">
            {b.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card data-testid="backtest-run">
      <CardHeader className="flex-row items-center justify-between gap-2">
        <div>
          <CardTitle>{b.label}</CardTitle>
          <div className="text-[11px] text-muted">
            Run {fmtDateTime(b.created_at)} · {b.n_observations} observations ({m.n_train} train, {m.n_test} test)
          </div>
        </div>
        <div className="flex items-center gap-2">
          <SimulatedTag show={b.is_demo} />
          <Button asChild size="sm" variant="outline">
            <a href={`${API}/research/export/backtest.csv?backtest_id=${b.id}`}>
              <Download /> CSV
            </a>
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-1 text-xs md:grid-cols-4">
          {["train", "embargo", "test"].map((k) => (
            <div key={k}>
              <span className="text-muted capitalize">{k}:</span> {Array.isArray(b.params[k]) ? (b.params[k] as unknown[]).join(" – ") : String(b.params[k] ?? "—")}
            </div>
          ))}
          <div>
            <span className="text-muted">Target:</span> D ≥ {String(b.params.threshold)} over {String(b.params.horizon)}d
          </div>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium">Out-of-sample test period</div>
          <MetricsTable indices={m.test ?? {}} />
        </div>
        {m.optimised_weights ? (
          <div className="text-xs">
            <span className="text-muted">Weights fitted on {m.optimised_weights.fitted_on}:</span>{" "}
            <span className="num">
              {Object.entries(m.optimised_weights.weights as Record<string, number>)
                .map(([k, v]) => `${k} ${v.toFixed(2)}`)
                .join(" · ")}
            </span>{" "}
            <span className="text-muted">(train AUC {fmtNum(m.optimised_weights.train_auc, 3)})</span>
          </div>
        ) : null}
        <div className="grid gap-3 lg:grid-cols-2">
          <div>
            <div className="mb-1 text-xs font-medium">Ablations (test)</div>
            <MetricsTable indices={m.ablations_test ?? {}} />
          </div>
          <div>
            <div className="mb-1 text-xs font-medium">Alternative weights (test)</div>
            <MetricsTable indices={m.alternative_weights_test ?? {}} />
          </div>
        </div>
        <details className="text-xs">
          <summary className="cursor-pointer text-muted">Training-period metrics (in-sample, for reference only)</summary>
          <div className="mt-2">
            <MetricsTable indices={m.train ?? {}} />
          </div>
        </details>
        <ul className="list-disc pl-4 text-xs text-muted">
          {b.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function StartBacktest({ onStarted }: { onStarted: () => void }) {
  const [open, setOpen] = useState(false);
  const [horizon, setHorizon] = useState("504");
  const [threshold, setThreshold] = useState("0.20");
  const [frequency, setFrequency] = useState("annual");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  async function submit() {
    setBusy(true);
    setErr(null);
    try {
      await apiSend("/research/backtests", "POST", { horizon: Number(horizon), threshold: Number(threshold), frequency });
      setOpen(false);
      onStarted();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" variant="secondary">
          <Play /> New walk-forward backtest
        </Button>
      </DialogTrigger>
      <DialogContent title="Start a walk-forward backtest" description="Train 2015–2019, embargo 2020–2021, test 2022–2024. Runs in the worker; results appear here when finished.">
        <div className="grid grid-cols-3 gap-2">
          <div>
            <Label>Horizon (trading days)</Label>
            <Select className="w-full" value={horizon} onChange={(e) => setHorizon(e.target.value)}>
              {[30, 90, 180, 252, 504].map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </Select>
          </div>
          <div>
            <Label>Threshold</Label>
            <Input type="number" step="0.01" min="0.01" max="2" value={threshold} onChange={(e) => setThreshold(e.target.value)} />
          </div>
          <div>
            <Label>Frequency</Label>
            <Select className="w-full" value={frequency} onChange={(e) => setFrequency(e.target.value)}>
              <option value="annual">Annual</option>
              <option value="quarterly">Quarterly</option>
              <option value="monthly">Monthly</option>
            </Select>
          </div>
        </div>
        {err ? <p className="mt-2 text-xs text-risk">{err}</p> : null}
        <div className="mt-3 flex justify-end">
          <Button onClick={submit} disabled={busy}>
            {busy ? "Queuing…" : "Queue backtest"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function PerformancePage() {
  const admin = useIsAdmin();
  const perf = useApi<PerfReport>("/performance?n_boot=300");
  const [queued, setQueued] = useState(false);
  const horizons = perf.data ? Object.keys(perf.data.live.horizons).sort((a, b) => Number(a) - Number(b)) : [];
  const defaultH = horizons.includes("504") ? "504" : horizons[horizons.length - 1];
  const walk = perf.data?.backtests.filter((b) => b.kind !== "reported_historical") ?? [];
  const reported = perf.data?.backtests.find((b) => b.kind === "reported_historical");
  const latestWalk = walk[0];
  const live504 = perf.data?.live.horizons["504"];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold">Model performance</h1>
          <p className="text-xs text-muted">Live forward-tested predictions, walk-forward backtests and the original reported experiment are kept separate.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button asChild size="sm" variant="outline">
            <a href={`${API}/research/export/predictions.csv`}>
              <Download /> Predictions CSV
            </a>
          </Button>
          <Button asChild size="sm" variant="outline">
            <a href={`${API}/research/export/scores.csv`}>
              <Download /> Scores CSV
            </a>
          </Button>
          <Button asChild size="sm" variant="outline">
            <a href={`${API}/research/report.pdf`}>
              <FileText /> PDF report
            </a>
          </Button>
          {admin ? <StartBacktest onStarted={() => setQueued(true)} /> : null}
        </div>
      </div>
      {queued ? <Notice kind="info">Backtest queued. Follow it on the Bot activity page; refresh here once it finishes.</Notice> : null}
      {perf.error ? <Notice kind="risk">Could not load performance: {String(perf.error.message)}</Notice> : null}
      {!perf.data && !perf.error ? <Skeleton className="h-64 w-full" /> : null}
      {perf.data?.live.contains_simulated ? (
        <Notice>These metrics include SIMULATED demo predictions. They demonstrate the pipeline and say nothing about real predictive power.</Notice>
      ) : null}

      {perf.data ? (
        <Card data-testid="comparison">
          <CardHeader>
            <CardTitle>Live vs backtest vs reported (two-year horizon, ROC-AUC)</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <THead>
                <TR>
                  <TH>Source</TH>
                  <TH className="text-right">n</TH>
                  <TH className="text-right">SI-v1</TH>
                  <TH className="text-right">SI-v2</TH>
                  <TH className="text-right">Correlation only</TH>
                  <TH className="text-right">Previous return gap</TH>
                </TR>
              </THead>
              <TBody>
                <TR>
                  <TD>Live forward test {perf.data.live.contains_simulated ? <SimulatedTag /> : null}</TD>
                  <TD className="num text-right">{fmtInt(live504?.resolved)}</TD>
                  <TD className="num text-right">{fmtNum(live504?.indices.si_v1?.roc_auc, 3)}</TD>
                  <TD className="num text-right">{fmtNum(live504?.indices.si_v2?.roc_auc, 3)}</TD>
                  <TD className="num text-right">{fmtNum(live504?.indices.correlation_only?.roc_auc, 3)}</TD>
                  <TD className="num text-right">{fmtNum(live504?.indices.previous_return_gap?.roc_auc, 3)}</TD>
                </TR>
                {latestWalk ? (
                  <TR>
                    <TD>Walk-forward test 2022–2024 {latestWalk.is_demo ? <SimulatedTag /> : null}</TD>
                    <TD className="num text-right">{fmtInt(latestWalk.metrics.n_test)}</TD>
                    <TD className="num text-right">{fmtNum(latestWalk.metrics.test?.si_v1?.roc_auc, 3)}</TD>
                    <TD className="num text-right">
                      {fmtNum(latestWalk.metrics.test?.si_v2_partial_default?.roc_auc, 3)} <span className="text-[10px] text-muted">(F,M,S only)</span>
                    </TD>
                    <TD className="num text-right">{fmtNum(latestWalk.metrics.test?.correlation_only?.roc_auc, 3)}</TD>
                    <TD className="num text-right">{fmtNum(latestWalk.metrics.test?.previous_return_gap?.roc_auc, 3)}</TD>
                  </TR>
                ) : null}
                {reported ? (
                  <TR>
                    <TD>Original experiment (reported)</TD>
                    <TD className="num text-right">{String(reported.params.observations ?? "—")}</TD>
                    <TD className="num text-right">{fmtNum(reported.metrics.si_v1?.roc_auc, 3)}</TD>
                    <TD className="num text-right text-muted">not tested</TD>
                    <TD className="num text-right">{fmtNum(reported.metrics.correlation_only?.roc_auc, 3)}</TD>
                    <TD className="num text-right">{fmtNum(reported.metrics.previous_return_gap?.roc_auc, 3)}</TD>
                  </TR>
                ) : null}
              </TBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}

      {perf.data ? (
        <Tabs defaultValue="live">
          <TabsList>
            <TabsTrigger value="live">Live forward test</TabsTrigger>
            <TabsTrigger value="backtests">Backtests ({walk.length})</TabsTrigger>
            <TabsTrigger value="reported">Reported benchmark</TabsTrigger>
          </TabsList>
          <TabsContent value="live">
            <p className="mb-2 text-xs text-muted">{perf.data.live.note}</p>
            {horizons.length ? (
              <Tabs defaultValue={defaultH}>
                <TabsList>
                  {horizons.map((h) => (
                    <TabsTrigger key={h} value={h}>
                      {h}d
                    </TabsTrigger>
                  ))}
                </TabsList>
                {horizons.map((h) => (
                  <TabsContent key={h} value={h}>
                    <LiveHorizon h={h} block={perf.data!.live.horizons[h]} />
                  </TabsContent>
                ))}
              </Tabs>
            ) : (
              <Empty>No prediction snapshots yet. The first ones are written on the next trading-day snapshot run.</Empty>
            )}
          </TabsContent>
          <TabsContent value="backtests" className="space-y-3">
            {walk.length ? walk.map((b) => <BacktestCard key={b.id} b={b} />) : <Empty>No walk-forward backtests yet.</Empty>}
          </TabsContent>
          <TabsContent value="reported">{reported ? <BacktestCard b={reported} /> : <Empty>No reported benchmark stored.</Empty>}</TabsContent>
        </Tabs>
      ) : null}
    </div>
  );
}
