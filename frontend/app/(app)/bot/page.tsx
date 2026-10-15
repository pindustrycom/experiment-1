"use client";
import { Pause, Play, RefreshCw, Save } from "lucide-react";
import { Fragment, useState } from "react";
import { Notice, Stat, StatusBadge, StatusDot } from "@/components/domain";
import { useIsAdmin } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { apiSend, useApi } from "@/lib/api";
import { ago, fmtDateTime, fmtInZone, fmtInt, fmtUsd } from "@/lib/format";
import type { JobRun, Page, ScheduledJob, SystemStatus } from "@/lib/types";

interface ProviderUsageRow {
  provider: string;
  requests: number;
  successes: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  last_call: string | null;
  last_success: string | null;
  avg_latency_ms: number | null;
}
interface Usage {
  providers: ProviderUsageRow[];
  ai_daily: { date: string; calls: number; input_tokens: number; output_tokens: number; cost_usd: number }[];
  recent_errors: { provider: string; operation: string; error: string | null; status_code: number | null; ts: string }[];
}

function duration(ms: number | null) {
  if (ms === null) return "—";
  if (ms < 1000) return `${ms} ms`;
  if (ms < 120000) return `${(ms / 1000).toFixed(1)} s`;
  return `${(ms / 60000).toFixed(1)} min`;
}

function summarizeResult(r: Record<string, unknown>): string {
  const parts: string[] = [];
  const walk = (o: Record<string, unknown>, prefix = "") => {
    for (const [k, v] of Object.entries(o)) {
      if (typeof v === "number") parts.push(`${prefix}${k.replace(/_/g, " ")} ${v}`);
      else if (v && typeof v === "object" && !Array.isArray(v) && !prefix) walk(v as Record<string, unknown>, `${k}: `);
    }
  };
  walk(r);
  return parts.slice(0, 6).join(" · ");
}

function ScheduleEditor({ job, onClose, onSaved }: { job: ScheduledJob; onClose: () => void; onSaved: () => void }) {
  const [cron, setCron] = useState(job.cron);
  const [err, setErr] = useState<string | null>(null);
  async function save() {
    setErr(null);
    try {
      await apiSend(`/jobs/schedules/${job.name}`, "PATCH", { cron });
      onSaved();
      onClose();
    } catch (e) {
      setErr((e as Error).message);
    }
  }
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent title={`Schedule: ${job.name}`} description={`Cron expression in ${job.timezone} (minute hour day-of-month month day-of-week). DST is handled automatically.`}>
        <Label>Cron</Label>
        <Input value={cron} onChange={(e) => setCron(e.target.value)} className="font-mono" aria-label="Cron expression" />
        <p className="mt-1 text-[11px] text-muted">
          Examples: <code>0 * * * *</code> hourly · <code>*/30 9-16 * * 1-5</code> every 30 min in market hours · <code>30 17 * * 1-5</code> weekdays 17:30.
          {job.trading_days_only ? " This job also skips NYSE holidays." : ""}
        </p>
        {err ? <p className="mt-2 text-xs text-risk">{err}</p> : null}
        <div className="mt-3 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={save}>
            <Save /> Save
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default function BotPage() {
  const admin = useIsAdmin();
  const [jobFilter, setJobFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [page, setPage] = useState(1);
  const [editing, setEditing] = useState<ScheduledJob | null>(null);
  const [msg, setMsg] = useState<{ kind: "info" | "risk"; text: string } | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);

  const sys = useApi<SystemStatus>("/system/status", { refreshInterval: 10000 });
  const jobs = useApi<ScheduledJob[]>("/jobs", { refreshInterval: 15000 });
  const qs = new URLSearchParams({ page: String(page), page_size: "30" });
  if (jobFilter) qs.set("job", jobFilter);
  if (statusFilter) qs.set("status", statusFilter);
  const runs = useApi<Page<JobRun>>(`/jobs/runs?${qs}`, { refreshInterval: 10000 });
  const usage = useApi<Usage>("/usage", { refreshInterval: 60000 });

  const refreshAll = () => {
    sys.mutate();
    jobs.mutate();
    runs.mutate();
  };

  async function act(fn: () => Promise<unknown>, ok: string) {
    setMsg(null);
    try {
      await fn();
      setMsg({ kind: "info", text: ok });
      refreshAll();
    } catch (e) {
      setMsg({ kind: "risk", text: (e as Error).message });
    }
  }

  const s = sys.data;
  const pages = runs.data ? Math.max(1, Math.ceil(runs.data.total / runs.data.page_size)) : 1;
  const failures = (runs.data?.items ?? []).filter((r) => r.status === "failed");

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold">Bot activity</h1>
          <p className="text-xs text-muted">Scheduled jobs run in the worker. Times are America/New_York; trading-day jobs skip weekends and NYSE holidays.</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={refreshAll}>
            <RefreshCw /> Refresh
          </Button>
          {admin && s ? (
            s.bot.paused ? (
              <Button size="sm" onClick={() => act(() => apiSend("/jobs/resume", "POST"), "Scheduler resumed.")} data-testid="resume">
                <Play /> Resume
              </Button>
            ) : (
              <Button size="sm" variant="secondary" onClick={() => act(() => apiSend("/jobs/pause", "POST"), "Scheduler paused. Manual runs still work.")} data-testid="pause">
                <Pause /> Pause
              </Button>
            )
          ) : null}
        </div>
      </div>
      {msg ? <Notice kind={msg.kind}>{msg.text}</Notice> : null}
      {s?.bot.state === "unknown" ? (
        <Notice>No worker activity in the last two hours. If the worker container is down, queued jobs will run once it is back.</Notice>
      ) : null}

      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6">
        {s ? (
          <>
            <Stat
              label="Scheduler"
              value={
                <span className="flex items-center gap-1.5 capitalize">
                  <StatusDot state={s.bot.paused ? "paused" : s.bot.state} />
                  {s.bot.paused ? "paused" : s.bot.state}
                </span>
              }
              sub={`last activity ${ago(s.bot.last_activity)}`}
              tone={s.bot.paused ? "warn" : undefined}
            />
            <Stat label="Running / queued" value={`${s.bot.running_jobs.length} / ${s.bot.queued}`} sub={s.bot.running_jobs.join(", ") || "idle"} />
            <Stat label="Claude model" value={<span className="text-sm">{s.ai.model}</span>} sub={`effort ${s.ai.effort}${s.ai.configured ? "" : " · API key not set"}`} tone={s.ai.configured ? undefined : "warn"} />
            <Stat label="AI calls this month" value={fmtInt(s.ai.calls)} sub={`${fmtInt(s.ai.input_tokens)} in · ${fmtInt(s.ai.output_tokens)} out tokens`} />
            <Stat
              label="AI cost this month"
              value={fmtUsd(s.ai.month_cost_usd)}
              sub={`limit ${fmtUsd(s.ai.monthly_limit_usd)}`}
              tone={s.ai.month_cost_usd >= s.ai.monthly_limit_usd ? "risk" : s.ai.month_cost_usd >= 0.8 * s.ai.monthly_limit_usd ? "warn" : undefined}
            />
            <Stat label="Awaiting analysis" value={fmtInt(s.pending_analysis)} sub={`news ${ago(s.last_news_update)} · prices ${ago(s.last_price_update)}`} />
          </>
        ) : (
          Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-16" />)
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Schedules</CardTitle>
        </CardHeader>
        <CardContent>
          <Table data-testid="schedules">
            <THead>
              <TR>
                <TH>Job</TH>
                <TH>Schedule</TH>
                <TH>Next run</TH>
                <TH>Last run</TH>
                <TH>Enabled</TH>
                {admin ? <TH className="text-right">Actions</TH> : null}
              </TR>
            </THead>
            <TBody>
              {(jobs.data ?? []).map((j) => (
                <TR key={j.name}>
                  <TD>
                    <div className="font-medium">{j.name}</div>
                    <div className="text-[11px] text-muted">{j.description}</div>
                  </TD>
                  <TD>
                    {j.cron ? <code className="text-xs">{j.cron}</code> : <span className="text-xs text-muted">manual only</span>}
                    {j.trading_days_only ? <Badge variant="default" className="ml-1">trading days</Badge> : null}
                  </TD>
                  <TD className="num text-xs">{j.enabled && !s?.bot.paused ? fmtInZone(j.next_run, j.timezone) : "—"}</TD>
                  <TD className="text-xs">
                    {j.last_run ? (
                      <span className="flex items-center gap-1.5">
                        <StatusBadge status={j.last_run.status} /> {ago(j.last_run.finished_at ?? j.last_run.created_at)}
                      </span>
                    ) : (
                      <span className="text-muted">never</span>
                    )}
                  </TD>
                  <TD>
                    <Switch
                      checked={j.enabled}
                      disabled={!admin || !j.cron}
                      aria-label={`Enable ${j.name}`}
                      onCheckedChange={(v) => act(() => apiSend(`/jobs/schedules/${j.name}`, "PATCH", { enabled: v }), `${j.name} ${v ? "enabled" : "disabled"}.`)}
                    />
                  </TD>
                  {admin ? (
                    <TD className="text-right">
                      <div className="flex justify-end gap-1">
                        {j.cron ? (
                          <Button size="sm" variant="ghost" onClick={() => setEditing(j)}>
                            Edit
                          </Button>
                        ) : null}
                        <Button
                          size="sm"
                          variant="outline"
                          data-testid={`run-${j.name}`}
                          onClick={() => act(() => apiSend("/jobs/run", "POST", { job: j.name }), `${j.name} queued.`)}
                        >
                          <Play /> Run now
                        </Button>
                      </div>
                    </TD>
                  ) : null}
                </TR>
              ))}
            </TBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle>Run history</CardTitle>
          <div className="flex gap-2">
            <Select value={jobFilter} onChange={(e) => (setJobFilter(e.target.value), setPage(1))} aria-label="Filter by job" className="w-44">
              <option value="">All jobs</option>
              {(jobs.data ?? []).map((j) => (
                <option key={j.name} value={j.name}>{j.name}</option>
              ))}
            </Select>
            <Select value={statusFilter} onChange={(e) => (setStatusFilter(e.target.value), setPage(1))} aria-label="Filter by status" className="w-36">
              <option value="">Any status</option>
              {["queued", "running", "success", "failed", "skipped", "interrupted"].map((x) => (
                <option key={x} value={x}>{x}</option>
              ))}
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <Table data-testid="runs">
            <THead>
              <TR>
                <TH>Job</TH>
                <TH>Trigger</TH>
                <TH>Status</TH>
                <TH>Started</TH>
                <TH className="text-right">Duration</TH>
                <TH className="text-right">Attempts</TH>
                <TH>Result</TH>
              </TR>
            </THead>
            <TBody>
              {(runs.data?.items ?? []).map((r) => (
                <Fragment key={r.id}>
                  <TR className="cursor-pointer" onClick={() => setExpanded(expanded === r.id ? null : r.id)}>
                    <TD className="font-medium">{r.job_name}</TD>
                    <TD className="text-xs text-muted">{r.trigger}</TD>
                    <TD>
                      <StatusBadge status={r.status} />
                    </TD>
                    <TD className="num text-xs">{fmtDateTime(r.started_at ?? r.created_at)}</TD>
                    <TD className="num text-right text-xs">{duration(r.duration_ms)}</TD>
                    <TD className="num text-right text-xs">{r.attempts}</TD>
                    <TD className="max-w-md truncate text-xs">{r.error ? <span className="text-risk">{r.error}</span> : summarizeResult(r.result)}</TD>
                  </TR>
                  {expanded === r.id ? (
                    <TR>
                      <TD colSpan={7}>
                        <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded bg-panel-2 p-2 text-[11px]">
                          {JSON.stringify({ params: r.params, result: r.result, error: r.error }, null, 2)}
                        </pre>
                      </TD>
                    </TR>
                  ) : null}
                </Fragment>
              ))}
            </TBody>
          </Table>
          {runs.data && runs.data.items.length === 0 ? <p className="py-4 text-center text-xs text-muted">No runs yet.</p> : null}
          {runs.data && runs.data.total > runs.data.page_size ? (
            <div className="mt-2 flex items-center justify-end gap-2 text-xs">
              <span className="text-muted">
                Page {page} of {pages}
              </span>
              <Button size="sm" variant="outline" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                Previous
              </Button>
              <Button size="sm" variant="outline" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
                Next
              </Button>
            </div>
          ) : null}
          {failures.length ? <p className="mt-2 text-xs text-muted">Failed runs are retried with exponential backoff up to each job's attempt limit; the last error is shown.</p> : null}
        </CardContent>
      </Card>

      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Provider health (last 30 days)</CardTitle>
          </CardHeader>
          <CardContent>
            <Table data-testid="providers">
              <THead>
                <TR>
                  <TH>Provider</TH>
                  <TH>Configured</TH>
                  <TH className="text-right">Requests</TH>
                  <TH className="text-right">Success</TH>
                  <TH className="text-right">Latency</TH>
                  <TH>Last success</TH>
                </TR>
              </THead>
              <TBody>
                {Object.entries(s?.providers ?? {}).map(([name, configured]) => {
                  const u = usage.data?.providers.find((p) => p.provider === name);
                  return (
                    <TR key={name}>
                      <TD className="font-medium">{name}</TD>
                      <TD>{configured ? <Badge variant="default">yes</Badge> : <span className="text-xs text-muted">no</span>}</TD>
                      <TD className="num text-right">{fmtInt(u?.requests ?? 0)}</TD>
                      <TD className="num text-right">{u && u.requests ? `${Math.round((100 * u.successes) / u.requests)}%` : "—"}</TD>
                      <TD className="num text-right">{u?.avg_latency_ms !== null && u?.avg_latency_ms !== undefined ? `${u.avg_latency_ms} ms` : "—"}</TD>
                      <TD className="text-xs">{u ? ago(u.last_success) : "—"}</TD>
                    </TR>
                  );
                })}
              </TBody>
            </Table>
            <p className="mt-1 text-[11px] text-muted">Provider keys are server-side environment variables. Use Settings → Health check for a live test.</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Recent provider errors</CardTitle>
          </CardHeader>
          <CardContent>
            {usage.data?.recent_errors.length ? (
              <ul className="space-y-1.5 text-xs">
                {usage.data.recent_errors.map((e, i) => (
                  <li key={i} className="border-b border-border pb-1.5 last:border-0">
                    <div className="flex justify-between">
                      <span className="font-medium">
                        {e.provider} · {e.operation}
                      </span>
                      <span className="text-muted">{fmtDateTime(e.ts)}</span>
                    </div>
                    <div className="break-words text-risk">{e.status_code ? `HTTP ${e.status_code} ` : ""}{e.error ?? "error"}</div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-muted">No provider errors recorded.</p>
            )}
            {usage.data?.ai_daily.length ? (
              <div className="mt-3">
                <div className="mb-1 text-xs font-medium">Claude usage by day</div>
                <Table>
                  <THead>
                    <TR>
                      <TH>Day</TH>
                      <TH className="text-right">Calls</TH>
                      <TH className="text-right">Tokens in / out</TH>
                      <TH className="text-right">Cost</TH>
                    </TR>
                  </THead>
                  <TBody>
                    {[...usage.data.ai_daily].reverse().slice(0, 14).map((d) => (
                      <TR key={d.date}>
                        <TD className="num">{d.date}</TD>
                        <TD className="num text-right">{d.calls}</TD>
                        <TD className="num text-right">
                          {fmtInt(d.input_tokens)} / {fmtInt(d.output_tokens)}
                        </TD>
                        <TD className="num text-right">{fmtUsd(d.cost_usd)}</TD>
                      </TR>
                    ))}
                  </TBody>
                </Table>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
      {editing ? <ScheduleEditor job={editing} onClose={() => setEditing(null)} onSaved={refreshAll} /> : null}
    </div>
  );
}
