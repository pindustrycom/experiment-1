"use client";
import { Activity, Plus, Save, Trash2 } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { AddPairDialog } from "@/components/add-pair-dialog";
import { Notice, StatusBadge } from "@/components/domain";
import { useIsAdmin, useUser } from "@/components/shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Input, Label, Select, Textarea } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiSend, useApi } from "@/lib/api";
import { COMPONENT_NAMES, fmtDateTime } from "@/lib/format";
import type { Page, Pair, User, Watchlist } from "@/lib/types";

type Sections = {
  bot: { paused: boolean };
  scoring: { si_v2_weights: Record<string, number>; min_coverage: number };
  llm: { model: string | null; effort: string | null; max_documents_per_run: number; refusal_fallback: boolean; analyze_routine_filings: boolean; pricing: Record<string, Record<string, number>> };
  budget: { monthly_ai_limit_usd: number; warn_at_fraction: number };
  news: { providers: Record<string, boolean>; lookback_hours: number; rss_templates: string[]; company_feeds: Record<string, string[]> };
  prices: { providers: string[]; history_start: string; benchmark: string };
  alerts: { default_threshold: number; default_change_threshold: number; confidence_floor: number };
  predictions: { horizons: number[]; base_threshold: number; base_horizon: number; threshold_scaling: string; threshold_overrides: Record<string, number>; sensitivity_multipliers: number[] };
};
interface SettingsResponse {
  sections: Sections;
  environment: { mode: string; claude_model_default: string; claude_effort_default: string; providers_configured: Record<string, boolean>; scheduler_timezone: string; note: string };
}
interface ModelVersion { id: number; index_name: string; version: string; weights: Record<string, number> | null; is_active: boolean; created_at: string; description: string | null }

const CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"];

function useSaver() {
  const [msg, setMsg] = useState<{ kind: "info" | "risk"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  async function save(section: keyof Sections, value: unknown, after?: () => void) {
    setBusy(true);
    setMsg(null);
    try {
      await apiSend("/settings", "PUT", { section, value });
      setMsg({ kind: "info", text: "Saved." });
      after?.();
    } catch (e) {
      setMsg({ kind: "risk", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }
  return { msg, busy, save };
}

function SaveBar({ saver, onSave, disabled }: { saver: ReturnType<typeof useSaver>; onSave: () => void; disabled?: boolean }) {
  return (
    <div className="mt-3 flex items-center justify-end gap-3">
      {saver.msg ? <span className={saver.msg.kind === "risk" ? "text-xs text-risk" : "text-xs text-ok"}>{saver.msg.text}</span> : null}
      <Button size="sm" onClick={onSave} disabled={saver.busy || disabled}>
        <Save /> {saver.busy ? "Saving…" : "Save"}
      </Button>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <Label>{label}</Label>
      {children}
      {hint ? <p className="mt-0.5 text-[11px] text-muted">{hint}</p> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ pairs & watchlists */
function PairRow({ p, admin, watchlists, onChange }: { p: Pair; admin: boolean; watchlists: Watchlist[]; onChange: () => void }) {
  const [threshold, setThreshold] = useState(String(p.alert_threshold));
  const [change, setChange] = useState(String(p.alert_change_threshold));
  const [every, setEvery] = useState(String(p.monitoring_interval_minutes));
  const [industry, setIndustry] = useState(p.industry ?? "");
  const [err, setErr] = useState<string | null>(null);
  const dirty = threshold !== String(p.alert_threshold) || change !== String(p.alert_change_threshold) || every !== String(p.monitoring_interval_minutes) || industry !== (p.industry ?? "");
  async function patch(body: Record<string, unknown>) {
    setErr(null);
    try {
      await apiSend(`/pairs/${p.slug}`, "PATCH", body);
      onChange();
    } catch (e) {
      setErr((e as Error).message);
    }
  }
  return (
    <TR>
      <TD>
        <Link href={`/pairs/${p.slug}`} className="font-medium hover:underline">
          {p.slug}
        </Link>
        <div className="text-[11px] text-muted">
          {p.company_a.name} / {p.company_b.name}
        </div>
        {err ? <div className="text-[11px] text-risk">{err}</div> : null}
      </TD>
      <TD>
        <Input value={industry} disabled={!admin} onChange={(e) => setIndustry(e.target.value)} className="h-7 w-44" aria-label={`${p.slug} industry`} />
      </TD>
      <TD>
        <StatusBadge status={p.status} />
      </TD>
      <TD>
        <Input type="number" min={15} value={every} disabled={!admin} onChange={(e) => setEvery(e.target.value)} className="h-7 w-20" aria-label={`${p.slug} interval`} />
      </TD>
      <TD>
        <Input type="number" min={0} max={100} value={threshold} disabled={!admin} onChange={(e) => setThreshold(e.target.value)} className="h-7 w-16" aria-label={`${p.slug} alert threshold`} />
      </TD>
      <TD>
        <Input type="number" min={0} max={100} value={change} disabled={!admin} onChange={(e) => setChange(e.target.value)} className="h-7 w-16" aria-label={`${p.slug} change threshold`} />
      </TD>
      <TD className="text-xs text-muted">{watchlists.filter((w) => p.watchlist_ids.includes(w.id)).map((w) => w.name).join(", ") || "—"}</TD>
      {admin ? (
        <TD className="text-right">
          <div className="flex justify-end gap-1">
            {dirty ? (
              <Button
                size="sm"
                onClick={() =>
                  patch({
                    industry: industry || null,
                    monitoring_interval_minutes: Number(every),
                    alert_threshold: Number(threshold),
                    alert_change_threshold: Number(change),
                  })
                }
              >
                Save
              </Button>
            ) : null}
            {p.status === "active" ? (
              <Button size="sm" variant="ghost" onClick={() => patch({ status: "disabled" })}>
                Disable
              </Button>
            ) : (
              <Button size="sm" variant="ghost" onClick={() => patch({ status: "active" })}>
                {p.status === "archived" ? "Restore" : "Enable"}
              </Button>
            )}
            {p.status !== "archived" ? (
              <Button size="sm" variant="ghost" onClick={() => patch({ status: "archived" })}>
                Archive
              </Button>
            ) : null}
          </div>
        </TD>
      ) : null}
    </TR>
  );
}

function WatchlistEditor({ wl, pairs, onClose, onSaved }: { wl: Watchlist | null; pairs: Pair[]; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(wl?.name ?? "");
  const [description, setDescription] = useState(wl?.description ?? "");
  const [ids, setIds] = useState<number[]>(wl?.pair_ids ?? []);
  const [err, setErr] = useState<string | null>(null);
  async function save() {
    setErr(null);
    try {
      const body = { name, description: description || null, pair_ids: ids };
      if (wl) await apiSend(`/watchlists/${wl.id}`, "PUT", body);
      else await apiSend("/watchlists", "POST", body);
      onSaved();
      onClose();
    } catch (e) {
      setErr((e as Error).message);
    }
  }
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent title={wl ? `Edit watchlist: ${wl.name}` : "New watchlist"}>
        <div className="space-y-2">
          <Field label="Name">
            <Input value={name} onChange={(e) => setName(e.target.value)} maxLength={100} aria-label="Watchlist name" />
          </Field>
          <Field label="Description">
            <Input value={description} onChange={(e) => setDescription(e.target.value)} maxLength={2000} />
          </Field>
          <Label>Pairs</Label>
          <div className="grid max-h-64 grid-cols-2 gap-1 overflow-auto rounded border border-border p-2 text-xs">
            {pairs.map((p) => (
              <label key={p.id} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={ids.includes(p.id)}
                  onChange={(e) => setIds((cur) => (e.target.checked ? [...cur, p.id] : cur.filter((x) => x !== p.id)))}
                />
                {p.slug} {p.status !== "active" ? <span className="text-muted">({p.status})</span> : null}
              </label>
            ))}
          </div>
        </div>
        {err ? <p className="mt-2 text-xs text-risk">{err}</p> : null}
        <div className="mt-3 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!name.trim()}>
            <Save /> Save
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function PairsTab({ admin }: { admin: boolean }) {
  const [status, setStatus] = useState("all");
  const pairs = useApi<Page<Pair>>(`/pairs?status=${status}&page_size=500`);
  const all = useApi<Page<Pair>>("/pairs?status=all&page_size=500");
  const wls = useApi<Watchlist[]>("/watchlists");
  const [editing, setEditing] = useState<Watchlist | null | "new">(null);
  const [err, setErr] = useState<string | null>(null);
  const refresh = () => {
    pairs.mutate();
    all.mutate();
    wls.mutate();
  };
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <CardTitle>Pairs</CardTitle>
          <div className="flex gap-2">
            <Select value={status} onChange={(e) => setStatus(e.target.value)} className="w-36" aria-label="Pair status filter">
              <option value="all">All</option>
              <option value="active">Active</option>
              <option value="disabled">Disabled</option>
              <option value="archived">Archived</option>
            </Select>
            {admin ? <AddPairDialog watchlists={wls.data ?? []} onCreated={refresh} /> : null}
          </div>
        </CardHeader>
        <CardContent>
          <Table data-testid="settings-pairs">
            <THead>
              <TR>
                <TH>Pair</TH>
                <TH>Industry</TH>
                <TH>Status</TH>
                <TH>Every (min)</TH>
                <TH>Alert ≥</TH>
                <TH>Δ alert</TH>
                <TH>Watchlists</TH>
                {admin ? <TH className="text-right">Actions</TH> : null}
              </TR>
            </THead>
            <TBody>
              {(pairs.data?.items ?? []).map((p) => (
                <PairRow key={`${p.id}-${p.status}-${p.alert_threshold}-${p.alert_change_threshold}-${p.monitoring_interval_minutes}-${p.industry}`} p={p} admin={admin} watchlists={wls.data ?? []} onChange={refresh} />
              ))}
            </TBody>
          </Table>
          <p className="mt-1 text-[11px] text-muted">
            Archiving stops monitoring but keeps every historical score, event and prediction. Monitoring interval controls how often news is checked for the pair (minimum 15 minutes).
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>Watchlists</CardTitle>
          {admin ? (
            <Button size="sm" variant="secondary" onClick={() => setEditing("new")}>
              <Plus /> New watchlist
            </Button>
          ) : null}
        </CardHeader>
        <CardContent>
          {err ? <p className="mb-2 text-xs text-risk">{err}</p> : null}
          <Table>
            <THead>
              <TR>
                <TH>Name</TH>
                <TH>Description</TH>
                <TH className="text-right">Pairs</TH>
                {admin ? <TH className="text-right">Actions</TH> : null}
              </TR>
            </THead>
            <TBody>
              {(wls.data ?? []).map((w) => (
                <TR key={w.id}>
                  <TD className="font-medium">
                    {w.name} {w.is_default ? <Badge variant="default">default</Badge> : null}
                  </TD>
                  <TD className="text-xs text-muted">{w.description}</TD>
                  <TD className="num text-right">{w.pair_ids.length}</TD>
                  {admin ? (
                    <TD className="text-right">
                      <Button size="sm" variant="ghost" onClick={() => setEditing(w)}>
                        Edit
                      </Button>
                      {!w.is_default ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          aria-label={`Delete ${w.name}`}
                          onClick={async () => {
                            setErr(null);
                            try {
                              await apiSend(`/watchlists/${w.id}`, "DELETE");
                              refresh();
                            } catch (e) {
                              setErr((e as Error).message);
                            }
                          }}
                        >
                          <Trash2 />
                        </Button>
                      ) : null}
                    </TD>
                  ) : null}
                </TR>
              ))}
            </TBody>
          </Table>
        </CardContent>
      </Card>
      {editing ? <WatchlistEditor wl={editing === "new" ? null : editing} pairs={all.data?.items ?? []} onClose={() => setEditing(null)} onSaved={refresh} /> : null}
    </div>
  );
}

/* ------------------------------------------------------------------ scoring */
function ScoringTab({ s, admin, reload }: { s: Sections["scoring"]; admin: boolean; reload: () => void }) {
  const [w, setW] = useState<Record<string, string>>(Object.fromEntries(Object.entries(s.si_v2_weights).map(([k, v]) => [k, String(v)])));
  const [cov, setCov] = useState(String(s.min_coverage));
  const versions = useApi<ModelVersion[]>("/settings/model-versions");
  const saver = useSaver();
  const total = Object.values(w).reduce((a, b) => a + (Number(b) || 0), 0);
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader>
          <CardTitle>SI-v2 weights</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-2 text-xs text-muted">
            Weights are research hypotheses, not fitted values. They are normalised to sum to 1. Saving new weights creates a new immutable model version; historical scores keep the version they were computed with.
          </p>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-6">
            {["N", "F", "M", "P", "S"].map((k) => (
              <Field key={k} label={`${k} · ${COMPONENT_NAMES[k]}`} hint={total > 0 ? `${((100 * (Number(w[k]) || 0)) / total).toFixed(1)}% effective` : undefined}>
                <Input type="number" min={0} step="1" value={w[k] ?? "0"} disabled={!admin} onChange={(e) => setW({ ...w, [k]: e.target.value })} aria-label={`Weight ${k}`} />
              </Field>
            ))}
            <Field label="Minimum coverage" hint="Share of weight that must be available for a provisional score">
              <Input type="number" min={0.05} max={1} step="0.05" value={cov} disabled={!admin} onChange={(e) => setCov(e.target.value)} aria-label="Minimum coverage" />
            </Field>
          </div>
          {admin ? (
            <SaveBar
              saver={saver}
              disabled={total <= 0}
              onSave={() =>
                saver.save("scoring", { si_v2_weights: Object.fromEntries(Object.entries(w).map(([k, v]) => [k, Number(v) || 0])), min_coverage: Number(cov) }, () => {
                  reload();
                  versions.mutate();
                })
              }
            />
          ) : null}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Model versions</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <THead>
              <TR>
                <TH>Index</TH>
                <TH>Version</TH>
                <TH>Weights</TH>
                <TH>Created</TH>
                <TH>Active</TH>
              </TR>
            </THead>
            <TBody>
              {(versions.data ?? []).map((m) => (
                <TR key={m.id}>
                  <TD>{m.index_name === "si_v1" ? "SI-v1" : "SI-v2"}</TD>
                  <TD className="font-mono text-xs">{m.version}</TD>
                  <TD className="num text-xs">
                    {m.weights
                      ? Object.entries(m.weights)
                          .map(([k, v]) => `${k} ${v.toFixed(2)}`)
                          .join(" · ")
                      : "—"}
                  </TD>
                  <TD className="text-xs">{fmtDateTime(m.created_at)}</TD>
                  <TD>{m.is_active ? <Badge>active</Badge> : <span className="text-xs text-muted">retired</span>}</TD>
                </TR>
              ))}
            </TBody>
          </Table>
          <p className="mt-1 text-[11px] text-muted">
            See docs/METHODOLOGY.md for every component definition, normalisation and the reconstructed SI-v1 formula.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ news & data */
function DataTab({ news, prices, admin, reload, configured }: { news: Sections["news"]; prices: Sections["prices"]; admin: boolean; reload: () => void; configured: Record<string, boolean> }) {
  const [prov, setProv] = useState(news.providers);
  const [lookback, setLookback] = useState(String(news.lookback_hours));
  const [templates, setTemplates] = useState(news.rss_templates.join("\n"));
  const [feeds, setFeeds] = useState(
    Object.entries(news.company_feeds)
      .flatMap(([t, urls]) => urls.map((u) => `${t} ${u}`))
      .join("\n"),
  );
  const [priceProv, setPriceProv] = useState(prices.providers.join(", "));
  const [start, setStart] = useState(prices.history_start);
  const [bench, setBench] = useState(prices.benchmark);
  const newsSaver = useSaver();
  const priceSaver = useSaver();
  function parseFeeds(): Record<string, string[]> {
    const out: Record<string, string[]> = {};
    for (const line of feeds.split("\n").map((l) => l.trim()).filter(Boolean)) {
      const [t, ...rest] = line.split(/\s+/);
      const url = rest.join(" ");
      if (!t || !url) continue;
      (out[t.toUpperCase()] ??= []).push(url);
    }
    return out;
  }
  const labels: Record<string, string> = { sec: "SEC EDGAR filings", finnhub: "Finnhub company news", alphavantage: "Alpha Vantage news", rss: "RSS / IR feeds" };
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader>
          <CardTitle>News sources</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-2 md:grid-cols-4">
            {Object.entries(prov).map(([k, v]) => (
              <label key={k} className="flex items-center justify-between rounded border border-border p-2 text-xs">
                <span>
                  {labels[k] ?? k}
                  <span className="block text-[11px] text-muted">{k === "rss" ? "uses the feeds below" : configured[k] ? "key configured" : "not configured on server"}</span>
                </span>
                <Switch checked={v} disabled={!admin} onCheckedChange={(c) => setProv({ ...prov, [k]: c })} aria-label={`Enable ${k}`} />
              </label>
            ))}
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="Lookback (hours)" hint="Window fetched on each ingest run">
              <Input type="number" min={1} max={720} value={lookback} disabled={!admin} onChange={(e) => setLookback(e.target.value)} />
            </Field>
            <Field label="RSS feed templates" hint="One URL per line; {ticker} is replaced by each symbol. Only use feeds you are licensed to read.">
              <Textarea rows={4} value={templates} disabled={!admin} onChange={(e) => setTemplates(e.target.value)} className="font-mono text-[11px]" />
            </Field>
            <Field label="Investor-relations feeds" hint="One per line: TICKER https://… (press releases)">
              <Textarea rows={4} value={feeds} disabled={!admin} onChange={(e) => setFeeds(e.target.value)} className="font-mono text-[11px]" />
            </Field>
          </div>
          {admin ? (
            <SaveBar
              saver={newsSaver}
              onSave={() =>
                newsSaver.save(
                  "news",
                  {
                    providers: prov,
                    lookback_hours: Number(lookback),
                    rss_templates: templates.split("\n").map((l) => l.trim()).filter(Boolean),
                    company_feeds: parseFeeds(),
                  },
                  reload,
                )
              }
            />
          ) : null}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Prices</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="Providers in priority order" hint="tiingo (total-return adjusted, needs key), stooq (free, split-adjusted, flagged price-return)">
              <Input value={priceProv} disabled={!admin} onChange={(e) => setPriceProv(e.target.value)} />
            </Field>
            <Field label="History start">
              <Input type="date" value={start} disabled={!admin} onChange={(e) => setStart(e.target.value)} />
            </Field>
            <Field label="Market benchmark" hint="Used for betas in the structural component">
              <Input value={bench} disabled={!admin} onChange={(e) => setBench(e.target.value.toUpperCase())} maxLength={10} />
            </Field>
          </div>
          {admin ? (
            <SaveBar
              saver={priceSaver}
              onSave={() =>
                priceSaver.save(
                  "prices",
                  { providers: priceProv.split(",").map((x) => x.trim().toLowerCase()).filter(Boolean), history_start: start, benchmark: bench },
                  reload,
                )
              }
            />
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ AI */
function AiTab({ llm, budget, env, admin, reload }: { llm: Sections["llm"]; budget: Sections["budget"]; env: SettingsResponse["environment"]; admin: boolean; reload: () => void }) {
  const [model, setModel] = useState(llm.model ?? "");
  const [effort, setEffort] = useState(llm.effort ?? "");
  const [maxDocs, setMaxDocs] = useState(String(llm.max_documents_per_run));
  const [fallback, setFallback] = useState(llm.refusal_fallback);
  const [routine, setRoutine] = useState(llm.analyze_routine_filings);
  const [limit, setLimit] = useState(String(budget.monthly_ai_limit_usd));
  const [warn, setWarn] = useState(String(budget.warn_at_fraction));
  const llmSaver = useSaver();
  const budgetSaver = useSaver();
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader>
          <CardTitle>Claude news analyst</CardTitle>
        </CardHeader>
        <CardContent>
          {!env.providers_configured.anthropic ? (
            <Notice>ANTHROPIC_API_KEY is not set on the server. Stories are stored but not analysed, and the news component stays unavailable.</Notice>
          ) : null}
          <div className="mt-2 grid gap-3 md:grid-cols-3">
            <Field label="Model" hint={`Blank uses the server default (${env.claude_model_default}).`}>
              <Select className="w-full" value={CLAUDE_MODELS.includes(model) || model === "" ? model : "__custom"} disabled={!admin} onChange={(e) => setModel(e.target.value === "__custom" ? "claude-" : e.target.value)}>
                <option value="">Server default</option>
                {CLAUDE_MODELS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
                <option value="__custom">Other…</option>
              </Select>
              {model && !CLAUDE_MODELS.includes(model) ? <Input className="mt-1" value={model} disabled={!admin} onChange={(e) => setModel(e.target.value)} aria-label="Custom model id" /> : null}
            </Field>
            <Field label="Effort" hint={`Blank uses ${env.claude_effort_default}. Higher effort is slower and costs more.`}>
              <Select className="w-full" value={effort} disabled={!admin} onChange={(e) => setEffort(e.target.value)}>
                <option value="">Server default</option>
                {["low", "medium", "high", "xhigh", "max"].map((x) => (
                  <option key={x} value={x}>{x}</option>
                ))}
              </Select>
            </Field>
            <Field label="Max stories analysed per run">
              <Input type="number" min={1} max={1000} value={maxDocs} disabled={!admin} onChange={(e) => setMaxDocs(e.target.value)} />
            </Field>
          </div>
          <div className="mt-3 flex flex-wrap gap-6 text-xs">
            <label className="flex items-center gap-2">
              <Switch checked={fallback} disabled={!admin} onCheckedChange={setFallback} /> Retry on a fallback model if a request is refused
            </label>
            <label className="flex items-center gap-2">
              <Switch checked={routine} disabled={!admin} onCheckedChange={setRoutine} /> Analyse routine SEC filings (10-Q, proxy, etc.)
            </label>
          </div>
          {admin ? (
            <SaveBar
              saver={llmSaver}
              onSave={() =>
                llmSaver.save(
                  "llm",
                  { model: model || null, effort: effort || null, max_documents_per_run: Number(maxDocs), refusal_fallback: fallback, analyze_routine_filings: routine },
                  reload,
                )
              }
            />
          ) : null}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>AI budget</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="Monthly limit (USD)" hint="Analysis stops for the month once reached; stories stay queued.">
              <Input type="number" min={0} step="1" value={limit} disabled={!admin} onChange={(e) => setLimit(e.target.value)} aria-label="Monthly AI limit" />
            </Field>
            <Field label="Warn at (fraction of limit)">
              <Input type="number" min={0} max={1} step="0.05" value={warn} disabled={!admin} onChange={(e) => setWarn(e.target.value)} />
            </Field>
          </div>
          {admin ? <SaveBar saver={budgetSaver} onSave={() => budgetSaver.save("budget", { monthly_ai_limit_usd: Number(limit), warn_at_fraction: Number(warn) }, reload)} /> : null}
          <div className="mt-3">
            <div className="mb-1 text-xs font-medium">Pricing used for cost tracking (USD per million tokens)</div>
            <Table>
              <THead>
                <TR>
                  <TH>Model</TH>
                  <TH className="text-right">Input</TH>
                  <TH className="text-right">Output</TH>
                  <TH className="text-right">Cache read</TH>
                  <TH className="text-right">Cache write</TH>
                </TR>
              </THead>
              <TBody>
                {Object.entries(llm.pricing).map(([m, p]) => (
                  <TR key={m}>
                    <TD className="font-mono text-xs">{m}</TD>
                    <TD className="num text-right">{p.input}</TD>
                    <TD className="num text-right">{p.output}</TD>
                    <TD className="num text-right">{p.cache_read}</TD>
                    <TD className="num text-right">{p.cache_write}</TD>
                  </TR>
                ))}
              </TBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ alerts & predictions */
function TargetsTab({ alerts, pred, admin, reload }: { alerts: Sections["alerts"]; pred: Sections["predictions"]; admin: boolean; reload: () => void }) {
  const [thr, setThr] = useState(String(alerts.default_threshold));
  const [chg, setChg] = useState(String(alerts.default_change_threshold));
  const [floor, setFloor] = useState(String(alerts.confidence_floor));
  const [horizons, setHorizons] = useState(pred.horizons.join(", "));
  const [base, setBase] = useState(String(pred.base_threshold));
  const [scaling, setScaling] = useState(pred.threshold_scaling);
  const [overrides, setOverrides] = useState(
    Object.entries(pred.threshold_overrides)
      .map(([h, v]) => `${h}=${v}`)
      .join(", "),
  );
  const [mult, setMult] = useState(pred.sensitivity_multipliers.join(", "));
  const alertSaver = useSaver();
  const predSaver = useSaver();
  const nums = (s: string) =>
    s
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean)
      .map(Number);
  const parsedOverrides = useMemo(
    () =>
      Object.fromEntries(
        overrides
          .split(",")
          .map((x) => x.split("=").map((y) => y.trim()))
          .filter((x) => x.length === 2 && x[0] && x[1])
          .map(([h, v]) => [h, Number(v)]),
      ) as Record<string, number>,
    [overrides],
  );
  const preview = useMemo(() => {
    const hs = nums(horizons).filter((h) => h > 0);
    const ov = parsedOverrides;
    return hs.map((h) => ({ h, t: ov[String(h)] !== undefined ? Number(ov[String(h)]) : scaling === "constant" ? Number(base) : Number(base) * Math.sqrt(h / pred.base_horizon) }));
  }, [horizons, parsedOverrides, scaling, base, pred.base_horizon]);
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader>
          <CardTitle>Alert defaults</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="SI-v2 alert threshold" hint="Applied to new pairs; each pair can override it">
              <Input type="number" min={0} max={100} value={thr} disabled={!admin} onChange={(e) => setThr(e.target.value)} />
            </Field>
            <Field label="7-day change alert (points)">
              <Input type="number" min={0} max={100} value={chg} disabled={!admin} onChange={(e) => setChg(e.target.value)} />
            </Field>
            <Field label="Minimum evidence confidence for alerts">
              <Input type="number" min={0} max={1} step="0.05" value={floor} disabled={!admin} onChange={(e) => setFloor(e.target.value)} />
            </Field>
          </div>
          {admin ? (
            <SaveBar saver={alertSaver} onSave={() => alertSaver.save("alerts", { default_threshold: Number(thr), default_change_threshold: Number(chg), confidence_floor: Number(floor) }, reload)} />
          ) : null}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Prediction horizons and divergence targets</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-2 text-xs text-muted">
            D(t,H) = |ln(PA(t+H)/PA(t)) − ln(PB(t+H)/PB(t))| on total-return prices. The {pred.base_horizon}-session threshold of {pred.base_threshold} reproduces the original experiment. Changing a target creates a new target version; existing snapshots keep theirs.
          </p>
          <div className="grid gap-3 md:grid-cols-3">
            <Field label="Horizons (trading days)">
              <Input value={horizons} disabled={!admin} onChange={(e) => setHorizons(e.target.value)} />
            </Field>
            <Field label={`Threshold at ${pred.base_horizon} days`}>
              <Input type="number" step="0.01" min={0.01} value={base} disabled={!admin} onChange={(e) => setBase(e.target.value)} />
            </Field>
            <Field label="Scaling for other horizons">
              <Select className="w-full" value={scaling} disabled={!admin} onChange={(e) => setScaling(e.target.value)}>
                <option value="sqrt">√(H/{pred.base_horizon}) (random-walk dispersion)</option>
                <option value="constant">Same threshold for every horizon</option>
              </Select>
            </Field>
            <Field label="Per-horizon overrides" hint="e.g. 30=0.08, 90=0.12">
              <Input value={overrides} disabled={!admin} onChange={(e) => setOverrides(e.target.value)} />
            </Field>
            <Field label="Sensitivity multipliers" hint="Threshold multiples evaluated on the performance page">
              <Input value={mult} disabled={!admin} onChange={(e) => setMult(e.target.value)} />
            </Field>
            <div className="text-xs">
              <Label>Effective thresholds</Label>
              <div className="num">{preview.map((p) => `${p.h}d: ${Number.isFinite(p.t) ? p.t.toFixed(3) : "—"}`).join(" · ")}</div>
            </div>
          </div>
          {admin ? (
            <SaveBar
              saver={predSaver}
              onSave={() =>
                predSaver.save(
                  "predictions",
                  {
                    horizons: nums(horizons),
                    base_threshold: Number(base),
                    threshold_scaling: scaling,
                    threshold_overrides: parsedOverrides,
                    sensitivity_multipliers: nums(mult),
                  },
                  reload,
                )
              }
            />
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ system & account */
function SystemTab({ env, admin }: { env: SettingsResponse["environment"]; admin: boolean }) {
  const me = useUser();
  const users = useApi<User[]>(admin ? "/auth/users" : null);
  const [health, setHealth] = useState<Record<string, { ok: boolean | null; latency_ms?: number; error?: string; note?: string }> | null>(null);
  const [checking, setChecking] = useState(false);
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [pwMsg, setPwMsg] = useState<{ kind: "info" | "risk"; text: string } | null>(null);
  const [nu, setNu] = useState({ email: "", password: "", role: "viewer" });
  const [nuMsg, setNuMsg] = useState<string | null>(null);

  async function check() {
    setChecking(true);
    try {
      setHealth(await apiSend("/settings/health-check", "POST"));
    } catch (e) {
      setHealth({ error: { ok: false, error: (e as Error).message } });
    } finally {
      setChecking(false);
    }
  }
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>Server environment</CardTitle>
          {admin ? (
            <Button size="sm" variant="secondary" onClick={check} disabled={checking} data-testid="health-check">
              <Activity /> {checking ? "Checking…" : "Health check"}
            </Button>
          ) : null}
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <div className="grid gap-1 md:grid-cols-3">
            <div>
              <span className="text-muted">Mode:</span> {env.mode === "demo" ? <Badge variant="default">DEMO, simulated data</Badge> : "production"}
            </div>
            <div>
              <span className="text-muted">Scheduler timezone:</span> {env.scheduler_timezone}
            </div>
            <div>
              <span className="text-muted">Claude default:</span> {env.claude_model_default} / {env.claude_effort_default}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(env.providers_configured).map(([k, v]) => (
              <Badge key={k} variant={v ? "ok" : "default"}>
                {k}: {v ? "configured" : "not set"}
              </Badge>
            ))}
          </div>
          <p className="text-muted">{env.note}</p>
          {health ? (
            <Table data-testid="health-results">
              <THead>
                <TR>
                  <TH>Provider</TH>
                  <TH>Result</TH>
                  <TH>Detail</TH>
                </TR>
              </THead>
              <TBody>
                {Object.entries(health).map(([k, r]) => (
                  <TR key={k}>
                    <TD>{k}</TD>
                    <TD>{r.ok === true ? <span className="text-ok">reachable</span> : r.ok === false ? <span className="text-risk">failed</span> : <span className="text-muted">not tested</span>}</TD>
                    <TD className="text-xs">{r.latency_ms !== undefined ? `${r.latency_ms} ms` : r.error ?? r.note ?? ""}</TD>
                  </TR>
                ))}
              </TBody>
            </Table>
          ) : null}
        </CardContent>
      </Card>
      <div className="grid gap-3 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Change password</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-xs text-muted">Signed in as {me?.email}</p>
            <Field label="Current password">
              <Input type="password" autoComplete="current-password" value={cur} onChange={(e) => setCur(e.target.value)} />
            </Field>
            <Field label="New password" hint="At least 12 characters">
              <Input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
            </Field>
            {pwMsg ? <p className={pwMsg.kind === "risk" ? "text-xs text-risk" : "text-xs text-ok"}>{pwMsg.text}</p> : null}
            <div className="flex justify-end">
              <Button
                size="sm"
                disabled={next.length < 12 || !cur}
                onClick={async () => {
                  setPwMsg(null);
                  try {
                    await apiSend("/auth/change-password", "POST", { current_password: cur, new_password: next });
                    setCur("");
                    setNext("");
                    setPwMsg({ kind: "info", text: "Password changed." });
                  } catch (e) {
                    setPwMsg({ kind: "risk", text: (e as Error).message });
                  }
                }}
              >
                Change password
              </Button>
            </div>
          </CardContent>
        </Card>
        {admin ? (
          <Card>
            <CardHeader>
              <CardTitle>Users</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Table>
                <THead>
                  <TR>
                    <TH>Email</TH>
                    <TH>Role</TH>
                  </TR>
                </THead>
                <TBody>
                  {(users.data ?? []).map((u) => (
                    <TR key={u.id}>
                      <TD>{u.email}</TD>
                      <TD>{u.role}</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
              <div className="grid grid-cols-3 gap-2">
                <Input placeholder="email" value={nu.email} onChange={(e) => setNu({ ...nu, email: e.target.value })} aria-label="New user email" />
                <Input placeholder="password (12+)" type="password" autoComplete="new-password" value={nu.password} onChange={(e) => setNu({ ...nu, password: e.target.value })} aria-label="New user password" />
                <Select className="w-full" value={nu.role} onChange={(e) => setNu({ ...nu, role: e.target.value })} aria-label="New user role">
                  <option value="viewer">viewer (read only)</option>
                  <option value="admin">admin</option>
                </Select>
              </div>
              {nuMsg ? <p className="text-xs text-risk">{nuMsg}</p> : null}
              <div className="flex justify-end">
                <Button
                  size="sm"
                  disabled={!nu.email || nu.password.length < 12}
                  onClick={async () => {
                    setNuMsg(null);
                    try {
                      await apiSend("/auth/users", "POST", nu);
                      setNu({ email: "", password: "", role: "viewer" });
                      users.mutate();
                    } catch (e) {
                      setNuMsg((e as Error).message);
                    }
                  }}
                >
                  <Plus /> Add user
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : null}
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const admin = useIsAdmin();
  const settings = useApi<SettingsResponse>("/settings");
  const [tab, setTab] = useState("pairs");
  useEffect(() => {
    const h = window.location.hash.replace("#", "");
    if (h) setTab(h);
  }, []);
  const d = settings.data;
  // Re-key the forms whenever the stored values change so local edits reset after a save.
  const key = d ? JSON.stringify(d.sections) : "";
  return (
    <div className="space-y-3">
      <div>
        <h1 className="text-lg font-semibold">Settings</h1>
        <p className="text-xs text-muted">
          {admin ? "Changes apply on the next scheduled run." : "You have read-only access; ask an admin to change settings."} API keys live in server environment variables and are never shown here.
        </p>
      </div>
      {settings.error ? <Notice kind="risk">Could not load settings: {String(settings.error.message)}</Notice> : null}
      <Tabs
        value={tab}
        onValueChange={(v) => {
          setTab(v);
          history.replaceState(null, "", `#${v}`);
        }}
      >
        <TabsList>
          <TabsTrigger value="pairs">Pairs & watchlists</TabsTrigger>
          <TabsTrigger value="scoring">Scoring</TabsTrigger>
          <TabsTrigger value="data">News & data</TabsTrigger>
          <TabsTrigger value="ai">AI & budget</TabsTrigger>
          <TabsTrigger value="targets">Alerts & targets</TabsTrigger>
          <TabsTrigger value="system">System & account</TabsTrigger>
        </TabsList>
        <TabsContent value="pairs">
          <PairsTab admin={admin} />
        </TabsContent>
        {d ? (
          <>
            <TabsContent value="scoring">
              <ScoringTab key={`s${key}`} s={d.sections.scoring} admin={admin} reload={() => settings.mutate()} />
            </TabsContent>
            <TabsContent value="data">
              <DataTab key={`d${key}`} news={d.sections.news} prices={d.sections.prices} admin={admin} reload={() => settings.mutate()} configured={d.environment.providers_configured} />
            </TabsContent>
            <TabsContent value="ai">
              <AiTab key={`a${key}`} llm={d.sections.llm} budget={d.sections.budget} env={d.environment} admin={admin} reload={() => settings.mutate()} />
            </TabsContent>
            <TabsContent value="targets">
              <TargetsTab key={`t${key}`} alerts={d.sections.alerts} pred={d.sections.predictions} admin={admin} reload={() => settings.mutate()} />
            </TabsContent>
            <TabsContent value="system">
              <SystemTab env={d.environment} admin={admin} />
            </TabsContent>
          </>
        ) : (
          <Skeleton className="mt-3 h-48 w-full" />
        )}
      </Tabs>
    </div>
  );
}
