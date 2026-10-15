"use client";
import { useState } from "react";
import { ArticleCard } from "@/components/article-card";
import { Empty, Notice } from "@/components/domain";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input, Label, Select } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { useApi } from "@/lib/api";
import { eventLabel } from "@/lib/format";
import type { Article, Page, Pair } from "@/lib/types";

const EVENT_TYPES = [
  "executive_change", "acquisition_merger", "product_launch", "product_recall", "regulatory_change",
  "lawsuit_investigation", "earnings_surprise", "guidance_change", "supply_chain_disruption",
  "competitive_development", "market_share_change", "consumer_demand_change", "restructuring_layoffs",
  "major_partnership", "debt_financing", "industry_shock", "other", "not_relevant",
];

const PROVIDERS = ["sec", "finnhub", "alphavantage", "rss", "demo"];

export default function NewsPage() {
  const [pair, setPair] = useState("");
  const [ticker, setTicker] = useState("");
  const [eventType, setEventType] = useState("");
  const [provider, setProvider] = useState("");
  const [status, setStatus] = useState("");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [materialOnly, setMaterialOnly] = useState(false);
  const [changedOnly, setChangedOnly] = useState(false);
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const qs = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
  if (pair) qs.set("pair", pair);
  if (ticker.trim()) qs.set("ticker", ticker.trim().toUpperCase());
  if (eventType) qs.set("event_type", eventType);
  if (provider) qs.set("provider", provider);
  if (status) qs.set("status", status);
  if (since) qs.set("since", new Date(`${since}T00:00:00`).toISOString());
  if (until) qs.set("until", new Date(`${until}T23:59:59`).toISOString());
  if (materialOnly) qs.set("material_only", "true");
  if (changedOnly) qs.set("changed_only", "true");

  const news = useApi<Page<Article>>(`/news?${qs}`, { refreshInterval: 120000 });
  const pairs = useApi<Page<Pair>>("/pairs?status=all&page_size=500");
  const reset = <T,>(fn: (v: T) => void) => (v: T) => {
    fn(v);
    setPage(1);
  };
  const pages = news.data ? Math.max(1, Math.ceil(news.data.total / pageSize)) : 1;

  return (
    <div className="space-y-3">
      <div>
        <h1 className="text-lg font-semibold">News intelligence</h1>
        <p className="text-xs text-muted">
          Stories are deduplicated across sources. AI interpretations are shown separately from the source text; only quotes found verbatim in the source count as evidence.
        </p>
      </div>
      <Card>
        <CardContent className="grid grid-cols-2 gap-2 pt-3 md:grid-cols-4 xl:grid-cols-8">
          <div>
            <Label>Pair</Label>
            <Select className="w-full" value={pair} onChange={(e) => reset(setPair)(e.target.value)} aria-label="Pair">
              <option value="">All pairs</option>
              {(pairs.data?.items ?? []).map((p) => (
                <option key={p.id} value={p.slug}>{p.slug}</option>
              ))}
            </Select>
          </div>
          <div>
            <Label>Ticker</Label>
            <Input value={ticker} maxLength={10} placeholder="e.g. KO" onChange={(e) => reset(setTicker)(e.target.value)} aria-label="Ticker" />
          </div>
          <div>
            <Label>Event type</Label>
            <Select className="w-full" value={eventType} onChange={(e) => reset(setEventType)(e.target.value)} aria-label="Event type">
              <option value="">Any</option>
              {EVENT_TYPES.map((t) => (
                <option key={t} value={t}>{eventLabel(t)}</option>
              ))}
            </Select>
          </div>
          <div>
            <Label>Source</Label>
            <Select className="w-full" value={provider} onChange={(e) => reset(setProvider)(e.target.value)} aria-label="Source">
              <option value="">Any</option>
              {PROVIDERS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </Select>
          </div>
          <div>
            <Label>Analysis</Label>
            <Select className="w-full" value={status} onChange={(e) => reset(setStatus)(e.target.value)} aria-label="Analysis status">
              <option value="">Any</option>
              <option value="analyzed">Analysed</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
              <option value="skipped">Skipped</option>
              <option value="budget_blocked">Blocked by AI budget</option>
              <option value="not_configured">AI not configured</option>
            </Select>
          </div>
          <div>
            <Label>From</Label>
            <Input type="date" value={since} onChange={(e) => reset(setSince)(e.target.value)} aria-label="From date" />
          </div>
          <div>
            <Label>To</Label>
            <Input type="date" value={until} onChange={(e) => reset(setUntil)(e.target.value)} aria-label="To date" />
          </div>
          <div className="flex flex-col justify-end gap-1.5 text-xs">
            <label className="flex items-center gap-2">
              <Switch checked={materialOnly} onCheckedChange={reset(setMaterialOnly)} aria-label="Material only" /> Material only
            </label>
            <label className="flex items-center gap-2">
              <Switch checked={changedOnly} onCheckedChange={reset(setChangedOnly)} aria-label="Changed index only" /> Changed an index
            </label>
          </div>
        </CardContent>
      </Card>

      {news.error ? <Notice kind="risk">Could not load news: {String(news.error.message)}</Notice> : null}
      {!news.data && !news.error ? (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      ) : null}
      {news.data && news.data.items.length === 0 ? <Empty>No stories match these filters.</Empty> : null}
      <div className="space-y-2" data-testid="news-list">
        {news.data?.items.map((a) => <ArticleCard key={a.id} a={a} />)}
      </div>
      {news.data && news.data.total > pageSize ? (
        <div className="flex items-center justify-end gap-2 text-xs">
          <span className="text-muted">
            Page {page} of {pages} · {news.data.total} stories
          </span>
          <Button size="sm" variant="outline" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
            Previous
          </Button>
          <Button size="sm" variant="outline" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
            Next
          </Button>
        </div>
      ) : null}
    </div>
  );
}
