"use client";
import { CheckCircle2, ChevronDown, ChevronRight, ExternalLink, XCircle } from "lucide-react";
import { useState } from "react";
import { SimulatedTag, StatusBadge } from "@/components/domain";
import { Badge } from "@/components/ui/badge";
import { eventLabel, fmtDateTime, fmtNum, fmtPct, fmtSigned } from "@/lib/format";
import type { Article, NewsEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

function directionText(d: string) {
  return d.replace(/_/g, " ");
}

export function EventDetail({ e }: { e: NewsEvent }) {
  return (
    <div className="space-y-2 rounded-md border border-border bg-panel-2/50 p-2.5 text-xs" data-testid="event-detail">
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
        <div><span className="text-muted">Scope</span> {e.scope.replace(/_/g, " ")}</div>
        <div><span className="text-muted">Severity</span> <span className="num">{fmtNum(e.severity, 2)}</span></div>
        <div><span className="text-muted">Asymmetry</span> <span className="num">{fmtNum(e.asymmetry, 2)}</span></div>
        <div><span className="text-muted">Persistence</span> <span className="num">{e.expected_persistence_days}d</span></div>
        <div><span className="text-muted">Source reliability</span> <span className="num">{fmtNum(e.source_reliability, 2)}</span></div>
        <div><span className="text-muted">AI certainty</span> <span className="num">{fmtNum(e.extraction_confidence, 2)}</span></div>
        <div><span className="text-muted">Quotes verified</span> <span className="num">{fmtPct(e.evidence_verified_ratio)}</span></div>
        <div><span className="text-muted">New info</span> {e.is_new_information ? "yes" : "repeat"}</div>
      </div>
      <div>
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Source-supported evidence</div>
        {e.evidence.length === 0 ? <p className="text-muted">No quotes — not scored.</p> : null}
        <ul className="space-y-1">
          {e.evidence.map((q, i) => (
            <li key={i} className="flex gap-1.5">
              {q.verified ? (
                <CheckCircle2 className="mt-0.5 size-3 shrink-0 text-ok" aria-label="quote found verbatim in source" />
              ) : (
                <XCircle className="mt-0.5 size-3 shrink-0 text-risk" aria-label="quote not found in source" />
              )}
              <div>
                <q className="italic">{q.quote}</q>
                <span className="text-muted"> — {q.supports}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
      {Object.keys(e.company_exposure || {}).length ? (
        <div>
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Company-specific exposure</div>
          {Object.entries(e.company_exposure).map(([t, x]) => (
            <div key={t}>
              <span className="font-mono">{t}</span>: {x.exposure} / {x.effect} — <span className="text-muted">{x.rationale}</span>
            </div>
          ))}
        </div>
      ) : null}
      {e.uncertainties.length ? (
        <div>
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted">Uncertainties</div>
          <ul className="list-disc pl-4 text-muted">
            {e.uncertainties.map((u, i) => (
              <li key={i}>{u}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <div className="text-[10px] text-muted">
        Effect on SI-v2:{" "}
        {e.index_effect ? (
          <>
            N weight {e.index_effect.n_weight === null ? "0 (not counted)" : fmtNum(e.index_effect.n_weight, 3)}; SI-v2 {fmtNum(e.index_effect.si_v2_before)} → {fmtNum(e.index_effect.si_v2_after)} ({fmtSigned(e.index_effect.si_v2_delta)}) in score #{e.index_effect.score_id}
          </>
        ) : (
          "not yet scored"
        )}
        {" · "}interpretation by {e.model_name} ({e.prompt_version}) {fmtDateTime(e.analyzed_at)}. AI interpretation is stored separately from the verified source facts above; the index uses deterministic rules.
      </div>
    </div>
  );
}

export function ArticleCard({ a, defaultOpen = false }: { a: Article; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const primary = a.documents[0];
  const material = a.events.filter((e) => e.is_material);
  return (
    <div className="border-b border-border px-3 py-2.5 last:border-0" data-testid="article">
      <div className="flex items-start gap-2">
        <button onClick={() => setOpen(!open)} className="mt-0.5 text-muted hover:text-fg" aria-expanded={open} aria-label="Toggle evidence">
          {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted">
            <span className="num">{fmtDateTime(a.first_published_at)}</span>
            {a.tickers.map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
            {primary ? <span>{primary.publisher ?? primary.provider}</span> : null}
            {primary?.doc_type === "filing" ? <Badge variant="blue">{primary.filing_form}</Badge> : null}
            {primary?.doc_type === "press_release" ? <Badge variant="blue">official</Badge> : null}
            {a.document_count > 1 ? <span>· {a.document_count} copies / {a.distinct_publishers} publishers (deduplicated)</span> : null}
            <SimulatedTag show={a.is_demo} />
            {a.analysis_status !== "analyzed" ? <StatusBadge status={a.analysis_status} /> : null}
          </div>
          <div className="mt-0.5 text-xs font-medium">
            {a.headline}
            {primary ? (
              <a href={primary.url} target="_blank" rel="noopener noreferrer nofollow" className="ml-1 inline-flex text-muted hover:text-accent" aria-label="Open original">
                <ExternalLink className="size-3" />
              </a>
            ) : null}
          </div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {a.events.map((e) => (
              <span key={e.id} className="inline-flex items-center gap-1 text-[11px]">
                <Badge variant={e.is_material ? (e.scope === "company_specific" ? "warn" : "default") : "default"}>{eventLabel(e.event_type)}</Badge>
                <span className="font-mono text-muted">{e.pair_slug}</span>
                <span className={cn(e.relative_direction.endsWith("disadvantaged") ? "text-risk" : e.relative_direction.endsWith("advantaged") ? "text-ok" : "text-muted")}>
                  {directionText(e.relative_direction)}
                </span>
                {e.changed_index ? <Badge variant="accent">moved index</Badge> : null}
              </span>
            ))}
            {a.analysis_error ? <span className="text-[11px] text-warn">{a.analysis_error}</span> : null}
          </div>
          {material[0] ? <p className="mt-1 text-xs text-muted">{material[0].summary}</p> : null}
          {open ? (
            <div className="mt-2 space-y-2">
              {a.events.map((e) => (
                <EventDetail key={e.id} e={e} />
              ))}
              <div className="text-[10px] text-muted">
                Sources:{" "}
                {a.documents.map((d) => (
                  <a key={d.id} href={d.url} target="_blank" rel="noopener noreferrer nofollow" className="mr-2 underline hover:text-accent">
                    {d.publisher ?? d.provider} (published {fmtDateTime(d.published_at)}, ingested {fmtDateTime(d.ingested_at)})
                  </a>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
