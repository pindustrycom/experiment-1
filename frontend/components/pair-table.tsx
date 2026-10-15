"use client";
import { ArrowDown, ArrowUp } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import { ConfidenceMeter, Delta, DirectionBadge, Empty, ScoreBar, ScoreCell, SimulatedTag } from "@/components/domain";
import { Badge } from "@/components/ui/badge";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { ago } from "@/lib/format";
import type { Pair } from "@/lib/types";
import { cn } from "@/lib/utils";

type SortKey = "slug" | "si_v1" | "si_v2" | "change" | "confidence" | "updated";

function sortValue(p: Pair, k: SortKey): string | number | null {
  switch (k) {
    case "slug":
      return p.slug;
    case "si_v1":
      return p.si_v1?.value ?? null;
    case "si_v2":
      return p.si_v2?.value ?? null;
    case "change":
      return p.si_v2?.change_7d ?? null;
    case "confidence":
      return p.confidence;
    case "updated":
      return p.last_update ? new Date(p.last_update).getTime() : null;
  }
}

export function sortPairs(pairs: Pair[], key: SortKey, dir: "asc" | "desc"): Pair[] {
  return [...pairs].sort((x, y) => {
    const a = sortValue(x, key);
    const b = sortValue(y, key);
    if (a === null && b === null) return 0;
    if (a === null) return 1; // unavailable values always last
    if (b === null) return -1;
    const c = a < b ? -1 : a > b ? 1 : 0;
    return dir === "asc" ? c : -c;
  });
}

export function PairTable({ pairs }: { pairs: Pair[] }) {
  const [key, setKey] = useState<SortKey>("si_v2");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const rows = useMemo(() => sortPairs(pairs, key, dir), [pairs, key, dir]);

  function Header({ k, children, className }: { k: SortKey; children: React.ReactNode; className?: string }) {
    const active = key === k;
    return (
      <TH className={className}>
        <button
          className={cn("inline-flex items-center gap-1 uppercase", active && "text-fg")}
          onClick={() => {
            if (active) setDir(dir === "asc" ? "desc" : "asc");
            else {
              setKey(k);
              setDir(k === "slug" ? "asc" : "desc");
            }
          }}
          aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : "none"}
        >
          {children}
          {active ? dir === "asc" ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" /> : null}
        </button>
      </TH>
    );
  }

  if (!pairs.length) return <Empty>No pairs match the current filters.</Empty>;
  return (
    <Table data-testid="pair-table">
      <THead>
        <TR>
          <Header k="slug">Pair</Header>
          <Header k="si_v1" className="text-right">SI-v1</Header>
          <Header k="si_v2">SI-v2</Header>
          <Header k="change" className="text-right">7-day Δ</Header>
          <TH>Direction</TH>
          <Header k="confidence">Confidence</Header>
          <Header k="updated" className="text-right">Last update</Header>
        </TR>
      </THead>
      <TBody>
        {rows.map((p) => (
          <TR key={p.id} className="hover:bg-panel-2/60">
            <TD>
              <Link href={`/pairs/${p.slug}`} className="group flex items-center gap-2">
                <span className="font-mono text-xs font-semibold group-hover:text-accent">{p.slug.replace("-", " / ")}</span>
                {p.status !== "active" ? <Badge>{p.status}</Badge> : null}
                <SimulatedTag show={p.is_demo} />
              </Link>
              <div className="text-[10px] text-muted">{p.industry ?? "Unclassified"}</div>
            </TD>
            <TD className="text-right">
              <ScoreCell value={p.si_v1?.value} status={p.si_v1?.status} threshold={p.alert_threshold} />
            </TD>
            <TD>
              <div className="flex items-center gap-2">
                <span className="w-14 text-right">
                  <ScoreCell value={p.si_v2?.value} status={p.si_v2?.status} threshold={p.alert_threshold} />
                </span>
                <ScoreBar value={p.si_v2?.value} threshold={p.alert_threshold} />
              </div>
            </TD>
            <TD className="text-right">
              <Delta value={p.si_v2?.change_7d} />
            </TD>
            <TD>
              <DirectionBadge d={p.direction} a={p.company_a.ticker} b={p.company_b.ticker} />
            </TD>
            <TD>
              <ConfidenceMeter value={p.confidence} />
            </TD>
            <TD className="text-right text-[11px] text-muted" title={p.last_update ?? ""}>
              {ago(p.last_update)}
            </TD>
          </TR>
        ))}
      </TBody>
    </Table>
  );
}
