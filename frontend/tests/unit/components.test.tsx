import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ScoreCell, SimulatedTag } from "@/components/domain";
import { PairTable, sortPairs } from "@/components/pair-table";
import type { Pair } from "@/lib/types";

vi.mock("next/link", () => ({ default: ({ href, children, ...p }: { href: string; children: React.ReactNode }) => <a href={href} {...p}>{children}</a> }));

function pair(slug: string, v2: number | null, opts: Partial<Pair> = {}): Pair {
  const [a, b] = slug.split("-");
  const company = (t: string, id: number) => ({ id, ticker: t, name: `${t} Inc.`, cik: null, sector: null, industry: null, sic_code: null, listing_status: "listed" });
  return {
    id: Math.floor(Math.random() * 1e6),
    uid: slug,
    slug,
    company_a: company(a, 1),
    company_b: company(b, 2),
    industry: "Test",
    status: "active",
    monitoring_interval_minutes: 60,
    alert_threshold: 70,
    alert_change_threshold: 10,
    notes: null,
    created_at: "2026-01-01T00:00:00Z",
    archived_at: null,
    watchlist_ids: [],
    si_v1: { value: 40, status: "complete", as_of: "2026-10-14T22:00:00Z", change_7d: null, model_version: "1.0" },
    si_v2: v2 === null ? { value: null, status: "unavailable", as_of: "2026-10-14T22:00:00Z", change_7d: null, model_version: null } : { value: v2, status: "complete", as_of: "2026-10-14T22:00:00Z", change_7d: 2, model_version: "2.0" },
    direction: "neutral",
    confidence: 0.5,
    explanation: null,
    last_update: "2026-10-14T22:00:00Z",
    is_demo: false,
    ...opts,
  };
}

describe("sortPairs", () => {
  it("sorts by SI-v2 and keeps unavailable scores last in both directions", () => {
    const ps = [pair("A-B", 30), pair("C-D", null), pair("E-F", 80)];
    expect(sortPairs(ps, "si_v2", "desc").map((p) => p.slug)).toEqual(["E-F", "A-B", "C-D"]);
    expect(sortPairs(ps, "si_v2", "asc").map((p) => p.slug)).toEqual(["A-B", "E-F", "C-D"]);
  });
});

describe("ScoreCell", () => {
  it("renders n/a for unavailable scores instead of 0", () => {
    render(<ScoreCell value={null} />);
    expect(screen.getByText("n/a")).toBeInTheDocument();
  });
  it("marks provisional scores", () => {
    render(<ScoreCell value={55.2} status="provisional" />);
    expect(screen.getByText("55.2")).toBeInTheDocument();
    expect(screen.getByText("prov")).toBeInTheDocument();
  });
});

describe("SimulatedTag", () => {
  it("is shown only for demo data", () => {
    const { rerender } = render(<SimulatedTag show />);
    expect(screen.getByText(/simulated/i)).toBeInTheDocument();
    rerender(<SimulatedTag show={false} />);
    expect(screen.queryByText(/simulated/i)).not.toBeInTheDocument();
  });
});

describe("PairTable", () => {
  it("lists pairs highest risk first and re-sorts on header click", () => {
    render(<PairTable pairs={[pair("KO-PEP", 26), pair("V-MA", 61), pair("XOM-CVX", null)]} />);
    const order = () => screen.getAllByRole("row").slice(1).map((r) => within(r).getByRole("link").getAttribute("href"));
    expect(order()).toEqual(["/pairs/V-MA", "/pairs/KO-PEP", "/pairs/XOM-CVX"]);
    fireEvent.click(screen.getByRole("button", { name: /^pair/i }));
    expect(order()).toEqual(["/pairs/KO-PEP", "/pairs/V-MA", "/pairs/XOM-CVX"]);
    fireEvent.click(screen.getByRole("button", { name: /^si-v2/i }));
    fireEvent.click(screen.getByRole("button", { name: /^si-v2/i }));
    expect(order()).toEqual(["/pairs/KO-PEP", "/pairs/V-MA", "/pairs/XOM-CVX"]);
  });

  it("labels demo pairs as simulated", () => {
    render(<PairTable pairs={[pair("KO-PEP", 26, { is_demo: true })]} />);
    expect(screen.getByText(/simulated/i)).toBeInTheDocument();
  });
});
