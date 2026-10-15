"use client";
import { Plus } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Input, Label } from "@/components/ui/input";
import { ApiError, apiSend } from "@/lib/api";
import type { Pair, Watchlist } from "@/lib/types";

export function AddPairDialog({ watchlists, onCreated }: { watchlists: Watchlist[]; onCreated: () => void }) {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ ticker_a: "", ticker_b: "", industry: "", alert_threshold: "70", monitoring_interval_minutes: "60" });
  const [wl, setWl] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await apiSend<Pair>("/pairs", "POST", {
        ticker_a: form.ticker_a,
        ticker_b: form.ticker_b,
        industry: form.industry || null,
        alert_threshold: Number(form.alert_threshold),
        monitoring_interval_minutes: Number(form.monitoring_interval_minutes),
        watchlist_ids: wl,
      });
      setOpen(false);
      setForm({ ticker_a: "", ticker_b: "", industry: "", alert_threshold: "70", monitoring_interval_minutes: "60" });
      onCreated();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to add pair");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm">
          <Plus /> Add pair
        </Button>
      </DialogTrigger>
      <DialogContent title="Add pair" description="Reversed duplicates (e.g. PEP/KO when KO/PEP exists) are rejected automatically.">
        <form onSubmit={submit} className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <Label htmlFor="ta">Ticker A</Label>
            <Input id="ta" value={form.ticker_a} onChange={(e) => setForm({ ...form, ticker_a: e.target.value.toUpperCase() })} required maxLength={10} />
          </div>
          <div className="space-y-1">
            <Label htmlFor="tb">Ticker B</Label>
            <Input id="tb" value={form.ticker_b} onChange={(e) => setForm({ ...form, ticker_b: e.target.value.toUpperCase() })} required maxLength={10} />
          </div>
          <div className="col-span-2 space-y-1">
            <Label htmlFor="ind">Industry</Label>
            <Input id="ind" value={form.industry} onChange={(e) => setForm({ ...form, industry: e.target.value })} placeholder="e.g. Beverages" />
          </div>
          <div className="space-y-1">
            <Label htmlFor="thr">Alert threshold (SI-v2)</Label>
            <Input id="thr" type="number" min={0} max={100} value={form.alert_threshold} onChange={(e) => setForm({ ...form, alert_threshold: e.target.value })} />
          </div>
          <div className="space-y-1">
            <Label htmlFor="freq">News check interval (min)</Label>
            <Input id="freq" type="number" min={15} value={form.monitoring_interval_minutes} onChange={(e) => setForm({ ...form, monitoring_interval_minutes: e.target.value })} />
          </div>
          {watchlists.length ? (
            <div className="col-span-2 space-y-1">
              <Label>Watchlists</Label>
              <div className="flex flex-wrap gap-2">
                {watchlists.map((w) => (
                  <label key={w.id} className="flex items-center gap-1 text-xs">
                    <input type="checkbox" checked={wl.includes(w.id)} onChange={(e) => setWl(e.target.checked ? [...wl, w.id] : wl.filter((x) => x !== w.id))} />
                    {w.name}
                  </label>
                ))}
              </div>
            </div>
          ) : null}
          {error ? <p className="col-span-2 text-xs text-risk" role="alert">{error}</p> : null}
          <div className="col-span-2 flex justify-end">
            <Button type="submit" disabled={busy}>
              {busy ? "Adding…" : "Add pair"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
