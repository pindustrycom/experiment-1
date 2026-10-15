"use client";
import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ComposedChart, Tooltip, XAxis, YAxis } from "recharts";

export const tooltipStyle = { background: "var(--panel)", border: "1px solid var(--border)", fontSize: 11, color: "var(--fg)" };

type Pt = Record<string, number | string | null>;

export function NormalizedPrices({ data, a, b }: { data: Pt[]; a: string; b: string }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ left: -12, right: 8, top: 4 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="date" tickLine={false} minTickGap={40} />
        <YAxis tickLine={false} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="a" name={a} stroke="var(--accent)" dot={false} strokeWidth={1.4} />
        <Line dataKey="b" name={b} stroke="var(--accent-2)" dot={false} strokeWidth={1.4} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function SeriesArea({ data, name, color = "var(--accent)", threshold }: { data: Pt[]; name: string; color?: string; threshold?: number }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ left: -12, right: 8, top: 4 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="date" tickLine={false} minTickGap={40} />
        <YAxis tickLine={false} domain={["auto", "auto"]} />
        <Tooltip contentStyle={tooltipStyle} />
        <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="3 3" />
        {threshold !== undefined ? (
          <>
            <ReferenceLine y={threshold} stroke="var(--warn)" strokeDasharray="4 4" />
            <ReferenceLine y={-threshold} stroke="var(--warn)" strokeDasharray="4 4" />
          </>
        ) : null}
        <Area dataKey="value" name={name} stroke={color} fill={color} fillOpacity={0.12} strokeWidth={1.4} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function RollingCorr({ data }: { data: Pt[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ left: -12, right: 8, top: 4 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="date" tickLine={false} minTickGap={40} />
        <YAxis tickLine={false} domain={[-0.2, 1]} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="c63" name="63-day" stroke="var(--accent-2)" dot={false} strokeWidth={1.2} connectNulls />
        <Line dataKey="c252" name="252-day" stroke="var(--accent)" dot={false} strokeWidth={1.6} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ZScore({ data }: { data: Pt[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ left: -12, right: 8, top: 4 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="date" tickLine={false} minTickGap={40} />
        <YAxis tickLine={false} domain={[-4, 4]} allowDataOverflow />
        <Tooltip contentStyle={tooltipStyle} />
        <ReferenceArea y1={-2} y2={2} fill="var(--panel-2)" fillOpacity={0.6} />
        <ReferenceLine y={2} stroke="var(--warn)" strokeDasharray="4 4" />
        <ReferenceLine y={-2} stroke="var(--warn)" strokeDasharray="4 4" />
        <ReferenceLine y={0} stroke="var(--muted)" />
        <Line dataKey="z" name="Out-of-sample z" stroke="var(--accent)" dot={false} strokeWidth={1.3} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export interface HistoryPoint {
  t: number;
  si_v1?: number | null;
  si_v2?: number | null;
  v1id?: number;
  v2id?: number;
}

export function ScoreHistory({ data, events, onSelect, threshold }: { data: HistoryPoint[]; events: { t: number; y: number; label: string }[]; onSelect: (scoreId: number) => void; threshold: number }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart
        data={data}
        margin={{ left: -12, right: 8, top: 8 }}
        onClick={(st: any) => {
          const p = st?.activePayload?.[0]?.payload as HistoryPoint | undefined;
          if (p?.v2id) onSelect(p.v2id);
          else if (p?.v1id) onSelect(p.v1id);
        }}
      >
        <CartesianGrid strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={(v) => new Date(v).toISOString().slice(0, 7)} tickLine={false} minTickGap={40} />
        <YAxis tickLine={false} domain={[0, 100]} />
        <Tooltip contentStyle={tooltipStyle} labelFormatter={(v) => new Date(Number(v)).toISOString().slice(0, 10)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine y={threshold} stroke="var(--risk)" strokeDasharray="4 4" label={{ value: "alert", fill: "var(--risk)", fontSize: 10, position: "insideTopRight" }} />
        <Line dataKey="si_v1" name="SI-v1" stroke="var(--muted)" dot={false} strokeWidth={1.4} connectNulls />
        <Line dataKey="si_v2" name="SI-v2" stroke="var(--accent)" dot={false} strokeWidth={1.8} connectNulls />
        <Scatter data={events} dataKey="y" name="Material events" fill="var(--warn)" shape="diamond" legendType="diamond" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
