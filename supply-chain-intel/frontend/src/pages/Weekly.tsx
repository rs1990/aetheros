import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line
} from "recharts";
import { api, WeeklyMetric, SupplierPerf, FreightSummary, WarrantySummary } from "../api/client";
import { MetricCard } from "../components/MetricCard";

export function Weekly() {
  const [weekly, setWeekly] = useState<WeeklyMetric[]>([]);
  const [suppliers, setSuppliers] = useState<SupplierPerf[]>([]);
  const [freight, setFreight] = useState<FreightSummary[]>([]);
  const [warranty, setWarranty] = useState<WarrantySummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.getWeeklyHistory(12),
      api.getSupplierPerf(),
      api.getFreightSummary(),
      api.getWarrantySummary(),
    ]).then(([w, s, f, wa]) => {
      setWeekly(w); setSuppliers(s); setFreight(f); setWarranty(wa);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500 p-8">Loading...</div>;

  const latest = weekly[weekly.length - 1];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">Weekly Intelligence</h1>
        {latest && <p className="text-gray-500 text-sm">Week of {latest.week_start}</p>}
      </div>

      {/* This week KPIs */}
      {latest && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="Inventory Turns" value={latest.inventory_turns} subtitle="Annualized" />
          <MetricCard label="Forecast MAPE" value={latest.forecast_mape} unit="%" direction="down-good" target={15} subtitle="Mean Abs Pct Error" />
          <MetricCard label="Avg Fill Rate" value={latest.avg_fill_rate_pct} unit="%" target={95} />
          <MetricCard label="Avg Supplier OTD" value={latest.avg_supplier_otd_pct} unit="%" target={90} />
          <MetricCard label="Freight Cost" value={`$${(latest.total_freight_cost / 1000).toFixed(0)}k`} />
          <MetricCard label="Warranty Cost" value={`$${(latest.total_warranty_cost / 1000).toFixed(0)}k`} direction="down-good" />
          <MetricCard label="PO Value" value={`$${(latest.total_po_value / 1000).toFixed(0)}k`} />
          <MetricCard label="Short Parts" value={latest.short_inventory_parts} direction="down-good" subtitle={latest.top_warranty_part ? `Top warranty: ${latest.top_warranty_part}` : undefined} />
        </div>
      )}

      {/* Trend charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <TrendChart title="Fill Rate Trend (12 weeks)" data={weekly} dataKey="avg_fill_rate_pct" color="#34d399" unit="%" />
        <TrendChart title="Supplier OTD Trend" data={weekly} dataKey="avg_supplier_otd_pct" color="#60a5fa" unit="%" />
        <BarChartCard title="Freight Cost (weekly)" data={weekly} dataKey="total_freight_cost" color="#fb923c" unit="$" />
        <BarChartCard title="Warranty Cost (weekly)" data={weekly} dataKey="total_warranty_cost" color="#f87171" unit="$" />
      </div>

      {/* Supplier performance table */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h2 className="text-white font-semibold mb-3">Supplier On-Time Delivery (30d)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-2 pr-4">Supplier</th>
                <th className="text-right py-2 pr-4">POs</th>
                <th className="text-right py-2 pr-4">On-Time</th>
                <th className="text-right py-2 pr-4">Late</th>
                <th className="text-right py-2">OTD %</th>
              </tr>
            </thead>
            <tbody>
              {suppliers.map((s) => (
                <tr key={s.supplier_id} className="border-b border-gray-800 hover:bg-gray-800">
                  <td className="py-2 pr-4 text-white">{s.name || s.supplier_id}</td>
                  <td className="py-2 pr-4 text-right text-gray-300">{s.total}</td>
                  <td className="py-2 pr-4 text-right text-emerald-400">{s.on_time}</td>
                  <td className="py-2 pr-4 text-right text-red-400">{s.late}</td>
                  <td className="py-2 text-right">
                    <span className={s.otd_pct !== null ? (s.otd_pct >= 90 ? "text-emerald-400" : s.otd_pct >= 80 ? "text-amber-400" : "text-red-400") : "text-gray-500"}>
                      {s.otd_pct !== null ? `${s.otd_pct}%` : "—"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Warranty by part */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h2 className="text-white font-semibold mb-3">Top Warranty Parts (30d)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-2 pr-4">Part</th>
                <th className="text-left py-2 pr-4">Description</th>
                <th className="text-right py-2 pr-4">Claims</th>
                <th className="text-right py-2">Cost</th>
              </tr>
            </thead>
            <tbody>
              {warranty.slice(0, 10).map((w) => (
                <tr key={w.part_number} className="border-b border-gray-800 hover:bg-gray-800">
                  <td className="py-2 pr-4 text-white font-mono text-xs">{w.part_number}</td>
                  <td className="py-2 pr-4 text-gray-300">{w.description}</td>
                  <td className="py-2 pr-4 text-right text-amber-400">{w.count}</td>
                  <td className="py-2 text-right text-gray-300">${w.cost.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Freight by carrier */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h2 className="text-white font-semibold mb-3">Carrier Performance (30d)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-2 pr-4">Carrier</th>
                <th className="text-right py-2 pr-4">Shipments</th>
                <th className="text-right py-2 pr-4">Late</th>
                <th className="text-right py-2 pr-4">OTD %</th>
                <th className="text-right py-2">Cost</th>
              </tr>
            </thead>
            <tbody>
              {freight.map((f) => (
                <tr key={f.carrier} className="border-b border-gray-800 hover:bg-gray-800">
                  <td className="py-2 pr-4 text-white">{f.carrier}</td>
                  <td className="py-2 pr-4 text-right text-gray-300">{f.total}</td>
                  <td className="py-2 pr-4 text-right text-red-400">{f.late}</td>
                  <td className="py-2 pr-4 text-right">
                    <span className={f.otd_pct !== null ? (f.otd_pct >= 92 ? "text-emerald-400" : "text-amber-400") : "text-gray-500"}>
                      {f.otd_pct !== null ? `${f.otd_pct}%` : "—"}
                    </span>
                  </td>
                  <td className="py-2 text-right text-gray-300">${f.cost.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function TrendChart({ title, data, dataKey, color, unit }: {
  title: string; data: WeeklyMetric[]; dataKey: keyof WeeklyMetric; color: string; unit?: string;
}) {
  const chartData = data.map(d => ({ week: d.week_start.slice(5), value: d[dataKey] as number }));
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <h3 className="text-gray-300 text-sm font-medium mb-3">{title}</h3>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="week" tick={{ fill: "#6b7280", fontSize: 10 }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} domain={["auto", "auto"]} />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151" }} formatter={(v: number) => [`${v?.toFixed(1)}${unit || ""}`, ""]} />
          <Line type="monotone" dataKey="value" stroke={color} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function BarChartCard({ title, data, dataKey, color, unit }: {
  title: string; data: WeeklyMetric[]; dataKey: keyof WeeklyMetric; color: string; unit?: string;
}) {
  const chartData = data.map(d => ({ week: d.week_start.slice(5), value: d[dataKey] as number }));
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <h3 className="text-gray-300 text-sm font-medium mb-3">{title}</h3>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="week" tick={{ fill: "#6b7280", fontSize: 10 }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151" }} formatter={(v: number) => [`${unit}${(v / 1000).toFixed(0)}k`, ""]} />
          <Bar dataKey="value" fill={color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
