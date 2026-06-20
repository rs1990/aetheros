import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "../api/client";
import { MetricCard } from "../components/MetricCard";

interface InventorySummary {
  snapshot_date: string;
  total_skus: number;
  in_stock: number;
  total_value: number;
  by_location: Array<{ location: string; skus: number; value: number; in_stock: number }>;
}

export function Inventory() {
  const [summary, setSummary] = useState<InventorySummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getInventorySummary().then(setSummary).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-gray-500 p-8">Loading...</div>;
  if (!summary) return <div className="text-red-400 p-8">Failed to load inventory</div>;

  const fillRate = summary.total_skus > 0 ? (summary.in_stock / summary.total_skus * 100) : 0;
  const byLocChart = summary.by_location.map(l => ({
    name: l.location,
    value: Math.round(l.value / 1000),
    fill_pct: l.skus > 0 ? Math.round(l.in_stock / l.skus * 100) : 0,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">Inventory</h1>
        <p className="text-gray-500 text-sm">Snapshot: {summary.snapshot_date}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Total SKUs" value={summary.total_skus} />
        <MetricCard label="In Stock" value={summary.in_stock} subtitle={`${fillRate.toFixed(1)}% fill rate`} />
        <MetricCard label="Total Value" value={`$${(summary.total_value / 1_000_000).toFixed(1)}M`} />
        <MetricCard label="Locations" value={summary.by_location.length} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h3 className="text-gray-300 text-sm font-medium mb-3">Inventory Value by Location ($k)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={byLocChart} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 70 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 10 }} />
              <YAxis dataKey="name" type="category" tick={{ fill: "#d1d5db", fontSize: 11 }} width={70} />
              <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151" }} formatter={(v: number) => [`$${v}k`, "Value"]} />
              <Bar dataKey="value" fill="#60a5fa" radius={[0, 2, 2, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h3 className="text-gray-300 text-sm font-medium mb-3">Fill Rate by Location</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={byLocChart} layout="vertical" margin={{ top: 0, right: 8, bottom: 0, left: 70 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis type="number" domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 10 }} />
              <YAxis dataKey="name" type="category" tick={{ fill: "#d1d5db", fontSize: 11 }} width={70} />
              <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151" }} formatter={(v: number) => [`${v}%`, "Fill Rate"]} />
              <Bar dataKey="fill_pct" fill="#34d399" radius={[0, 2, 2, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h2 className="text-white font-semibold mb-3">Distribution Center Detail</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2 pr-4">Location</th>
              <th className="text-right py-2 pr-4">SKUs</th>
              <th className="text-right py-2 pr-4">In Stock</th>
              <th className="text-right py-2 pr-4">Fill Rate</th>
              <th className="text-right py-2">Value</th>
            </tr>
          </thead>
          <tbody>
            {summary.by_location.map(l => {
              const fr = l.skus > 0 ? l.in_stock / l.skus * 100 : 0;
              return (
                <tr key={l.location} className="border-b border-gray-800 hover:bg-gray-800">
                  <td className="py-2 pr-4 text-white">{l.location}</td>
                  <td className="py-2 pr-4 text-right text-gray-300">{l.skus}</td>
                  <td className="py-2 pr-4 text-right text-gray-300">{l.in_stock}</td>
                  <td className="py-2 pr-4 text-right">
                    <span className={fr >= 95 ? "text-emerald-400" : fr >= 90 ? "text-amber-400" : "text-red-400"}>
                      {fr.toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2 text-right text-gray-300">${(l.value / 1000).toFixed(0)}k</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
