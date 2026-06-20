import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";
import { api, DailyMetric, Anomaly } from "../api/client";
import { MetricCard } from "../components/MetricCard";
import { clsx } from "clsx";

export function Daily() {
  const [today, setToday] = useState<DailyMetric | null>(null);
  const [history, setHistory] = useState<DailyMetric[]>([]);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    Promise.all([api.getToday(), api.getDailyHistory(30), api.getAnomalies()])
      .then(([t, h, a]) => { setToday(t); setHistory(h); setAnomalies(a); })
      .finally(() => setLoading(false));
  }, []);

  const refresh = async () => {
    setRefreshing(true);
    try {
      const [t, h, a] = await Promise.all([api.computeDaily(), api.getDailyHistory(30), api.getAnomalies()]);
      setToday(t); setHistory(h); setAnomalies(a);
    } finally {
      setRefreshing(false);
    }
  };

  if (loading) return <div className="text-gray-500 p-8">Loading...</div>;
  if (!today) return <div className="text-red-400 p-8">Failed to load metrics</div>;

  const alerts = today.alerts || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Daily Operations</h1>
          <p className="text-gray-500 text-sm">{today.date}</p>
        </div>
        <button
          onClick={refresh}
          disabled={refreshing}
          className="px-4 py-2 bg-blue-700 hover:bg-blue-600 disabled:opacity-50 text-white text-sm rounded-lg font-medium"
        >
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.map((a, i) => (
            <div key={i} className={clsx(
              "flex items-start gap-3 p-3 rounded-lg border text-sm",
              a.level === "critical" ? "bg-red-950 border-red-700 text-red-300" : "bg-amber-950 border-amber-700 text-amber-300"
            )}>
              <span>{a.level === "critical" ? "●" : "◆"}</span>
              <span>{a.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Fill Rate" value={today.fill_rate_pct} unit="%" target={95} direction="up-good" />
        <MetricCard label="Supplier OTD" value={today.supplier_otd_pct} unit="%" target={90} direction="up-good" />
        <MetricCard label="Production Attainment" value={today.production_attainment_pct} unit="%" target={95} direction="up-good" />
        <MetricCard label="Freight On-Time" value={today.freight_on_time_pct} unit="%" target={92} direction="up-good" />
        <MetricCard label="Open Backorders" value={today.open_backorders} direction="down-good" />
        <MetricCard label="Critical Backorders" value={today.critical_backorders} direction="down-good" subtitle={today.critical_backorders > 0 ? "Requires immediate action" : "Clear"} />
        <MetricCard label="Freight Exceptions" value={today.freight_exception_count} direction="down-good" subtitle="Last 7 days" />
        <MetricCard label="Warranty Claims" value={today.warranty_claims_count} direction="down-good" subtitle={`$${(today.warranty_cost / 1000).toFixed(0)}k cost (7d)`} />
      </div>

      {/* Trend charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartCard title="Fill Rate (30d)" data={history} dataKey="fill_rate_pct" target={95} unit="%" color="#34d399" />
        <ChartCard title="Supplier OTD (30d)" data={history} dataKey="supplier_otd_pct" target={90} unit="%" color="#60a5fa" />
        <ChartCard title="Production Attainment (30d)" data={history} dataKey="production_attainment_pct" target={95} unit="%" color="#a78bfa" />
        <ChartCard title="Freight On-Time (30d)" data={history} dataKey="freight_on_time_pct" target={92} unit="%" color="#fb923c" />
      </div>

      {/* Warranty anomalies */}
      {anomalies.length > 0 && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h2 className="text-white font-semibold mb-3">Warranty Anomalies Detected</h2>
          <div className="space-y-2">
            {anomalies.map((a, i) => (
              <div key={i} className={clsx(
                "flex items-center justify-between p-3 rounded-lg border text-sm",
                a.severity === "critical" ? "bg-red-950 border-red-700" : "bg-amber-950 border-amber-700"
              )}>
                <div>
                  <span className="text-white font-medium">{a.part_number}</span>
                  <span className="text-gray-400 ml-2">{a.part_description}</span>
                </div>
                <div className="text-right">
                  <div className={a.severity === "critical" ? "text-red-400" : "text-amber-400"}>
                    z={a.z_score} · {a.recent_avg_claims_per_day.toFixed(1)}/day recent
                  </div>
                  <div className="text-gray-500">vs {a.baseline_avg_claims_per_day.toFixed(1)}/day baseline</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ChartCard({ title, data, dataKey, target, unit, color }: {
  title: string;
  data: DailyMetric[];
  dataKey: keyof DailyMetric;
  target?: number;
  unit?: string;
  color: string;
}) {
  const chartData = data.map(d => ({ date: d.date.slice(5), value: d[dataKey] as number }));
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <h3 className="text-gray-300 text-sm font-medium mb-3">{title}</h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="date" tick={{ fill: "#6b7280", fontSize: 10 }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 6 }}
            labelStyle={{ color: "#9ca3af" }}
            formatter={(v: number) => [`${v?.toFixed(1)}${unit || ""}`, ""]}
          />
          {target && <ReferenceLine y={target} stroke="#ef4444" strokeDasharray="4 2" strokeWidth={1} />}
          <Line type="monotone" dataKey="value" stroke={color} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
