const BASE = "http://localhost:8001";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export interface DailyMetric {
  date: string;
  fill_rate_pct: number;
  open_backorders: number;
  critical_backorders: number;
  supplier_otd_pct: number;
  production_attainment_pct: number;
  freight_exception_count: number;
  freight_on_time_pct: number;
  warranty_claims_count: number;
  warranty_cost: number;
  inventory_value: number;
  active_pos: number;
  po_overdue_count: number;
  alerts?: Array<{ level: string; metric: string; message: string }>;
}

export interface WeeklyMetric {
  week_start: string;
  week_end: string;
  inventory_turns: number;
  forecast_mape: number;
  avg_fill_rate_pct: number;
  avg_supplier_otd_pct: number;
  avg_production_attainment_pct: number;
  total_freight_cost: number;
  total_warranty_cost: number;
  total_po_value: number;
  excess_inventory_value: number;
  short_inventory_parts: number;
  top_warranty_part: string | null;
}

export interface Anomaly {
  part_number: string;
  part_description: string;
  z_score: number;
  recent_avg_claims_per_day: number;
  baseline_avg_claims_per_day: number;
  severity: "warning" | "critical";
}

export interface SupplierPerf {
  supplier_id: string;
  name: string;
  total: number;
  on_time: number;
  late: number;
  open: number;
  otd_pct: number | null;
}

export interface FreightSummary {
  carrier: string;
  total: number;
  on_time: number;
  late: number;
  cost: number;
  otd_pct: number | null;
}

export interface WarrantySummary {
  part_number: string;
  description: string;
  count: number;
  cost: number;
}

export const api = {
  getToday: () => get<DailyMetric>("/api/metrics/daily/today"),
  getDailyHistory: (days = 30) => get<DailyMetric[]>(`/api/metrics/daily?days=${days}`),
  getWeeklyHistory: (weeks = 12) => get<WeeklyMetric[]>(`/api/metrics/weekly?weeks=${weeks}`),
  getAnomalies: () => get<Anomaly[]>("/api/metrics/anomalies"),
  getSupplierPerf: () => get<SupplierPerf[]>("/api/metrics/suppliers/performance"),
  getFreightSummary: () => get<FreightSummary[]>("/api/metrics/freight/summary"),
  getWarrantySummary: () => get<WarrantySummary[]>("/api/metrics/warranty/summary"),
  getInventorySummary: () => get<{
    snapshot_date: string;
    total_skus: number;
    in_stock: number;
    total_value: number;
    by_location: Array<{ location: string; skus: number; value: number; in_stock: number }>;
  }>("/api/metrics/inventory/summary"),
  triggerIngestion: () => post("/api/ingest/run"),
  computeDaily: () => post<DailyMetric>("/api/metrics/daily/compute"),
  computeWeekly: () => post<WeeklyMetric>("/api/metrics/weekly/compute"),
};
