import { clsx } from "clsx";

interface Props {
  label: string;
  value: string | number;
  unit?: string;
  target?: number;
  direction?: "up-good" | "down-good";
  subtitle?: string;
  size?: "sm" | "md";
}

function getStatus(value: number, target: number, direction: "up-good" | "down-good"): "green" | "yellow" | "red" {
  const ratio = value / target;
  if (direction === "up-good") {
    if (ratio >= 0.98) return "green";
    if (ratio >= 0.93) return "yellow";
    return "red";
  } else {
    if (ratio <= 1.02) return "green";
    if (ratio <= 1.2) return "yellow";
    return "red";
  }
}

export function MetricCard({ label, value, unit, target, direction = "up-good", subtitle, size = "md" }: Props) {
  const numVal = typeof value === "number" ? value : parseFloat(String(value));
  const status = target !== undefined && !isNaN(numVal) ? getStatus(numVal, target, direction) : null;

  return (
    <div className={clsx(
      "bg-gray-900 border rounded-lg",
      status === "green" && "border-emerald-700",
      status === "yellow" && "border-amber-600",
      status === "red" && "border-red-700",
      !status && "border-gray-700",
      size === "sm" ? "p-3" : "p-4"
    )}>
      <p className="text-gray-400 text-xs font-medium uppercase tracking-wider">{label}</p>
      <div className="flex items-end gap-1 mt-1">
        <span className={clsx(
          "font-bold tabular-nums",
          size === "sm" ? "text-xl" : "text-2xl",
          status === "green" && "text-emerald-400",
          status === "yellow" && "text-amber-400",
          status === "red" && "text-red-400",
          !status && "text-white"
        )}>
          {typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : value}
        </span>
        {unit && <span className="text-gray-500 text-sm mb-0.5">{unit}</span>}
      </div>
      {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
      {target !== undefined && (
        <p className="text-gray-600 text-xs mt-1">Target: {target}{unit}</p>
      )}
    </div>
  );
}
