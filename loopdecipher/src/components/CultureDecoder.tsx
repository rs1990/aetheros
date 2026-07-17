import Card from "@/components/ui/Card";
import type { CultureInsight } from "@/lib/types";

const SENTIMENT_STYLE: Record<CultureInsight["sentiment"], string> = {
  positive: "border-emerald-800 bg-emerald-950/30",
  negative: "border-red-800 bg-red-950/30",
  neutral: "border-slate-700 bg-slate-900/30",
};

export default function CultureDecoder({ insights }: { insights: CultureInsight[] }) {
  return (
    <Card>
      <h2 className="mb-1 text-lg font-semibold text-slate-100">Culture Decoder</h2>
      <p className="mb-4 text-xs text-slate-500">
        Signal compiled from public forum discussion and general interview patterns.
      </p>
      <ul className="space-y-3">
        {insights.map((insight, i) => (
          <li
            key={i}
            className={`rounded-lg border p-3 text-sm text-slate-200 ${SENTIMENT_STYLE[insight.sentiment]}`}
          >
            <p>{insight.insight}</p>
            <p className="mt-1 text-xs text-slate-500">— {insight.source}</p>
          </li>
        ))}
      </ul>
    </Card>
  );
}
