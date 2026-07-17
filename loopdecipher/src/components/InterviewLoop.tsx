import Card from "@/components/ui/Card";
import type { InterviewRound, Question } from "@/lib/types";

const SOURCE_STYLE: Record<InterviewRound["source"], string> = {
  "forum-sourced": "text-emerald-400 border-emerald-800 bg-emerald-950/40",
  typical: "text-slate-400 border-slate-700 bg-slate-800/40",
};

const SOURCE_LABEL: Record<InterviewRound["source"], string> = {
  "forum-sourced": "confirmed from forum reports",
  typical: "typical for this role — not company-confirmed",
};

export default function InterviewLoop({
  rounds,
  questions,
}: {
  rounds: InterviewRound[];
  questions: Question[];
}) {
  const questionById = new Map(questions.map((q) => [q.id, q]));

  if (rounds.length === 0) return null;

  return (
    <Card>
      <h2 className="mb-1 text-lg font-semibold text-slate-100">Interview Loop</h2>
      <p className="mb-5 text-sm text-slate-400">
        The rounds to expect, in order. Rounds marked &ldquo;typical&rdquo; are a reasonable estimate
        for this role/level, not a confirmed report of this company&rsquo;s process.
      </p>

      <ol className="space-y-5">
        {rounds
          .slice()
          .sort((a, b) => a.order - b.order)
          .map((round) => (
            <li key={round.order} className="border-l-2 border-slate-800 pl-4">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-slate-100">{round.name}</h3>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${SOURCE_STYLE[round.source]}`}
                  title={round.sourceDetail}
                >
                  {SOURCE_LABEL[round.source]}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-500">{round.format}</p>

              <div className="mt-2 flex flex-wrap gap-1.5">
                {round.focus.map((f) => (
                  <span
                    key={f}
                    className="rounded-md border border-indigo-800 bg-indigo-950/30 px-2 py-0.5 text-xs text-indigo-300"
                  >
                    {f}
                  </span>
                ))}
              </div>

              {round.sampleQuestionIds.length > 0 && (
                <ul className="mt-2 space-y-1 text-sm text-slate-400">
                  {round.sampleQuestionIds.map((id) => {
                    const q = questionById.get(id);
                    if (!q) return null;
                    return (
                      <li key={id} className="flex gap-2">
                        <span className="text-slate-600">·</span>
                        {q.text}
                      </li>
                    );
                  })}
                </ul>
              )}

              {round.sourceDetail && (
                <p className="mt-1 text-xs text-slate-600">{round.sourceDetail}</p>
              )}
            </li>
          ))}
      </ol>
    </Card>
  );
}
