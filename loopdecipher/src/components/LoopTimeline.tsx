import Card from "@/components/ui/Card";
import type { StudyWeek } from "@/lib/types";

export default function LoopTimeline({ schedule }: { schedule: StudyWeek[] }) {
  return (
    <Card>
      <h2 className="mb-4 text-lg font-semibold text-slate-100">Prep Timeline</h2>
      <ol className="flex flex-wrap gap-4">
        {schedule.map((week, index) => (
          <li key={week.week} className="flex flex-1 min-w-[140px] items-center gap-3">
            <div className="flex flex-col items-center">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-sm font-semibold text-white">
                {week.week}
              </div>
              {index < schedule.length - 1 && (
                <div className="mt-1 h-full w-px flex-1 bg-slate-700 sm:hidden" />
              )}
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200">Week {week.week}</p>
              <p className="text-xs text-slate-400">{week.focus}</p>
            </div>
          </li>
        ))}
      </ol>
    </Card>
  );
}
