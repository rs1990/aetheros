import Card from "@/components/ui/Card";
import type { StudyWeek } from "@/lib/types";

export default function StudySchedule({ schedule }: { schedule: StudyWeek[] }) {
  return (
    <Card>
      <h2 className="mb-4 text-lg font-semibold text-slate-100">Study Roadmap</h2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {schedule.map((week) => (
          <div key={week.week} className="rounded-lg border border-slate-800 p-4">
            <div className="mb-2 flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
                {week.week}
              </span>
              <h3 className="text-sm font-semibold text-slate-200">{week.focus}</h3>
            </div>
            <ul className="space-y-1 text-xs text-slate-400">
              {week.tasks.map((task, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-slate-600">-</span>
                  {task}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </Card>
  );
}
