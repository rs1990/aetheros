"use client";

import { useMemo, useState } from "react";
import Card from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import type { Difficulty, Question, QuestionCategory } from "@/lib/types";

const CATEGORIES: QuestionCategory[] = ["Technical", "System Design", "Coding", "Behavioral"];
const DIFFICULTIES: Difficulty[] = ["Easy", "Medium", "Hard"];

const DIFFICULTY_COLOR: Record<Difficulty, string> = {
  Easy: "text-emerald-400 border-emerald-800 bg-emerald-950/40",
  Medium: "text-amber-400 border-amber-800 bg-amber-950/40",
  Hard: "text-red-400 border-red-800 bg-red-950/40",
};

export default function QuestionBank({ questions }: { questions: Question[] }) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<QuestionCategory | "All">("All");
  const [difficulty, setDifficulty] = useState<Difficulty | "All">("All");
  const [forumOnly, setForumOnly] = useState(false);
  const [completed, setCompleted] = useState<Record<string, boolean>>({});

  const filtered = useMemo(() => {
    return questions.filter((q) => {
      if (category !== "All" && q.category !== category) return false;
      if (difficulty !== "All" && q.difficulty !== difficulty) return false;
      if (forumOnly && q.source !== "forum-sourced") return false;
      if (search && !q.text.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [questions, category, difficulty, forumOnly, search]);

  const forumCount = questions.filter((q) => q.source === "forum-sourced").length;

  function toggleCompleted(id: string) {
    setCompleted((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  return (
    <Card>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">
          Question Bank <span className="text-sm font-normal text-slate-500">({filtered.length} / {questions.length})</span>
        </h2>
        {forumCount > 0 && (
          <span className="rounded-full border border-indigo-800 bg-indigo-950/50 px-3 py-1 text-xs text-indigo-300">
            {forumCount} forum-verified
          </span>
        )}
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search questions..."
          className="flex-1 min-w-[180px]"
        />
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value as QuestionCategory | "All")}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
        >
          <option value="All">All Categories</option>
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value as Difficulty | "All")}
          className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
        >
          <option value="All">All Difficulties</option>
          {DIFFICULTIES.map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
        <label className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300">
          <input type="checkbox" checked={forumOnly} onChange={(e) => setForumOnly(e.target.checked)} />
          Forum-sourced only
        </label>
      </div>

      <ul className="max-h-[600px] space-y-2 overflow-y-auto pr-1">
        {filtered.map((q) => (
          <li
            key={q.id}
            className={`flex items-start gap-3 rounded-lg border border-slate-800 p-3 ${completed[q.id] ? "opacity-50" : ""}`}
          >
            <input
              type="checkbox"
              checked={!!completed[q.id]}
              onChange={() => toggleCompleted(q.id)}
              className="mt-1"
            />
            <div className="flex-1">
              <p className="text-sm text-slate-200">{q.text}</p>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <span className="rounded-full border border-slate-700 px-2 py-0.5 text-xs text-slate-400">
                  {q.category}
                </span>
                <span className={`rounded-full border px-2 py-0.5 text-xs ${DIFFICULTY_COLOR[q.difficulty]}`}>
                  {q.difficulty}
                </span>
                {q.source === "forum-sourced" && (
                  <span className="rounded-full border border-indigo-800 bg-indigo-950/50 px-2 py-0.5 text-xs text-indigo-300">
                    Forum-Verified{q.sourceDetail ? ` · ${q.sourceDetail}` : ""}
                  </span>
                )}
              </div>
            </div>
          </li>
        ))}
        {filtered.length === 0 && (
          <li className="py-8 text-center text-sm text-slate-500">No questions match your filters.</li>
        )}
      </ul>
    </Card>
  );
}
