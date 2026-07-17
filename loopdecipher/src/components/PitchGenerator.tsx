"use client";

import { useMemo, useState } from "react";
import Card from "@/components/ui/Card";
import { Input, Textarea } from "@/components/ui/Input";

const REVERSE_QUESTION_TEMPLATES = [
  "What does success look like in this role after the first 90 days?",
  "How does the team decide what to work on next?",
  "What's the biggest technical challenge the team is tackling right now?",
  "How is {tech} used here, and what tradeoffs came with that choice?",
  "What does the on-call/incident process look like for this team?",
  "How do engineers here grow into more senior roles?",
];

export default function PitchGenerator({ mustKnowTech }: { mustKnowTech: string[] }) {
  const [role, setRole] = useState("");
  const [years, setYears] = useState("");
  const [strength, setStrength] = useState("");
  const [achievement, setAchievement] = useState("");

  const pitch = useMemo(() => {
    if (!role && !strength && !achievement) return "";
    return `Hi, I'm a ${years || "[X]"}-year ${role || "[your role]"} who focuses on ${strength || "[your core strength]"}. Most recently, ${achievement || "[a concrete achievement with a number attached]"}. I'm looking for a role where I can go deeper on ${mustKnowTech.slice(0, 2).join(" and ") || "problems like this"}, which is why this position stood out to me.`;
  }, [role, years, strength, achievement, mustKnowTech]);

  const reverseQuestions = useMemo(() => {
    const tech = mustKnowTech[0] || "the core stack";
    return REVERSE_QUESTION_TEMPLATES.map((q) => q.replace("{tech}", tech));
  }, [mustKnowTech]);

  return (
    <Card>
      <h2 className="mb-4 text-lg font-semibold text-slate-100">90-Second Pitch Builder</h2>
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <Input value={years} onChange={(e) => setYears(e.target.value)} placeholder="Years of experience" />
          <Input value={role} onChange={(e) => setRole(e.target.value)} placeholder="Your role/title" />
        </div>
        <Input value={strength} onChange={(e) => setStrength(e.target.value)} placeholder="Core strength (e.g. distributed systems)" />
        <Textarea
          value={achievement}
          onChange={(e) => setAchievement(e.target.value)}
          placeholder="A quantifiable achievement (e.g. cut p99 latency 40%)"
          rows={2}
        />
      </div>

      {pitch && (
        <div className="mt-4 rounded-lg border border-indigo-800 bg-indigo-950/30 p-3 text-sm text-slate-200">
          {pitch}
        </div>
      )}

      <h3 className="mb-2 mt-6 text-sm font-semibold text-slate-300">Reverse Questions to Ask</h3>
      <ul className="space-y-1.5 text-sm text-slate-400">
        {reverseQuestions.map((q, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-indigo-400">·</span>
            {q}
          </li>
        ))}
      </ul>
    </Card>
  );
}
