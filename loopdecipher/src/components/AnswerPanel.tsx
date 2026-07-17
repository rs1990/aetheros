"use client";

import { useState } from "react";
import Button from "@/components/ui/Button";
import MermaidDiagram from "@/components/MermaidDiagram";
import type { AnswerRequest, AnswerResult, Difficulty, QuestionCategory } from "@/lib/types";

interface AnswerPanelProps {
  questionId: string;
  questionText: string;
  category: QuestionCategory;
  difficulty: Difficulty;
  companyName?: string;
  roleName?: string;
  resumeText?: string;
}

export default function AnswerPanel({
  questionId,
  questionText,
  category,
  difficulty,
  companyName,
  roleName,
  resumeText,
}: AnswerPanelProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResult | null>(null);

  async function fetchAnswer() {
    setLoading(true);
    setError(null);
    try {
      const payload: AnswerRequest = {
        questionId,
        questionText,
        category,
        difficulty,
        companyName,
        roleName,
        resumeText,
      };
      const res = await fetch("/api/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to generate an answer.");
      }
      setResult((await res.json()) as AnswerResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function toggle() {
    const next = !open;
    setOpen(next);
    if (next && !result && !loading) fetchAnswer();
  }

  return (
    <div className="mt-2">
      <Button variant="ghost" onClick={toggle} className="px-2 py-1 text-xs">
        {open ? "Hide answer" : "Get answer"}
      </Button>

      {open && (
        <div className="mt-2 rounded-lg border border-slate-800 bg-slate-950/60 p-3">
          {loading && <p className="text-xs text-slate-500">Thinking this through...</p>}

          {error && (
            <div className="flex items-center gap-3">
              <p className="text-xs text-red-400">{error}</p>
              <Button variant="secondary" onClick={fetchAnswer} className="px-2 py-1 text-xs">
                Retry
              </Button>
            </div>
          )}

          {result && (
            <div className="space-y-3">
              <p className="whitespace-pre-wrap text-sm text-slate-200">{result.answer}</p>

              {result.diagram && <MermaidDiagram definition={result.diagram} />}

              {result.sources.length > 0 && (
                <div>
                  <p className="mb-1 text-xs font-semibold text-slate-400">Sources</p>
                  <ul className="space-y-1 text-xs">
                    {result.sources.map((source, i) => (
                      <li key={i}>
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-indigo-400 hover:underline"
                        >
                          {source.title}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <p className="text-[10px] uppercase tracking-wide text-slate-600">
                Generated with {result.model}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
