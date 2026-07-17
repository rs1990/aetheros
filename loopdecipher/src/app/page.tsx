"use client";

import { useState } from "react";
import InputForm from "@/components/InputForm";
import LoopTimeline from "@/components/LoopTimeline";
import QuestionBank from "@/components/QuestionBank";
import CultureDecoder from "@/components/CultureDecoder";
import PitchGenerator from "@/components/PitchGenerator";
import StudySchedule from "@/components/StudySchedule";
import AudioSandbox from "@/components/AudioSandbox";
import ResumeOptimizer from "@/components/ResumeOptimizer";
import type { DecipherRequest, DecipherResult } from "@/lib/types";

export default function Home() {
  const [result, setResult] = useState<DecipherResult | null>(null);
  const [lastRequest, setLastRequest] = useState<DecipherRequest | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(payload: DecipherRequest) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/decipher", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to generate study guide.");
      }
      const data = (await res.json()) as DecipherResult;
      setResult(data);
      setLastRequest(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-slate-100">LoopDecipher</h1>
        <p className="mt-2 text-slate-400">
          Paste a job posting. Get a personalized, forum-verified interview study guide.
        </p>
      </header>

      <InputForm onSubmit={handleSubmit} loading={loading} />

      {error && (
        <div className="mt-6 rounded-lg border border-red-800 bg-red-950/50 px-4 py-3 text-red-300">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-12 space-y-10">
          {result.mode === "mock" && (
            <div className="rounded-lg border border-amber-800 bg-amber-950/40 px-4 py-3 text-sm text-amber-300">
              Running in local-first mock mode (no API keys configured). Add
              ANTHROPIC_API_KEY to unlock live forum-verified synthesis.
            </div>
          )}
          <LoopTimeline schedule={result.studySchedule} />
          <div className="grid gap-10 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <QuestionBank
                questions={result.questions}
                companyName={lastRequest?.companyName}
                roleName={lastRequest?.roleName}
                resumeText={lastRequest?.resumeText}
              />
            </div>
            <div className="space-y-8">
              <CultureDecoder insights={result.cultureInsights} />
              <PitchGenerator mustKnowTech={result.mustKnowTech} />
            </div>
          </div>
          <StudySchedule schedule={result.studySchedule} />
          <AudioSandbox />
        </div>
      )}

      <div className="mt-12">
        <ResumeOptimizer
          initialResumeText={lastRequest?.resumeText}
          initialJobDescription={lastRequest?.jobDescription}
          initialCompanyName={lastRequest?.companyName}
          initialRoleName={lastRequest?.roleName}
        />
      </div>
    </main>
  );
}
