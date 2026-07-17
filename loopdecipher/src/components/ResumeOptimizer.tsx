"use client";

import { useState } from "react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { Input, Textarea } from "@/components/ui/Input";
import { parseResumeFile } from "@/lib/parseResumeFile";
import type { AtsRequest, AtsResult, AtsVerdict } from "@/lib/types";

const VERDICT_STYLE: Record<AtsVerdict, string> = {
  strong: "text-emerald-400 border-emerald-800 bg-emerald-950/40",
  moderate: "text-amber-400 border-amber-800 bg-amber-950/40",
  weak: "text-red-400 border-red-800 bg-red-950/40",
};

const SCORE_RING_COLOR: Record<AtsVerdict, string> = {
  strong: "stroke-emerald-400",
  moderate: "stroke-amber-400",
  weak: "stroke-red-400",
};

function ScoreRing({ score, verdict }: { score: number; verdict: AtsVerdict }) {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  return (
    <svg width="110" height="110" viewBox="0 0 110 110" className="shrink-0">
      <circle cx="55" cy="55" r={radius} className="fill-none stroke-slate-800" strokeWidth="10" />
      <circle
        cx="55"
        cy="55"
        r={radius}
        className={`fill-none ${SCORE_RING_COLOR[verdict]}`}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        transform="rotate(-90 55 55)"
      />
      <text x="55" y="61" textAnchor="middle" className="fill-slate-100 text-2xl font-bold">
        {score}
      </text>
    </svg>
  );
}

export default function ResumeOptimizer({
  initialResumeText,
  initialJobDescription,
  initialCompanyName,
  initialRoleName,
}: {
  initialResumeText?: string;
  initialJobDescription?: string;
  initialCompanyName?: string;
  initialRoleName?: string;
}) {
  const [resumeText, setResumeText] = useState(initialResumeText ?? "");
  const [jobDescription, setJobDescription] = useState(initialJobDescription ?? "");
  const [jobUrl, setJobUrl] = useState("");
  const [companyName, setCompanyName] = useState(initialCompanyName ?? "");
  const [roleName, setRoleName] = useState(initialRoleName ?? "");
  const [resumeFileName, setResumeFileName] = useState<string | null>(null);
  const [result, setResult] = useState<AtsResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleResumeFile(file: File) {
    setResumeFileName(`Parsing ${file.name}...`);
    try {
      const text = await parseResumeFile(file);
      setResumeText(text);
      setResumeFileName(file.name);
    } catch (err) {
      setResumeFileName(`${file.name} — ${err instanceof Error ? err.message : "failed to parse"}`);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!resumeText.trim()) {
      setError("Paste your resume text or upload a .txt file.");
      return;
    }
    if (!jobDescription.trim() && !jobUrl.trim()) {
      setError("Paste the job description or enter a URL to it.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const payload: AtsRequest = {
        resumeText: resumeText.trim(),
        jobDescription: jobDescription.trim() || undefined,
        jobUrl: jobDescription.trim() ? undefined : jobUrl.trim(),
        companyName: companyName.trim() || undefined,
        roleName: roleName.trim() || undefined,
      };
      const res = await fetch("/api/ats", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Failed to score the resume.");
      }
      setResult((await res.json()) as AtsResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <h2 className="mb-1 text-lg font-semibold text-slate-100">Resume &amp; ATS Match Checker</h2>
      <p className="mb-4 text-sm text-slate-400">
        See how your resume scores against a job posting, what an ATS parser would flag, and how to
        rewrite weak bullets.
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <Input
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            placeholder="Company (optional)"
          />
          <Input
            value={roleName}
            onChange={(e) => setRoleName(e.target.value)}
            placeholder="Role (optional)"
          />
        </div>

        <div>
          <label className="mb-1 block text-sm text-slate-400">Job Posting URL (optional)</label>
          <Input
            type="url"
            value={jobUrl}
            onChange={(e) => setJobUrl(e.target.value)}
            placeholder="https://company.com/careers/senior-backend-engineer"
            disabled={jobDescription.trim().length > 0}
          />
        </div>

        <div>
          <label className="mb-1 block text-sm text-slate-400">
            Job Description {jobUrl.trim() ? "(leave blank to use the URL above)" : ""}
          </label>
          <Textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the full job posting text here..."
            rows={6}
          />
        </div>

        <div>
          <label className="mb-1 block text-sm text-slate-400">Resume (paste text or upload .txt, .pdf, or .docx)</label>
          <Textarea
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
            placeholder="Paste your resume text here..."
            rows={8}
          />
          <div className="mt-2 flex items-center gap-3">
            <input
              type="file"
              accept=".txt,.pdf,.docx,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={(e) => e.target.files?.[0] && handleResumeFile(e.target.files[0])}
              className="text-xs text-slate-400 file:mr-3 file:rounded-md file:border-0 file:bg-slate-800 file:px-3 file:py-1.5 file:text-slate-200 hover:file:bg-slate-700"
            />
            {resumeFileName && <span className="text-xs text-slate-500">{resumeFileName}</span>}
          </div>
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <Button type="submit" disabled={loading} className="w-full">
          {loading ? "Scoring your resume..." : "Check ATS Match"}
        </Button>
      </form>

      {result && (
        <div className="mt-8 space-y-6 border-t border-slate-800 pt-6">
          {result.mode === "mock" && (
            <div className="rounded-lg border border-amber-800 bg-amber-950/40 px-4 py-3 text-sm text-amber-300">
              Running in local-first mock mode (keyword overlap, no API keys configured). Add
              ANTHROPIC_API_KEY for a real semantic ATS review.
            </div>
          )}

          <div className="flex items-center gap-5">
            <ScoreRing score={result.matchScore} verdict={result.verdict} />
            <div>
              <span
                className={`inline-block rounded-full border px-2.5 py-0.5 text-xs font-medium uppercase tracking-wide ${VERDICT_STYLE[result.verdict]}`}
              >
                {result.verdict} match
              </span>
              <p className="mt-2 text-sm text-slate-300">{result.summary}</p>
            </div>
          </div>

          <div className="grid gap-6 sm:grid-cols-2">
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-300">Matched Keywords</h3>
              <div className="flex flex-wrap gap-1.5">
                {result.matchedKeywords.length === 0 && (
                  <span className="text-sm text-slate-500">None found.</span>
                )}
                {result.matchedKeywords.map((k) => (
                  <span
                    key={k}
                    className="rounded-md border border-emerald-800 bg-emerald-950/30 px-2 py-0.5 text-xs text-emerald-300"
                  >
                    {k}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-300">Missing Keywords</h3>
              <div className="flex flex-wrap gap-1.5">
                {result.missingKeywords.length === 0 && (
                  <span className="text-sm text-slate-500">None — great coverage.</span>
                )}
                {result.missingKeywords.map((k) => (
                  <span
                    key={k}
                    className="rounded-md border border-red-800 bg-red-950/30 px-2 py-0.5 text-xs text-red-300"
                  >
                    {k}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {result.formattingWarnings.length > 0 && (
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-300">ATS Formatting Warnings</h3>
              <ul className="space-y-1.5 text-sm text-slate-400">
                {result.formattingWarnings.map((w, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-amber-400">!</span>
                    {w}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.bulletRewrites.length > 0 && (
            <div>
              <h3 className="mb-2 text-sm font-semibold text-slate-300">Suggested Bullet Rewrites</h3>
              <div className="space-y-3">
                {result.bulletRewrites.map((b, i) => (
                  <div key={i} className="rounded-lg border border-slate-800 bg-slate-900/60 p-3 text-sm">
                    <p className="text-slate-500 line-through">{b.original}</p>
                    <p className="mt-1 text-slate-100">{b.rewritten}</p>
                    <p className="mt-1 text-xs text-slate-500">{b.reason}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
