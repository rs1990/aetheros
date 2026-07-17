"use client";

import { useState } from "react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { Input, Textarea } from "@/components/ui/Input";
import { parseResumeFile } from "@/lib/parseResumeFile";
import type { DecipherRequest } from "@/lib/types";

interface InputFormProps {
  onSubmit: (payload: DecipherRequest) => void;
  loading: boolean;
}

export default function InputForm({ onSubmit, loading }: InputFormProps) {
  const [companyName, setCompanyName] = useState("");
  const [roleName, setRoleName] = useState("");
  const [jobUrl, setJobUrl] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [resumeText, setResumeText] = useState("");
  const [resumeFileName, setResumeFileName] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

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

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!jobDescription.trim() && !jobUrl.trim()) {
      setValidationError("Paste the job description or enter a URL to it.");
      return;
    }
    setValidationError(null);
    onSubmit({
      companyName,
      roleName,
      jobDescription: jobDescription.trim() || undefined,
      jobUrl: jobDescription.trim() ? undefined : jobUrl.trim(),
      resumeText: resumeText || undefined,
    });
  }

  return (
    <Card>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="mb-1 block text-sm text-slate-400">Company</label>
            <Input
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="e.g. Stripe"
              required
            />
          </div>
          <div>
            <label className="mb-1 block text-sm text-slate-400">Role</label>
            <Input
              value={roleName}
              onChange={(e) => setRoleName(e.target.value)}
              placeholder="e.g. Senior Backend Engineer"
              required
            />
          </div>
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
          <p className="mt-1 text-xs text-slate-500">
            Some sites block automated fetching — if it fails, paste the text below instead.
          </p>
        </div>

        <div>
          <label className="mb-1 block text-sm text-slate-400">
            Job Description {jobUrl.trim() ? "(leave blank to use the URL above)" : ""}
          </label>
          <Textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the full job posting text here..."
            rows={8}
          />
        </div>

        {validationError && <p className="text-sm text-red-400">{validationError}</p>}

        <div>
          <label className="mb-1 block text-sm text-slate-400">
            Resume (optional — paste text or upload .txt, .pdf, or .docx)
          </label>
          <Textarea
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
            placeholder="Paste your resume text here..."
            rows={4}
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

        <Button type="submit" disabled={loading} className="w-full">
          {loading ? "Deciphering the loop..." : "Generate Study Guide"}
        </Button>
      </form>
    </Card>
  );
}
