"use client";

import { useState } from "react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import { Input, Textarea } from "@/components/ui/Input";
import type { DecipherRequest } from "@/lib/types";

interface InputFormProps {
  onSubmit: (payload: DecipherRequest) => void;
  loading: boolean;
}

export default function InputForm({ onSubmit, loading }: InputFormProps) {
  const [companyName, setCompanyName] = useState("");
  const [roleName, setRoleName] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [resumeText, setResumeText] = useState("");
  const [resumeFileName, setResumeFileName] = useState<string | null>(null);

  async function handleResumeFile(file: File) {
    if (!file.type.startsWith("text/") && !file.name.endsWith(".txt")) {
      setResumeFileName(`${file.name} (only .txt parsing supported — paste text instead for PDFs)`);
      return;
    }
    const text = await file.text();
    setResumeText(text);
    setResumeFileName(file.name);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({ companyName, roleName, jobDescription, resumeText: resumeText || undefined });
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
          <label className="mb-1 block text-sm text-slate-400">Job Description</label>
          <Textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            placeholder="Paste the full job posting text here..."
            rows={8}
            required
          />
        </div>

        <div>
          <label className="mb-1 block text-sm text-slate-400">
            Resume (optional — paste text or upload .txt)
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
              accept=".txt,text/plain"
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
