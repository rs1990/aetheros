import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { fetchJobDescriptionFromUrl } from "@/lib/fetchJobPosting";
import { buildMockAtsResult } from "@/lib/mockEngine";
import { isMockMode } from "@/lib/mode";
import type { AtsRequest, AtsResult } from "@/lib/types";

const MODEL = process.env.ANTHROPIC_MODEL_ATS || "claude-sonnet-5";

const RESULT_SCHEMA = {
  type: "object",
  properties: {
    matchScore: { type: "integer" },
    verdict: { type: "string", enum: ["strong", "moderate", "weak"] },
    matchedKeywords: { type: "array", items: { type: "string" } },
    missingKeywords: { type: "array", items: { type: "string" } },
    formattingWarnings: { type: "array", items: { type: "string" } },
    bulletRewrites: {
      type: "array",
      items: {
        type: "object",
        properties: {
          original: { type: "string" },
          rewritten: { type: "string" },
          reason: { type: "string" },
        },
        required: ["original", "rewritten", "reason"],
        additionalProperties: false,
      },
    },
    summary: { type: "string" },
  },
  required: [
    "matchScore",
    "verdict",
    "matchedKeywords",
    "missingKeywords",
    "formattingWarnings",
    "bulletRewrites",
    "summary",
  ],
  additionalProperties: false,
} as const;

export async function POST(request: Request) {
  const body = (await request.json()) as AtsRequest;
  const { resumeText, jobUrl, companyName, roleName } = body;
  let jobDescription = body.jobDescription;

  if (!resumeText || (!jobDescription && !jobUrl)) {
    return NextResponse.json(
      { error: "resumeText and either jobDescription or jobUrl are required." },
      { status: 400 },
    );
  }

  if (!jobDescription && jobUrl) {
    try {
      jobDescription = await fetchJobDescriptionFromUrl(jobUrl);
    } catch (err) {
      return NextResponse.json(
        { error: err instanceof Error ? err.message : "Failed to fetch the job posting URL." },
        { status: 422 },
      );
    }
  }

  if (isMockMode()) {
    const mock = buildMockAtsResult(resumeText, jobDescription!);
    return NextResponse.json({ ...mock, mode: "mock" } satisfies AtsResult);
  }

  try {
    const client = new Anthropic();
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 4000,
      output_config: { format: { type: "json_schema", schema: RESULT_SCHEMA } },
      system: `You are an ATS (Applicant Tracking System) resume screener and resume coach. Given a resume and a job description:

1. Score how well the resume matches the job description as an ATS would: keyword/skill overlap, title alignment, seniority signal. matchScore is 0-100.
2. verdict: "strong" (>=70), "moderate" (40-69), "weak" (<40) — must be consistent with matchScore.
3. List matchedKeywords actually present in the resume (skills, tools, domains) and missingKeywords the job description emphasizes but the resume doesn't demonstrate. Keep these to real terms from the job description, not invented ones.
4. formattingWarnings: flag things a real ATS parser could choke on (tables, columns, graphics-only contact info, missing dates, no section headers, non-standard fonts implied by odd spacing) — only flag what the text actually suggests, don't invent generic advice.
5. bulletRewrites: pick up to 5 of the resume's weakest existing bullet points and rewrite each to lead with a strong action verb, include the job description's relevant keywords where truthful, and push toward quantifying impact — never invent numbers or experience the candidate didn't state. Include a one-line reason for each rewrite.
6. summary: 2-3 sentences, direct, telling the candidate exactly what to fix first.

Never fabricate resume content, employers, dates, or metrics. Ground every claim in the text provided.`,
      messages: [
        {
          role: "user",
          content: `${companyName ? `Company: ${companyName}\n` : ""}${roleName ? `Role: ${roleName}\n` : ""}
Job Description:
${jobDescription}

Resume:
${resumeText}`,
        },
      ],
    });

    const textBlock = response.content.find((b) => b.type === "text");
    if (!textBlock || textBlock.type !== "text") {
      throw new Error("No text content in Claude response");
    }

    const parsed = JSON.parse(textBlock.text) as Omit<AtsResult, "mode">;
    const result: AtsResult = { ...parsed, mode: "live" };
    return NextResponse.json(result);
  } catch (error) {
    console.error("ATS scoring failed, falling back to mock mode:", error);
    const mock = buildMockAtsResult(resumeText, jobDescription!);
    return NextResponse.json({ ...mock, mode: "mock" } satisfies AtsResult);
  }
}
