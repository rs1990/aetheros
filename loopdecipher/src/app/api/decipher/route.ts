import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { scrapeForumSnippets } from "@/lib/scraper";
import { fetchJobDescriptionFromUrl } from "@/lib/fetchJobPosting";
import { buildMockResult } from "@/lib/mockEngine";
import { isMockMode } from "@/lib/mode";
import type { DecipherRequest, DecipherResult } from "@/lib/types";

const MODEL = process.env.ANTHROPIC_MODEL || "claude-opus-4-8";

const RESULT_SCHEMA = {
  type: "object",
  properties: {
    questions: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          text: { type: "string" },
          category: {
            type: "string",
            enum: ["Technical", "System Design", "Coding", "Behavioral"],
          },
          difficulty: { type: "string", enum: ["Easy", "Medium", "Hard"] },
          source: { type: "string", enum: ["predicted", "forum-sourced"] },
          sourceDetail: { type: "string" },
        },
        required: ["id", "text", "category", "difficulty", "source"],
        additionalProperties: false,
      },
    },
    cultureInsights: {
      type: "array",
      items: {
        type: "object",
        properties: {
          insight: { type: "string" },
          source: { type: "string" },
          sentiment: { type: "string", enum: ["positive", "negative", "neutral"] },
        },
        required: ["insight", "source", "sentiment"],
        additionalProperties: false,
      },
    },
    mustKnowTech: { type: "array", items: { type: "string" } },
    studySchedule: {
      type: "array",
      items: {
        type: "object",
        properties: {
          week: { type: "integer" },
          focus: { type: "string" },
          tasks: { type: "array", items: { type: "string" } },
        },
        required: ["week", "focus", "tasks"],
        additionalProperties: false,
      },
    },
    interviewLoop: {
      type: "array",
      items: {
        type: "object",
        properties: {
          order: { type: "integer" },
          name: { type: "string" },
          format: { type: "string" },
          focus: { type: "array", items: { type: "string" } },
          sampleQuestionIds: { type: "array", items: { type: "string" } },
          source: { type: "string", enum: ["forum-sourced", "typical"] },
          sourceDetail: { type: "string" },
        },
        required: ["order", "name", "format", "focus", "sampleQuestionIds", "source"],
        additionalProperties: false,
      },
    },
  },
  required: ["questions", "cultureInsights", "mustKnowTech", "studySchedule", "interviewLoop"],
  additionalProperties: false,
} as const;

function extractMustKnowTech(jobDescription: string): string[] {
  const KNOWN_TECH = [
    "Kafka", "Go", "React", "TypeScript", "Python", "Java", "Kubernetes",
    "Docker", "AWS", "GCP", "Azure", "PostgreSQL", "MySQL", "MongoDB",
    "Redis", "GraphQL", "gRPC", "Spark", "Distributed Systems", "Microservices",
    "Node.js", "Rust", "C++", "SQL", "Terraform",
  ];
  const found = KNOWN_TECH.filter((tech) =>
    new RegExp(`\\b${tech.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(jobDescription),
  );
  return found.length > 0 ? found : ["Core CS Fundamentals", "System Design"];
}

export async function POST(request: Request) {
  const body = (await request.json()) as DecipherRequest;
  const { jobUrl, companyName, roleName, resumeText } = body;
  let jobDescription = body.jobDescription;

  if (!companyName || !roleName || (!jobDescription && !jobUrl)) {
    return NextResponse.json(
      { error: "companyName, roleName, and either jobDescription or jobUrl are required." },
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

  const mustKnowTech = extractMustKnowTech(jobDescription!);

  if (isMockMode()) {
    return NextResponse.json(buildMockResult(mustKnowTech));
  }

  try {
    const forumSnippets = await scrapeForumSnippets(companyName, roleName);

    const client = new Anthropic();
    const forumContext = forumSnippets.length
      ? forumSnippets
          .map((s, i) => `[${i + 1}] (${s.source}, ${s.url})\n${s.text}`)
          .join("\n\n")
      : "No forum-sourced snippets were found for this company/role.";

    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 16000,
      output_config: { format: { type: "json_schema", schema: RESULT_SCHEMA } },
      system: `You are an expert technical interview coach. Be objective and direct, not agreeable — don't inflate a candidate's readiness, don't soften a skill gap, and don't present a guess as verified fact. You merge three data layers into a single personalized study guide:
1. The job description — extract must-know technologies and domains.
2. Real forum-sourced interview reports — when a snippet describes an actual asked question or theme (e.g. "they asked me to build an in-memory database"), generate a close variation of that exact challenge and mark it source: "forum-sourced" with sourceDetail citing the snippet's source/url.
3. The candidate's resume — identify skill gaps relative to the role and weight questions toward those gaps.

Generate approximately 100 questions spanning Behavioral, Coding, System Design, and Technical categories, with a realistic Easy/Medium/Hard spread. Only mark a question "forum-sourced" if it is a direct variation of something in the provided forum snippets — everything else is "predicted". Also produce 3-6 culture insights (from forum tone/content, or general reasonable inference if no forum data), and a 4-6 week study schedule.

Also produce interviewLoop: the actual sequence of interview rounds a candidate for this role at this company should expect, ordered 1..N.
- If the forum snippets describe the company's real process (e.g. "phone screen then 5 onsite rounds: 2 coding, 1 system design, 1 behavioral, 1 bar raiser"), reconstruct that loop faithfully and mark each round source: "forum-sourced" with sourceDetail citing which snippet it came from. Do not embellish beyond what the snippets actually describe.
- If no forum data describes the process, build a realistic loop from the role level and domain implied by the job description (e.g. a senior IC backend role typically means recruiter screen, technical phone screen, then an onsite with coding, system design, and behavioral rounds; an entry-level role is usually shorter). Mark these rounds source: "typical" — do not claim company-specific knowledge you don't have.
- Each round needs: a name (e.g. "Round 2: Technical Phone Screen"), a format (duration, interviewer count, medium — coding platform, whiteboard, take-home, etc., stated plainly, not invented with false precision if unknown), 1-3 focus areas, and sampleQuestionIds referencing 1-4 ids from the questions array that best represent what that round would actually cover. Every id in sampleQuestionIds MUST be an id that literally appears in the questions array you generated in this same response — never invent an id that isn't there.`,
      messages: [
        {
          role: "user",
          content: `Company: ${companyName}
Role: ${roleName}

Job Description:
${jobDescription}

Forum-Sourced Interview Reports:
${forumContext}

${resumeText ? `Candidate Resume:\n${resumeText}` : "No resume provided — skip gap analysis, weight toward the job description only."}`,
        },
      ],
    });

    const textBlock = response.content.find((b) => b.type === "text");
    if (!textBlock || textBlock.type !== "text") {
      throw new Error("No text content in Claude response");
    }

    const parsed = JSON.parse(textBlock.text) as Omit<DecipherResult, "mode">;
    const validQuestionIds = new Set(parsed.questions.map((q) => q.id));
    parsed.interviewLoop = parsed.interviewLoop.map((round) => ({
      ...round,
      sampleQuestionIds: round.sampleQuestionIds.filter((id) => validQuestionIds.has(id)),
    }));
    const result: DecipherResult = { ...parsed, mode: "live" };
    return NextResponse.json(result);
  } catch (error) {
    console.error("Decipher synthesis failed, falling back to mock mode:", error);
    return NextResponse.json(buildMockResult(mustKnowTech));
  }
}
