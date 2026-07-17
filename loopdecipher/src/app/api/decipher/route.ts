import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { scrapeForumSnippets } from "@/lib/scraper";
import { buildMockResult } from "@/lib/mockEngine";
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
  },
  required: ["questions", "cultureInsights", "mustKnowTech", "studySchedule"],
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

function isMockMode(): boolean {
  return (
    process.env.NEXT_PUBLIC_USE_MOCK_MODE === "true" || !process.env.ANTHROPIC_API_KEY
  );
}

export async function POST(request: Request) {
  const body = (await request.json()) as DecipherRequest;
  const { jobDescription, companyName, roleName, resumeText } = body;

  if (!jobDescription || !companyName || !roleName) {
    return NextResponse.json(
      { error: "jobDescription, companyName, and roleName are required." },
      { status: 400 },
    );
  }

  const mustKnowTech = extractMustKnowTech(jobDescription);

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
      system: `You are an expert technical interview coach. You merge three data layers into a single personalized study guide:
1. The job description — extract must-know technologies and domains.
2. Real forum-sourced interview reports — when a snippet describes an actual asked question or theme (e.g. "they asked me to build an in-memory database"), generate a close variation of that exact challenge and mark it source: "forum-sourced" with sourceDetail citing the snippet's source/url.
3. The candidate's resume — identify skill gaps relative to the role and weight questions toward those gaps.

Generate approximately 100 questions spanning Behavioral, Coding, System Design, and Technical categories, with a realistic Easy/Medium/Hard spread. Only mark a question "forum-sourced" if it is a direct variation of something in the provided forum snippets — everything else is "predicted". Also produce 3-6 culture insights (from forum tone/content, or general reasonable inference if no forum data), and a 4-6 week study schedule.`,
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
    const result: DecipherResult = { ...parsed, mode: "live" };
    return NextResponse.json(result);
  } catch (error) {
    console.error("Decipher synthesis failed, falling back to mock mode:", error);
    return NextResponse.json(buildMockResult(mustKnowTech));
  }
}
