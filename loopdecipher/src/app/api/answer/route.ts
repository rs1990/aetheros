import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { AnswerRequest, AnswerResult, AnswerSource } from "@/lib/types";

// Behavioral answers are synthesized from facts the candidate already gave us
// (their resume) — low hallucination risk, so the cheap model is enough.
const BEHAVIORAL_MODEL = "claude-haiku-4-5";
// Technical/Coding/System Design answers assert CS facts and tradeoffs — worth
// the mid-tier model plus optional web search grounding for anything the model
// isn't already confident about.
const TECHNICAL_MODEL = "claude-sonnet-5";

const DIAGRAM_INSTRUCTION =
  "Include a system architecture diagram as a Mermaid flowchart inside a fenced ```mermaid code block. Keep it to the components and data flow the answer actually discusses.";

const SYSTEM_PROMPT = `You are a precise technical interview coach helping a candidate prepare a real answer to one interview question.

Rules — accuracy over confidence:
- Never invent facts, benchmarks, version numbers, or citations. If you are not fully certain of a specific claim, say so explicitly instead of stating it as fact.
- For well-established computer science concepts you already know solidly (classic algorithms, standard data structures, textbook complexity analysis), answer directly — do not spend a search call confirming things you're already sure of.
- For claims that are company-specific, product-specific, benchmark-specific, or otherwise likely to have changed since your training — use the web_search tool before stating them as fact.
- End your response with a "Sources:" section. If you used web_search, list only the real URLs the tool actually returned — never fabricate a URL or title. If you did not need to search, write "Sources: none needed — general CS knowledge" instead.
- Keep the answer interview-realistic: a strong spoken answer, not an essay. Follow it with a short "Key concepts" section explaining the underlying ideas a candidate should understand, not just recite.`;

function pickModel(category: string): string {
  return category === "Behavioral" ? BEHAVIORAL_MODEL : TECHNICAL_MODEL;
}

function extractDiagram(text: string): { answer: string; diagram?: string } {
  const match = text.match(/```mermaid\n([\s\S]*?)```/);
  if (!match) return { answer: text.trim() };
  return {
    answer: (text.slice(0, match.index) + text.slice(match.index! + match[0].length)).trim(),
    diagram: match[1].trim(),
  };
}

export async function POST(request: Request) {
  const body = (await request.json()) as AnswerRequest;
  const { questionText, category, difficulty, companyName, roleName, resumeText } = body;

  if (!questionText || !category || !difficulty) {
    return NextResponse.json(
      { error: "questionText, category, and difficulty are required." },
      { status: 400 },
    );
  }

  if (!process.env.ANTHROPIC_API_KEY) {
    return NextResponse.json(
      { error: "Answer generation needs ANTHROPIC_API_KEY set — this endpoint has no offline mode." },
      { status: 503 },
    );
  }

  const model = pickModel(category);
  const wantsDiagram = category === "System Design";
  const usesSearch = category !== "Behavioral";

  try {
    const client = new Anthropic();
    const response = await client.messages.create({
      model,
      max_tokens: 3000,
      system: wantsDiagram ? `${SYSTEM_PROMPT}\n\n${DIAGRAM_INSTRUCTION}` : SYSTEM_PROMPT,
      tools: usesSearch
        ? [{ type: "web_search_20260209", name: "web_search", max_uses: 3 }]
        : undefined,
      messages: [
        {
          role: "user",
          content: `Question (${category}, ${difficulty}): ${questionText}
${companyName ? `Company: ${companyName}` : ""}
${roleName ? `Role: ${roleName}` : ""}
${
  resumeText
    ? `Candidate resume — ground any behavioral/experience claims in this and only this, do not invent experience the candidate doesn't have:\n${resumeText}`
    : "No resume provided — for behavioral questions, give a strong example answer structure instead of a fabricated personal story."
}`,
        },
      ],
    });

    let rawText = "";
    const sources: AnswerSource[] = [];

    for (const block of response.content) {
      if (block.type === "text") {
        rawText += block.text;
      } else if (block.type === "web_search_tool_result" && Array.isArray(block.content)) {
        for (const item of block.content) {
          sources.push({ title: item.title, url: item.url });
        }
      }
    }

    const { answer, diagram } = extractDiagram(rawText);
    const result: AnswerResult = { answer, sources, diagram, model };
    return NextResponse.json(result);
  } catch (error) {
    console.error("Answer generation failed:", error);
    return NextResponse.json(
      { error: "Failed to generate an answer. Try again in a moment." },
      { status: 502 },
    );
  }
}
