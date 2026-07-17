/**
 * LLM narration script drafter.
 * Input:  sourced facts pulled from Wikipedia/Wikidata (never free-generated trivia).
 * Output: a narration script tagged with fact_type and source citations.
 *
 * The LLM is instructed ONLY to rewrite/structure the provided facts — it must
 * not invent dates, names, or events. If no verified content is available, it
 * returns a signal so the pipeline can mark the landmark as "no content".
 */

import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const SYSTEM_PROMPT = `You are a professional audio tour script writer for WayTale.

Your job: rewrite the provided sourced facts into engaging, natural narration scripts for travelers.

STRICT RULES:
1. Use ONLY facts from the provided source material. Never add dates, names, or events not in the sources.
2. If a fact involves legend, folklore, or unverified claims, you MUST label it clearly (e.g., "According to local legend..." or "It is said that...").
3. If you cannot write a meaningful script from the provided facts, respond with JSON: {"error": "insufficient_content"}
4. Keep verified facts and legends clearly separated in the script.
5. No marketing language, superlatives, or vague claims ("one of the most amazing...").

OUTPUT FORMAT (JSON only, no markdown wrapper):
{
  "ambient_short": "60-90 second script for passive listening while passing the landmark",
  "deep_dive_history": "3-5 minute deep-dive focused on historical facts",
  "deep_dive_geography": "3-5 minute deep-dive focused on geography/terrain/why this location matters",
  "deep_dive_culture": "3-5 minute deep-dive focused on cultural/social context",
  "fact_type": "verified" | "legend" | "mixed",
  "legend_notes": "descriptions of any legend/folklore content flagged in the scripts (empty string if none)"
}`;

export async function draftNarrationScripts({ landmark, facts, sources }) {
  const userContent = `
LANDMARK: ${landmark.name}
LOCATION: ${landmark.latitude}, ${landmark.longitude}
WIKIDATA: ${landmark.wikidata_id}

SOURCED FACTS (use ONLY these — do not add external knowledge):
${facts}

SOURCE CITATIONS:
${sources.map((s, i) => `[${i + 1}] ${s.source}: ${s.source_url}`).join('\n')}
`.trim();

  const message = await client.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 4096,
    system: SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userContent }],
  });

  const raw = message.content[0].text.trim();

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`LLM returned non-JSON: ${raw.slice(0, 200)}`);
  }

  if (parsed.error === 'insufficient_content') {
    return null;
  }

  return {
    ambient_short:        parsed.ambient_short,
    deep_dive_history:    parsed.deep_dive_history,
    deep_dive_geography:  parsed.deep_dive_geography,
    deep_dive_culture:    parsed.deep_dive_culture,
    fact_type:            parsed.fact_type,
    legend_notes:         parsed.legend_notes ?? '',
  };
}
