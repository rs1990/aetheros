/**
 * Core pipeline orchestrator.
 * For a given landmark (by wikidata_id):
 *   1. Fetch/upsert landmark metadata into DB
 *   2. Pull sourced facts from Wikipedia
 *   3. Draft narration scripts via LLM (Claude)
 *   4. Synthesize audio via Amazon Polly
 *   5. Store all content in DB with status='pending' (awaiting editorial review)
 */

import 'dotenv/config';
import { db } from './db/client.js';
import { fetchLandmarkByWikidataId } from './sources/wikidata.js';
import { fetchPageSummary, fetchPageSections, detectFactType } from './sources/wikipedia.js';
import { draftNarrationScripts } from './llm/draft-narration.js';
import { synthesizeAllVariants } from './tts/polly.js';

export async function runPipeline(wikidataId, { skipTts = false, premium = false } = {}) {
  console.log(`\n=== Pipeline: ${wikidataId} ===`);

  // 1. Fetch Wikidata metadata
  console.log('  [1/5] Fetching Wikidata metadata...');
  const [wikidataRow] = await fetchLandmarkByWikidataId(wikidataId);
  if (!wikidataRow) throw new Error(`Wikidata ID not found: ${wikidataId}`);

  // 2. Upsert landmark record
  console.log('  [2/5] Upserting landmark...');
  const { rows: [landmark] } = await db.query(`
    INSERT INTO landmarks (wikidata_id, name, description, latitude, longitude, wikipedia_en, image_url, country_code, language)
    VALUES ($1, $2, $3, $4, $5, $6, $7, 'US', 'en')
    ON CONFLICT (wikidata_id) DO UPDATE SET
      name = EXCLUDED.name,
      description = EXCLUDED.description,
      wikipedia_en = EXCLUDED.wikipedia_en,
      updated_at = NOW()
    RETURNING *
  `, [
    wikidataRow.wikidata_id,
    wikidataRow.name,
    wikidataRow.description,
    wikidataRow.latitude,
    wikidataRow.longitude,
    wikidataRow.wikipedia_en,
    wikidataRow.image_url,
  ]);
  console.log(`  Landmark: ${landmark.name} (${landmark.id})`);

  // 3. Pull facts from Wikipedia
  console.log('  [3/5] Fetching Wikipedia content...');
  const sources = [];
  let factText = '';

  if (landmark.wikipedia_en) {
    const summary = await fetchPageSummary(landmark.wikipedia_en);
    const sections = await fetchPageSections(landmark.wikipedia_en);

    if (summary) {
      const sourceUrl = summary.source_url;
      factText += `SUMMARY:\n${summary.extract}\n\n`;
      sources.push({ source: 'wikipedia', source_url: sourceUrl, raw_text: summary.extract, fact_type: 'verified' });
    }

    if (sections?.extract) {
      factText += `DETAILED EXTRACT:\n${sections.extract}\n\n`;
      const ft = detectFactType(sections.extract);
      sources.push({ source: 'wikipedia', source_url: sections.source_url, raw_text: sections.extract, fact_type: ft });
    }
  } else {
    factText = wikidataRow.description ?? 'No Wikipedia article found.';
    sources.push({ source: 'wikidata', source_url: `https://www.wikidata.org/wiki/${wikidataId}`, raw_text: factText, fact_type: 'verified' });
  }

  // Store raw facts in DB
  for (const f of sources) {
    await db.query(`
      INSERT INTO landmark_facts (landmark_id, source, source_url, raw_text, fact_type)
      VALUES ($1, $2, $3, $4, $5)
      ON CONFLICT DO NOTHING
    `, [landmark.id, f.source, f.source_url, f.raw_text, f.fact_type]);
  }

  // 4. Draft narration via LLM
  console.log('  [4/5] Drafting narration scripts (LLM)...');
  const scripts = await draftNarrationScripts({ landmark, facts: factText, sources });
  if (!scripts) {
    console.log('  ⚠ Insufficient content — marking landmark as no_content');
    return { landmark, status: 'no_content' };
  }

  // 5. Synthesize audio (optional — skip if no AWS creds configured)
  let audioMap = {};
  if (!skipTts && process.env.AWS_ACCESS_KEY_ID) {
    console.log('  [5/5] Synthesizing audio (Amazon Polly)...');
    audioMap = await synthesizeAllVariants({ landmarkId: landmark.id, scripts, premium });
  } else {
    console.log('  [5/5] TTS skipped (no AWS creds or skipTts=true)');
  }

  // 6. Store content records in DB
  const contentMap = {
    ambient:              { type: 'ambient',    variant: null,        length: 'short', script: scripts.ambient_short },
    deep_dive_history:    { type: 'deep_dive',  variant: 'history',   length: 'long',  script: scripts.deep_dive_history },
    deep_dive_geography:  { type: 'deep_dive',  variant: 'geography', length: 'long',  script: scripts.deep_dive_geography },
    deep_dive_culture:    { type: 'deep_dive',  variant: 'culture',   length: 'long',  script: scripts.deep_dive_culture },
  };

  for (const [key, meta] of Object.entries(contentMap)) {
    if (!meta.script) continue;
    const audioUrl = audioMap[key === 'ambient' ? 'ambient_short' : key] ?? null;

    await db.query(`
      INSERT INTO landmark_content
        (landmark_id, content_type, variant, length, script, fact_type, sources, audio_url, tts_provider, language, region, status)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'en', 'US', 'pending')
      ON CONFLICT DO NOTHING
    `, [
      landmark.id,
      meta.type,
      meta.variant,
      meta.length,
      meta.script,
      scripts.fact_type,
      JSON.stringify(sources.map(s => ({ source: s.source, url: s.source_url }))),
      audioUrl,
      audioUrl ? 'polly' : null,
    ]);
  }

  console.log(`  ✓ Pipeline complete: ${landmark.name}`);
  return { landmark, scripts, audioMap, status: 'pending_review' };
}
