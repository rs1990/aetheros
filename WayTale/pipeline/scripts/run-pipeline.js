/**
 * Batch pipeline runner.
 * Runs the full content pipeline for all seed landmarks.
 * Safe to re-run — DB upserts and audio cache prevent duplicate work.
 *
 * Usage:
 *   node scripts/run-pipeline.js              # all 10 landmarks, no TTS
 *   node scripts/run-pipeline.js --tts        # with Amazon Polly
 *   node scripts/run-pipeline.js --id Q8733   # single landmark by Wikidata ID
 */

import 'dotenv/config';
import { runPipeline } from '../src/pipeline.js';
import { SEED_LANDMARKS } from './seed-landmarks.js';
import { db } from '../src/db/client.js';

const args = process.argv.slice(2);
const withTts   = args.includes('--tts');
const singleId  = args[args.indexOf('--id') + 1];
const skipTts   = !withTts;

const targets = singleId
  ? SEED_LANDMARKS.filter(l => l.id === singleId)
  : SEED_LANDMARKS;

if (!targets.length) {
  console.error(`No landmark found for --id ${singleId}`);
  process.exit(1);
}

console.log(`\nWayTale Phase 0 Pipeline`);
console.log(`Landmarks: ${targets.length} | TTS: ${withTts ? 'ON (Amazon Polly)' : 'OFF'}\n`);

const results = { ok: [], failed: [], no_content: [] };

for (const landmark of targets) {
  try {
    const result = await runPipeline(landmark.id, { skipTts });
    if (result.status === 'no_content') {
      results.no_content.push(landmark.name);
    } else {
      results.ok.push(landmark.name);
    }
  } catch (err) {
    console.error(`  ✗ FAILED ${landmark.name}: ${err.message}`);
    results.failed.push({ name: landmark.name, error: err.message });
  }
}

console.log('\n=== Pipeline Summary ===');
console.log(`✓ Success:     ${results.ok.length}`);
console.log(`⚠ No content: ${results.no_content.length}`);
console.log(`✗ Failed:     ${results.failed.length}`);
if (results.failed.length) {
  results.failed.forEach(f => console.log(`  - ${f.name}: ${f.error}`));
}

await db.end();
