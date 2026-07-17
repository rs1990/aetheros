/**
 * Admin CLI review tool.
 * Lists pending content, lets reviewer approve or reject each script.
 * Approved content gets status='reviewed' → ready for publish.
 *
 * Usage: node scripts/admin-review.js
 */

import 'dotenv/config';
import readline from 'readline';
import { db } from '../src/db/client.js';

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const ask = (q) => new Promise(r => rl.question(q, r));

const { rows: pending } = await db.query(`
  SELECT
    lc.id, lc.content_type, lc.variant, lc.fact_type, lc.status,
    lc.script, lc.sources, lc.audio_url,
    l.name AS landmark_name, l.wikidata_id
  FROM landmark_content lc
  JOIN landmarks l ON l.id = lc.landmark_id
  WHERE lc.status = 'pending'
  ORDER BY l.name, lc.content_type, lc.variant
`);

if (!pending.length) {
  console.log('No pending content to review.');
  rl.close();
  await db.end();
  process.exit(0);
}

console.log(`\n=== WayTale Admin Review ===`);
console.log(`${pending.length} items pending review.\n`);

for (const row of pending) {
  console.log('─'.repeat(60));
  console.log(`LANDMARK:     ${row.landmark_name} (${row.wikidata_id})`);
  console.log(`TYPE:         ${row.content_type} / ${row.variant ?? 'N/A'}`);
  console.log(`FACT TYPE:    ${row.fact_type}`);
  if (row.audio_url) console.log(`AUDIO:        ${row.audio_url}`);
  console.log('\nSCRIPT:');
  console.log(row.script);
  console.log('\nSOURCES:');
  const sources = JSON.parse(row.sources);
  sources.forEach((s, i) => console.log(`  [${i+1}] ${s.source}: ${s.url}`));

  const answer = await ask('\n[a]pprove / [r]eject / [s]kip? ').then(a => a.trim().toLowerCase());

  if (answer === 'a') {
    await db.query(`UPDATE landmark_content SET status = 'reviewed', updated_at = NOW() WHERE id = $1`, [row.id]);
    console.log('  ✓ Approved');
  } else if (answer === 'r') {
    const reason = await ask('  Rejection reason: ');
    await db.query(`UPDATE landmark_content SET status = 'rejected', updated_at = NOW() WHERE id = $1`, [row.id]);
    console.log(`  ✗ Rejected: ${reason}`);
  } else {
    console.log('  → Skipped');
  }
}

console.log('\n=== Review complete ===');
rl.close();
await db.end();
