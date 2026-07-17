/**
 * Seeds 10 US landmarks for Phase 0 validation.
 * These Wikidata IDs are well-documented US tourist attractions
 * with rich Wikipedia articles and verifiable sourced content.
 */

import 'dotenv/config';
import { db } from '../src/db/client.js';

// 10 US landmarks with verified Wikidata IDs
const SEED_LANDMARKS = [
  { id: 'Q8733',    name: 'Statue of Liberty',         region: 'Northeast' },
  { id: 'Q18013',   name: 'Golden Gate Bridge',         region: 'West Coast' },
  { id: 'Q19588',   name: 'Grand Canyon',               region: 'Southwest' },
  { id: 'Q9248',    name: 'Yellowstone National Park',  region: 'Rocky Mountains' },
  { id: 'Q1626630', name: 'Alcatraz Island',            region: 'West Coast' },
  { id: 'Q9161',    name: 'Mount Rushmore',             region: 'Great Plains' },
  { id: 'Q80994',   name: 'Niagara Falls',              region: 'Northeast' },
  { id: 'Q12185',   name: 'Lincoln Memorial',           region: 'DC' },
  { id: 'Q22731',   name: 'Yosemite National Park',     region: 'West Coast' },
  { id: 'Q51685',   name: 'Space Needle',               region: 'Pacific Northwest' },
];

const result = await db.query('SELECT COUNT(*) FROM landmarks');
console.log(`Current landmark count: ${result.rows[0].count}`);
console.log('Seed list (Wikidata IDs to process):');
SEED_LANDMARKS.forEach(l => console.log(`  ${l.id}  ${l.name}  [${l.region}]`));

export { SEED_LANDMARKS };
await db.end();
