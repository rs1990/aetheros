/**
 * Amazon Polly TTS client.
 * Converts narration scripts to audio files.
 * Runs ONCE per script — output stored in audio_cache/ (or S3).
 * TTS quality tiered: high-traffic landmarks get Neural engine, long-tail gets Standard.
 */

import {
  PollyClient,
  SynthesizeSpeechCommand,
  Engine,
  OutputFormat,
  VoiceId,
} from '@aws-sdk/client-polly';
import { createWriteStream, mkdirSync, existsSync } from 'fs';
import { pipeline } from 'stream/promises';
import path from 'path';

const polly = new PollyClient({ region: process.env.AWS_REGION ?? 'us-east-1' });

const AUDIO_DIR = process.env.AUDIO_OUTPUT_DIR ?? './audio_cache';

if (!existsSync(AUDIO_DIR)) mkdirSync(AUDIO_DIR, { recursive: true });

/**
 * @param {object} opts
 * @param {string} opts.landmarkId   - UUID of the landmark
 * @param {string} opts.contentType  - 'ambient' | 'deep_dive_history' | etc.
 * @param {string} opts.script       - plain text narration script
 * @param {boolean} [opts.premium]   - use Neural engine (more expensive)
 * @returns {Promise<string>} local file path (or CDN URL if using S3)
 */
export async function synthesizeAudio({ landmarkId, contentType, script, premium = false }) {
  const filename = `${landmarkId}_${contentType}.mp3`;
  const filePath = path.join(AUDIO_DIR, filename);

  if (existsSync(filePath)) {
    console.log(`  [polly] cache hit: ${filename}`);
    return filePath;
  }

  const command = new SynthesizeSpeechCommand({
    Text: script,
    TextType: 'text',
    OutputFormat: OutputFormat.MP3,
    VoiceId: VoiceId.Matthew,  // US English male; swap to Joanna for female
    Engine: premium ? Engine.NEURAL : Engine.STANDARD,
    LanguageCode: 'en-US',
  });

  const response = await polly.send(command);

  if (!response.AudioStream) {
    throw new Error(`Polly returned no AudioStream for ${landmarkId}/${contentType}`);
  }

  // Stream to file
  const writer = createWriteStream(filePath);
  await pipeline(response.AudioStream, writer);

  console.log(`  [polly] generated: ${filename}`);
  return filePath;
}

/**
 * Synthesize all content variants for a landmark.
 * Returns a map of contentType -> filePath.
 */
export async function synthesizeAllVariants({ landmarkId, scripts, premium = false }) {
  const variants = [
    { key: 'ambient_short',       script: scripts.ambient_short },
    { key: 'deep_dive_history',   script: scripts.deep_dive_history },
    { key: 'deep_dive_geography', script: scripts.deep_dive_geography },
    { key: 'deep_dive_culture',   script: scripts.deep_dive_culture },
  ];

  const results = {};
  for (const { key, script } of variants) {
    if (!script) continue;
    results[key] = await synthesizeAudio({
      landmarkId,
      contentType: key,
      script,
      premium,
    });
  }

  return results;
}
