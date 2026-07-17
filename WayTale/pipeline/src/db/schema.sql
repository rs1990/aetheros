-- WayTale database schema
-- Requires PostGIS extension

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Landmarks: fixed points of interest
CREATE TABLE IF NOT EXISTS landmarks (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  wikidata_id  TEXT UNIQUE NOT NULL,
  name         TEXT NOT NULL,
  description  TEXT,
  latitude     DOUBLE PRECISION NOT NULL,
  longitude    DOUBLE PRECISION NOT NULL,
  location     GEOGRAPHY(POINT, 4326) GENERATED ALWAYS AS (
                 ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
               ) STORED,
  category     TEXT[],
  country_code TEXT NOT NULL DEFAULT 'US',
  language     TEXT NOT NULL DEFAULT 'en',
  wikipedia_en TEXT,   -- Wikipedia page title
  image_url    TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS landmarks_location_idx ON landmarks USING GIST (location);
CREATE INDEX IF NOT EXISTS landmarks_wikidata_idx ON landmarks (wikidata_id);

-- Content: pre-generated narration scripts + audio URLs
CREATE TABLE IF NOT EXISTS landmark_content (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  landmark_id  UUID NOT NULL REFERENCES landmarks(id) ON DELETE CASCADE,
  content_type TEXT NOT NULL CHECK (content_type IN ('ambient', 'deep_dive')),
  variant      TEXT,   -- 'history' | 'culture' | 'geography' | 'combined'
  length       TEXT CHECK (length IN ('short', 'long')),
  script       TEXT NOT NULL,
  fact_type    TEXT NOT NULL CHECK (fact_type IN ('verified', 'legend', 'mixed')),
  sources      JSONB  NOT NULL DEFAULT '[]',
  audio_url    TEXT,   -- CDN URL after TTS generation
  tts_provider TEXT,   -- 'polly' | 'google' | 'elevenlabs' | 'device'
  language     TEXT NOT NULL DEFAULT 'en',
  region       TEXT NOT NULL DEFAULT 'US',
  status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'reviewed', 'published', 'rejected')),
  version      INTEGER NOT NULL DEFAULT 1,
  created_at   TIMESTAMPTZ DEFAULT NOW(),
  updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS content_landmark_idx ON landmark_content (landmark_id);
CREATE INDEX IF NOT EXISTS content_status_idx   ON landmark_content (status);
CREATE INDEX IF NOT EXISTS content_type_idx     ON landmark_content (content_type, variant, length);

-- Raw source facts pulled from Wikipedia/Wikidata before LLM processing
CREATE TABLE IF NOT EXISTS landmark_facts (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  landmark_id UUID NOT NULL REFERENCES landmarks(id) ON DELETE CASCADE,
  source      TEXT NOT NULL,  -- 'wikipedia' | 'wikidata' | 'wikivoyage'
  source_url  TEXT,
  raw_text    TEXT NOT NULL,
  fact_type   TEXT NOT NULL DEFAULT 'verified'
                CHECK (fact_type IN ('verified', 'legend')),
  fetched_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS facts_landmark_idx ON landmark_facts (landmark_id);
