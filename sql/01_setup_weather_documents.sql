CREATE TABLE IF NOT EXISTS weather_documents (

    id TEXT PRIMARY KEY,

    location TEXT NOT NULL,

    source_type TEXT NOT NULL,

    headline TEXT,

    narrative_text TEXT NOT NULL,

    issued_at TIMESTAMPTZ,

    payload JSONB,

    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()

);

CREATE INDEX IF NOT EXISTS idx_weather_location
ON weather_documents(location);