CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (

    id SERIAL PRIMARY KEY,

    document_id TEXT REFERENCES weather_documents(id),

    chunk_index INT,

    chunk_text TEXT,

    embedding VECTOR(384),

    model_name TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()

);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);