# Databricks notebook source

# MAGIC %md
# MAGIC # Weather Intelligence - Ingest Weather Embeddings
# MAGIC
# MAGIC This notebook:
# MAGIC
# MAGIC 1. Reads weather documents from Lakebase.
# MAGIC 2. Splits long weather text into chunks.
# MAGIC 3. Generates sentence embeddings.
# MAGIC 4. Stores embeddings into pgvector.
# MAGIC 5. Verifies the results.

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers psycopg2-binary

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values

import lakebase

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800

CHUNK_OVERLAP = 100

model = SentenceTransformer(MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Weather Documents

# COMMAND ----------

with lakebase.get_connection() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                id,
                narrative_text
            FROM weather_documents
            ORDER BY synced_at
        """)

        documents = cur.fetchall()

print(f"Found {len(documents)} weather documents.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Weather Documents

# COMMAND ----------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Embeddings

# COMMAND ----------

rows = []

for document in documents:

    chunks = chunk_text(document["narrative_text"])

    for index, chunk in enumerate(chunks):

        embedding = model.encode(chunk).tolist()

        rows.append(
            (
                document["id"],
                index,
                chunk,
                embedding,
                MODEL_NAME
            )
        )

print(f"Created {len(rows)} embeddings.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Insert Embeddings into Lakebase

# COMMAND ----------

with lakebase.get_connection() as conn:

    with conn.cursor() as cur:

        sql = """
        INSERT INTO weather_embeddings
        (
            document_id,
            chunk_index,
            chunk_text,
            embedding,
            model_name
        )
        VALUES %s
        """

        values = []

        for row in rows:

            values.append(
                (
                    row[0],
                    row[1],
                    row[2],
                    str(row[3]),
                    row[4]
                )
            )

        execute_values(
            cur,
            sql,
            values,
            template="""
            (
                %s,
                %s,
                %s,
                %s::vector,
                %s
            )
            """
        )

        conn.commit()

print(f"Inserted {len(rows)} embeddings into weather_embeddings.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Embeddings

# COMMAND ----------

with lakebase.get_connection() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                COUNT(*) AS total_embeddings
            FROM weather_embeddings
        """)

        result = cur.fetchone()

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Records

# COMMAND ----------

with lakebase.get_connection() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                document_id,
                chunk_index,
                model_name,
                created_at
            FROM weather_embeddings
            ORDER BY created_at DESC
            LIMIT 10
        """)

        sample_rows = cur.fetchall()

for row in sample_rows:
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Complete
# MAGIC
# MAGIC This notebook:
# MAGIC
# MAGIC - Read weather documents from Lakebase
# MAGIC - Split narrative text into chunks
# MAGIC - Generated sentence embeddings using
# MAGIC   `sentence-transformers/all-MiniLM-L6-v2`
# MAGIC - Stored embeddings in the `weather_embeddings` table
# MAGIC - Verified the inserted records
