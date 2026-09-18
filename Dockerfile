FROM python:3.13-slim

WORKDIR /app

# trafilatura/lxml need libxml2/libxslt at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libxml2 libxslt1.1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY mcp_server/ mcp_server/
COPY web_retrieval/ web_retrieval/

# Bake the embedding model into the image so it doesn't have to be
# re-downloaded from Hugging Face on every cold start.
RUN python -c "from web_retrieval.embedding_model import load_embedding_model; load_embedding_model()"

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
