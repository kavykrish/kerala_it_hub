FROM python:3.13-slim

WORKDIR /app

# trafilatura/lxml need libxml2/libxslt at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libxml2 libxslt1.1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so it doesn't have to be
# re-downloaded from Hugging Face on every cold start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY backend/ backend/
COPY mcp_server/ mcp_server/
COPY web_retrieval/ web_retrieval/

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
