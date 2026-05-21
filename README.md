# DocBot — Backend (agent_developer)

A **Django 5 + Django REST Framework** backend powering the DocBot enterprise internal document Q&A system. Features an **Agentic RAG pipeline** with hybrid dense/sparse retrieval, multi-step reasoning, confidence gating, and strict document-only answering — supporting **Khmer and English** languages.

---

## ✨ Features

- 🤖 **Agentic RAG** — Multi-step retrieval with planner, sub-query expansion, reranking, confidence gating, and self-verification before answering
- 🔍 **Hybrid Search** — BGE-M3 dense embeddings + BM25 sparse retrieval fused over Qdrant
- 📚 **Document Ingestion** — PDF, DOCX, PPTX, XLSX, images (OCR), and more via Docling/Unstructured
- 🗄️ **Vector Store** — Qdrant with `is_active` soft-delete filtering, language-aware retrieval
- 💬 **Streaming Chat** — Server-Sent Events (SSE) streaming responses
- 📋 **Queue System** — Celery-based async job queue for high-load question handling
- 🔐 **Auth** — JWT tokens + optional LDAP/Active Directory integration
- 🏢 **Multi-tenant** — Departments, document types, per-user access control
- 📊 **Feedback** — User thumbs up/down ratings stored in PostgreSQL
- 🐳 **Docker-ready** — Full docker-compose setup with PostgreSQL, Redis, and Qdrant

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | Django 5.2 + DRF 3.16 |
| Language | Python 3.11+ |
| Database | PostgreSQL (psycopg2) |
| Vector DB | Qdrant 1.12 |
| Cache / Queue | Redis + Celery |
| LLM | Ollama (`gemma4:26b` — local) |
| Embedding | BGE-M3 (FlagEmbedding) |
| Reranker | BGE-Reranker-v2-m3 |
| RAG Framework | LangChain + langchain-ollama |
| OCR | Surya OCR + faster-whisper |
| Auth | JWT (simplejwt) + django-auth-ldap |
| Server | Gunicorn + WhiteNoise |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose**
- **Ollama** running locally with `gemma4:26b` pulled
- BGE-M3 model downloaded to `app/models/bge-m3/`
- BGE-Reranker-v2-m3 downloaded to `app/models/bge-reranker-v2-m3/`

### 1. Configure environment

```bash
cp .env.example .env.dev
```

Edit `.env.dev` with your values — key settings:

```env
SECRET_KEY=your-strong-secret-key
DB_PASSWORD=your-db-password

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Ollama LLM
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=gemma4:26b

# Model paths (inside container)
BGE_M3_MODEL_PATH=/usr/src/app/models/bge-m3
RERANKER_MODEL_PATH=/usr/src/app/models/bge-reranker-v2-m3

# Agentic RAG
AGENTIC_RAG_ENABLED=True
AGENTIC_TOP_K_BROAD=40
AGENTIC_TOP_K_RERANK=15
AGENTIC_CONFIDENCE_THRESHOLD_EN=0.30
AGENTIC_CONFIDENCE_THRESHOLD_KM=0.25
```

### 2. Start all services

```bash
docker-compose up --build
```

This starts:
- **Django API** on port `8001`
- **PostgreSQL** database
- **Redis** cache + Celery broker
- **Qdrant** vector database on port `6333`

### 3. Apply migrations

```bash
docker-compose exec web python manage.py migrate
```

### 4. Create a superuser

```bash
docker-compose exec web python manage.py createsuperuser
```

### 5. Access the API

- **API root**: `http://localhost:8001/api/v1/`
- **Health check**: `http://localhost:8001/health/`
- **Django admin**: `http://localhost:8001/admin/`

---

## 📡 API Endpoints

All endpoints are prefixed with `/api/v1/`.

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `login/` | Login (local or LDAP) — returns JWT tokens |
| POST | `refresh/` | Refresh access token |
| POST | `logout/` | Invalidate session |
| GET | `whoami/` | Get current user info |
| GET/PUT | `profile/` | View/update user profile |
| POST | `profile/avatar/` | Upload avatar |

### Chat
| Method | Endpoint | Description |
|---|---|---|
| GET | `chats/sessions/` | List all chat sessions |
| GET | `chats/sessions/<id>/` | Get session with messages |
| POST | `chats/session/messages/` | Create a message in a session |
| GET | `chats/stream/` | SSE streaming chat (Agentic RAG) |
| POST | `chats/sessions/<id>/archive/` | Archive a session |
| POST | `chats/sessions/<id>/restore/` | Restore archived session |
| DELETE | `chats/sessions/<id>/delete/` | Delete a session |
| PATCH | `chats/sessions/<id>/rename/` | Rename a session |
| POST | `chats/feedback/` | Submit thumbs up/down feedback |
| GET | `chats/feedback/list/` | List all feedback (superadmin only) |
| POST | `chats/queue/enqueue/` | Enqueue a question (async queue) |
| GET | `chats/queue/status/<job_id>/` | Poll queue job status |
| GET | `chats/queue/stats/` | Queue health stats |

### Documents
| Method | Endpoint | Description |
|---|---|---|
| POST | `documents/upload/` | Upload and index a document |
| GET | `documents/list/` | List active documents |
| GET | `documents/<id>/` | Get document details |
| PUT | `documents/<id>/update/` | Update document metadata |
| DELETE | `documents/<id>/delete/` | Soft-delete a document |
| POST | `documents/<id>/restore/` | Restore a soft-deleted document |
| GET | `documents/<id>/pdf-url/` | Get signed PDF view URL |
| POST | `documents/<id>/upload-version/` | Upload a new document version |
| GET | `documents/<id>/versions/` | List all versions |
| GET | `documents/trash/` | List soft-deleted documents |
| GET | `documents/years/` | Get list of unique document years |

### Departments & Document Types
| Method | Endpoint | Description |
|---|---|---|
| GET/POST | `departments/` | List / create departments |
| PUT/DELETE | `departments/<id>/update/` | Update / delete department |
| GET/POST | `document-types/` | List / create document types |
| PUT/DELETE | `document-types/<id>/update/` | Update / delete document type |

### Health & Utilities
| Method | Endpoint | Description |
|---|---|---|
| GET | `health/` | Simple health check |
| GET | `qdrant/health/` | Qdrant connectivity check |
| GET | `ldap/health/` | LDAP connectivity check |
| GET | `get-signed-download-url/` | Generate a signed download URL |

---

## 🧠 Agentic RAG Architecture

The system uses a multi-step agentic pipeline instead of single-pass RAG:

```
User Query
    │
    ▼
┌─────────────────┐
│  Language Gate  │  Detects km/en — rejects other languages
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│    Planner      │  Classifies intent (lookup / comparison /
│                 │  summarization / policy / multi-hop)
│                 │  Generates sub-queries if needed
└────────┬────────┘
         │
    ▼
┌─────────────────────────────────────┐
│          Multi-step Retrieval        │
│  Pass A: Hybrid search top_k=40     │
│  Pass B: Sub-query searches (merge) │
│  Pass C: Rerank → top 15            │
└────────┬────────────────────────────┘
         │
    ▼
┌─────────────────┐
│ Confidence Gate │  Low score → clarifying question or
│                 │  "not found in documents" response
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Answer Synthesis│  LLM synthesizes answer from chunks only
│                 │  Each claim → [doc_title | page | section]
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Self-Verification│ Verifies every claim maps to a chunk;
│                 │  no external knowledge; citations present
└────────┬────────┘
         │
    ▼
Streaming Response to Client
```

### Key modules (`base/services/agent/rag/`)

| File | Purpose |
|---|---|
| `config.py` | All configurable thresholds and settings |
| `planner.py` | Intent classification + sub-query generation |
| `tools.py` | `hybrid_search`, `rerank`, `get_chunk`, `summarize_table`, `confidence_score` |
| `retriever.py` | Multi-pass retrieval orchestration |
| `agent.py` | Full agentic pipeline coordinator |
| `vector_store.py` | Qdrant upsert + hybrid search |
| `generative.py` | LLM call integration (entry point) |

---

## 📁 Project Structure

```
agent_developer/
├── app/
│   ├── core/                    # Django settings, URLs, ASGI/WSGI, Celery
│   └── models/                  # Local model weights (not in git)
│       ├── bge-m3/              # BGE-M3 embedding model
│       └── bge-reranker-v2-m3/  # Reranker model
├── base/
│   ├── services/
│   │   └── agent/rag/           # Agentic RAG pipeline modules
│   ├── ldap/                    # LDAP bootstrap data
│   └── utils/                   # api_response helpers, etc.
├── chat/                        # Chat sessions, messages, streaming, feedback
├── department/                  # Department management
├── docs_type/                   # Document type management
├── document/                    # Document upload, indexing, versioning
├── user/                        # User auth, profiles, LDAP
├── docker-compose.yml           # Full stack docker-compose
├── Dockerfile                   # App container image
├── requirements.txt             # Python dependencies
├── manage.py
└── .env.example                 # Environment variable template
```

---

## 🔑 Environment Variables

### Core Django

| Variable | Description |
|---|---|
| `SECRET_KEY` | Django secret key (**change in production**) |
| `DEBUG` | `True` for dev, `False` for production |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated allowed host names |
| `TIME_ZONE` | Default: `Asia/Phnom_Penh` |

### Database

| Variable | Description |
|---|---|
| `DB_ENGINE` | `django.db.backends.postgresql` |
| `DB_DATABASE` | PostgreSQL database name |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_HOST` | DB host (default: `db` in Docker) |
| `DB_PORT` | DB port (default: `5432`) |

### Vector DB & LLM

| Variable | Description |
|---|---|
| `QDRANT_HOST` | Qdrant host (default: `qdrant`) |
| `QDRANT_PORT` | Qdrant port (default: `6333`) |
| `OLLAMA_BASE_URL` | Ollama API base URL |
| `OLLAMA_MODEL` | Model name (default: `gemma4:26b`) |
| `BGE_M3_MODEL_PATH` | Path to BGE-M3 model |
| `RERANKER_MODEL_PATH` | Path to reranker model |

### Agentic RAG Thresholds

| Variable | Default | Description |
|---|---|---|
| `AGENTIC_RAG_ENABLED` | `True` | Enable/disable agentic pipeline |
| `AGENTIC_TOP_K_BROAD` | `40` | Pass A retrieval count |
| `AGENTIC_TOP_K_SUBQUERY` | `20` | Per sub-query retrieval count |
| `AGENTIC_TOP_K_RERANK` | `15` | Final reranked chunks count |
| `AGENTIC_CONFIDENCE_THRESHOLD_EN` | `0.30` | Confidence cutoff for English |
| `AGENTIC_CONFIDENCE_THRESHOLD_KM` | `0.25` | Confidence cutoff for Khmer |
| `AGENTIC_MAX_SUBQUERIES` | `3` | Max sub-queries per question |
| `AGENTIC_MAX_RETRIES` | `2` | Retrieval retry attempts |
| `AGENTIC_VERIFICATION_ENABLED` | `True` | Enable self-verification step |

### Queue

| Variable | Default | Description |
|---|---|---|
| `AGENTIC_QUEUE_MAX_WORKERS` | `4` | Celery worker concurrency |
| `AGENTIC_QUEUE_MAX_QUEUE_SIZE` | `100` | Max queued jobs |
| `AGENTIC_QUEUE_TIMEOUT_SECONDS` | `5` | Job enqueue timeout |

### LDAP (optional)

| Variable | Description |
|---|---|
| `ENABLE_LDAP_AUTH` | `True` to enable LDAP login |
| `AUTH_LDAP_SERVER_URI` | LDAP server URI |
| `AUTH_LDAP_BIND_DN` | Bind DN for LDAP |
| `AUTH_LDAP_USER_BASE_DN` | Base DN for user lookup |
| `AUTH_LDAP_GROUP_BASE_DN` | Base DN for group lookup |

---

## 🐳 Docker Services

| Service | Image | Port | Description |
|---|---|---|---|
| `web` | Custom (Dockerfile) | `8001` | Django API |
| `db` | `postgres:15` | `5432` | PostgreSQL |
| `redis` | `redis:7` | `6379` | Cache + Celery broker |
| `qdrant` | `qdrant/qdrant` | `6333` | Vector database |
| `ollama` | `ollama/ollama` | `11434` | Local LLM server |

---

## 🔄 Changing the LLM

The default model is **`gemma4:26b`** via Ollama. To change:

1. Update `OLLAMA_MODEL` in `.env.dev`
2. Pull the new model: `docker-compose exec ollama ollama pull <model-name>`
3. Restart the app: `docker-compose restart web`

To use a different LLM provider (OpenAI, Gemini, etc.):
- See [`base/services/agent/rag/README.md`](base/services/agent/rag/README.md)
- Update the LangChain model instantiation in `generative.py`
- Add the provider package to `requirements.txt`

---

## 🧪 Running Tests

```bash
docker-compose exec web python manage.py test
```

---

## 📖 Further Reading

- [Frontend README](../agent_client/README.md)
- [RAG Service README](base/services/agent/rag/README.md)
- [LDAP Group Mapping](LDAP_GROUP_MAPPING.md)
