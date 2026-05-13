# Base Services Directory

This folder contains shared service logic, AI integrations, and utility modules for the Django backend. It is organized to support advanced features such as Retrieval-Augmented Generation (RAG), agent testing, and audio processing.

## Structure

```
services/
├── __init__.py
├── README.md
├── agent/
│   └── rag/
│       ├── __init__.py
│       ├── providers.py
│       └── services/
│           ├── __init__.py
│           ├── document_loader.py
│           ├── generative.py
│           ├── token_logger.py
│           └── vector_store.py
├── audio/
│   ├── __init__.py
│   ├── response_logic.py
│   └── setup.py
```

## Contents

### `agent/rag/`

- **providers.py**: Defines RAG providers and integration logic.
- **services/**: Contains modules for document loading, generative AI, token logging, and vector database operations.
  - **document_loader.py**: Loads and preprocesses documents for RAG.
  - **generative.py**: Implements generative AI logic.
  - **token_logger.py**: Tracks token usage for analytics or billing.
  - **vector_store.py**: Manages vector database (e.g., ChromaDB) interactions.

### `audio/`

- Modules for audio processing and response logic.
  - **response_logic.py**: Implements audio response features.
  - **setup.py**: Setup script for audio services.

## Usage

- Import and use these services in your Django apps for advanced AI, document, and audio features.

- Configure environment variables in `.env.dev` or `.env.prod` files as needed for RAG and vector store integrations.

- Ensure your Google API key has access to the Gemini models you configure in the environment variables.

## Configuration

### Gemini Model Configuration

The RAG services use Google's Gemini models for generative AI operations. You can configure which Gemini models to use by setting environment variables in the `.env.dev` or`.env.prod` file located at root of project with variable below.

#### Environment Variables

```bash
# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Gemini Model Configuration
GEMINI_MODEL = gemini-2.0-flash
```