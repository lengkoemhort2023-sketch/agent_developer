## Folder version marker

- `RAG_PACKAGE_VERSION`: `rag-2026-04-17-followup-portable-v1`
- Defined in: `base/services/agent/rag/__init__.py`

This marker lets you confirm you are using the updated `rag` folder version.

## Drop-in replacement note

This `rag` folder now contains all config constants required by its own modules
(`RETRIEVAL_K`, `QUERY_VALIDATION_ENABLED`, `FOLLOWUP_SUGGESTIONS_ENABLED`,
`FOLLOWUP_SUGGESTIONS_COUNT`).

Follow-up defaults are enabled in `rag/config.py`:

- `FOLLOWUP_SUGGESTIONS_ENABLED=True` (default)
- `FOLLOWUP_SUGGESTIONS_COUNT=3` (default)

You can still override them via environment variables:

- `AGENTIC_FOLLOWUPS_ENABLED`
- `AGENTIC_FOLLOWUPS_COUNT`

### RAG LLM provider (reference-only)

This RAG service uses LangChain. The default LLM is Google Gemini. If you want to try other providers (OpenAI), follow the official LangChain integration guides below for installation, environment variables, and usage details.

Provider docs:

- Google Gemini: https://python.langchain.com/docs/integrations/chat/google_generative_ai/
- OpenAI: https://python.langchain.com/docs/integrations/chat/openai/

### Where to change in code

- `app/base/services/agent/rag/services/generative.py`

  - Update the chat model instantiation used by `ResponseGenerationService`.

- `app/base/services/agent/rag/services/document_loader.py`
  - Update the chat model used for PDF-related processing. Make sure the LLM provider support media or pdf input.

### After adding a new provider.

- Add the provider packages to `app/requirements.txt` (e.g., `langchain-openai`, plus the provider SDK).
- Rebuild the app images and restart the services.
- Make sure to add new environment variables to configure :
  - For instance, OPENAI_MODEL: "Model name", OPENAI_API_KEY : "API KEY".  
