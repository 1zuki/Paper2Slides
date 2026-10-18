# Tsunami-surrogate slides: setup and commands

This guide records the verified setup state for `/home/izu/Projects/Paper2Slides` and the commands for generating an academic deck from the current tsunami-surrogate manuscript.

## Verified status

The repository was not ready for a clean API-backed run when checked:

- `paper2slides/.env` was missing.
- Python dependencies were missing from the system interpreter. A local `.venv/` was created and `requirements.txt` installed successfully.
- `frontend/node_modules/` was missing.
- The original OpenAI-compatible attempt returned HTTP 401 for the shell's `OPENAI_API_KEY` value.
- MinerU was installed in the project virtual environment, but its executable was not on `PATH` until `.venv/bin` was added.
- The available Codex gateway uses a streaming Responses route with a gateway-specific `messages` request body; the OpenAI SDK's default non-streaming Responses shape does not work against it.
- The gateway's standalone image model was not permitted for the current credential, so the Paper2Slides image stage could not be completed.
- Text provider routing is now selectable: `RAG_LLM_PROVIDER=openai_compatible` preserves Chat Completions, while `RAG_LLM_PROVIDER=codex_gateway` sends text/vision requests to the external Responses API.

The real Paper2Slides attempt parsed the 41-page manuscript with MinerU and produced a reusable parsed checkpoint under `outputs/main/paper/rag_output/`. With the provider adapter enabled, the external gateway completed summary extraction and generated a 7-slide plan. Generation then stopped because no `IMAGE_GEN_API_KEY` is configured, and the gateway image model is not authorized for the current credential.

## One-time local setup

```bash
cd /home/izu/Projects/Paper2Slides
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"

# Needed only for the web UI
cd frontend
npm install
cd ..
```

## API configuration

Create the local environment file:

```bash
cd /home/izu/Projects/Paper2Slides
cp paper2slides/.env.example paper2slides/.env
${EDITOR:-nano} paper2slides/.env
```

### Text planning and summarization

Create an OpenAI API key at <https://platform.openai.com/api-keys>, then set:

```dotenv
RAG_LLM_API_KEY="sk-..."
RAG_LLM_BASE_URL=""
LLM_MODEL="gpt-4o"
RAG_LLM_MAX_TOKENS="16000"
```

The normal `openai_compatible` provider continues using Chat Completions. The `codex_gateway` option routes text/vision work through the gateway adapter.

To route the text/vision stages through that gateway, use:

```dotenv
RAG_LLM_PROVIDER="codex_gateway"
CODEX_API_KEY="..."
CODEX_BASE_URL="https://codex.finnvnoi.top/backend-api/codex"
CODEX_MODEL="gpt-5.6-sol"
```

The adapter targets the gateway's streaming `/v1/responses` route. Fast mode works without embeddings; normal RAG mode still requires a separate embedding service configured through `RAG_EMBEDDING_API_KEY` and `RAG_EMBEDDING_BASE_URL`. Image generation remains a separate provider and still needs `IMAGE_GEN_API_KEY`.

### Image generation through OpenRouter

1. Create a key at <https://openrouter.ai/keys>.
2. Choose an image-capable model available to that account.
3. Set:

```dotenv
IMAGE_GEN_PROVIDER="openrouter"
IMAGE_GEN_API_KEY="sk-or-v1-..."
IMAGE_GEN_BASE_URL="https://openrouter.ai/api/v1"
IMAGE_GEN_MODEL="google/gemini-3-pro-image-preview"
```

### Image generation through Google Gemini

1. Create a key at <https://aistudio.google.com/apikey>.
2. Set an image-capable model and request an image MIME type:

```dotenv
IMAGE_GEN_PROVIDER="google"
IMAGE_GEN_API_KEY="AIza..."
GOOGLE_GENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
IMAGE_GEN_MODEL="models/gemini-1.5-flash"
IMAGE_GEN_RESPONSE_MIME_TYPE="image/png"
```

Keep keys only in `paper2slides/.env` or the process environment. Do not commit them or paste them into logs.

## Check the setup

```bash
cd /home/izu/Projects/Paper2Slides
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"
./scripts/check_config.sh
```

The checker verifies that the required variables exist. It does not prove that an API key is active.

## CLI: generate academic slides

```bash
cd /home/izu/Projects/Paper2Slides
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"

python -m paper2slides \
  --input /home/izu/Projects/tsunami-surrogate/paper/build/main.pdf \
  --output slides \
  --content paper \
  --style academic \
  --length medium \
  --parallel 2 \
  --output-dir outputs
```

Short preview:

```bash
python -m paper2slides \
  --input /home/izu/Projects/tsunami-surrogate/paper/build/main.pdf \
  --output slides \
  --content paper \
  --style academic \
  --length short \
  --fast \
  --parallel 2 \
  --output-dir outputs
```

## Resume after a partial run

If parsing completed but summary failed, reuse the parsed checkpoint:

```bash
python -m paper2slides \
  --input /home/izu/Projects/tsunami-surrogate/paper/build/main.pdf \
  --output slides \
  --content paper \
  --style academic \
  --length short \
  --fast \
  --from-stage summary \
  --parallel 2 \
  --output-dir outputs
```

Restart points:

```text
--from-stage rag       parse and query again
--from-stage summary   reuse parsed/RAG checkpoint
--from-stage plan      reuse the summary and rebuild the slide plan
--from-stage generate  reuse the plan and regenerate slide images
```

List outputs:

```bash
python -m paper2slides --list --output-dir outputs
```

## Web interface

```bash
cd /home/izu/Projects/Paper2Slides
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"
./scripts/start.sh
```

Open <http://localhost:5173>. Check services with:

```bash
./scripts/check_status.sh
```

## Current source-backed preview

Because the API-backed Paper2Slides run was blocked by the credential and image-provider issues above, the workspace also contains a rendered source-backed academic preview built from the current manuscript and its original figures. It includes a montage and individual slide PNGs for reviewing the narrative and visual direction. It is not a completed Paper2Slides PPTX/PDF and does not claim that the API pipeline completed.

## Research-profile deck

Use the research profile when the goal is a technical seminar, lab meeting, or reusable research record rather than a short overview:

```bash
cd /home/izu/Projects/Paper2Slides
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"

python -m paper2slides \
  --input /home/izu/Projects/tsunami-surrogate/paper/build/main.pdf \
  --output slides \
  --content paper \
  --style academic \
  --profile research \
  --min-slides 24 \
  --max-slides 36 \
  --include-appendix \
  --parallel 2 \
  --output-dir outputs-tsu
```

The research profile has separate checkpoints, so it does not reuse the standard summary. With `--include-appendix`, the RAG and summary checkpoints also receive an `_appendix` suffix. Inspect the plan before spending image-generation time:

```bash
python - <<'PY'
import json
from pathlib import Path

plan = Path("outputs-tsu/main/paper/normal/slides_academic_long_research_24-36_appendix/checkpoint_plan.json")
data = json.loads(plan.read_text())
meta = data["plan"].get("metadata", {})
print("requested:", meta.get("requested_slide_range"))
print("actual:", meta.get("actual_slide_count"))
for i, slide in enumerate(data["plan"].get("sections", []), 1):
    print(f"{i:02d}. {slide['title']}")
PY
```

If only the plan needs revision, keep the research RAG/summary checkpoints and rerun from `plan`:

```bash
python -m paper2slides \
  --input /home/izu/Projects/tsunami-surrogate/paper/build/main.pdf \
  --output slides \
  --content paper \
  --style academic \
  --profile research \
  --min-slides 24 \
  --max-slides 36 \
  --include-appendix \
  --from-stage plan \
  --output-dir outputs-tsu
```
