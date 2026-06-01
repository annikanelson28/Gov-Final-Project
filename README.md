# Gov-Final-Project

Summarizes a fixed set of news articles using three LLMs — Claude (Anthropic),
ChatGPT (OpenAI), and Grok (xAI) — under identical conditions so their outputs can
be compared.

## Setup

```bash
python3 -m pip install anthropic openai
```

(Grok uses the `openai` package — the xAI API is OpenAI-compatible.)

Set the API key for whichever provider you're running. Keys are read from
environment variables and are never stored in the repo:

```bash
export ANTHROPIC_API_KEY="..."   # starts with sk-ant-
export OPENAI_API_KEY="..."      # starts with sk-
export XAI_API_KEY="..."         # starts with xai-
```

## Usage

Each provider writes its own output file. Smoke-test first, then run the full 60:

```bash
python3 run_summaries.py --input articles.csv --output claude_summaries.csv --providers claude --limit 2
python3 run_summaries.py --input articles.csv --output claude_summaries.csv --providers claude
python3 run_summaries.py --input articles.csv --output chatgpt_summaries.csv --providers openai
python3 run_summaries.py --input articles.csv --output grok_summaries.csv --providers grok
```

Model names are set in the `MODELS` dictionary at the top of `run_summaries.py`.

## Files

- `run_summaries.py` — the summarization pipeline
- `articles.csv` — master list of input articles
- `*_summaries.csv` — generated output, one file per provider
