#!/usr/bin/env python3
"""
run_summaries.py
================
Batch summarization harness for the "Measuring Partisan Bias in Balanced News
Summarization" project.

Runs each article through Claude, OpenAI (ChatGPT), and Grok using an IDENTICAL
prompt and configuration, writing every summary to a tidy CSV ready for
annotation and BERTScore analysis.

----------------------------------------------------------------------------
WHY THE DESIGN LOOKS LIKE THIS (read before changing anything)
----------------------------------------------------------------------------
1. ONE ARTICLE PER CALL. Each summary is generated in its own stateless request
   with no other article in context, so there is zero cross-article
   contamination and no position effects. This is essential for a per-article
   compression measurement.

2. IDENTICAL PROMPT. The same neutral prompt is sent to all three providers.
   It deliberately says nothing about "balance" or "both sides" -- we want to
   observe each model's DEFAULT behavior, which is what real users get.

3. SAMPLING SETTINGS -- IMPORTANT ASYMMETRY ACROSS PROVIDERS.
   As of mid-2026:
     * Newest Claude models (Opus 4.7+/4.8) REJECT a custom temperature (400 error).
     * Newest OpenAI models (GPT-5 family) only accept the default temperature (1).
     * Grok (xAI) still accepts temperature normally.
   You therefore CANNOT set temperature=0 uniformly on the latest models.
   Pick ONE of two strategies and keep it consistent (see DETERMINISM_MODE):
     - "older_models": use models that still accept temperature=0 on all three
       providers, for maximum determinism and cross-model comparability.
     - "newest_models": use the latest models, omit temperature everywhere, and
       rely on REPLICATES (multiple runs per article) to characterize variance.
   This script handles both; just set the constants below.

4. REPLICATES. Each article is summarized N_RUNS times per model so you can
   measure output stability (important when temperature can't be pinned to 0).

5. FULL PROVENANCE. Every row records the exact model string, run index,
   timestamp, token usage, and the verbatim summary, so results are reproducible.
----------------------------------------------------------------------------

SETUP
-----
    pip install anthropic openai pandas

    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."
    export XAI_API_KEY="xai-..."

INPUT
-----
    A CSV named articles.csv (or pass --input) with at least these columns:
        article_id   - unique id (e.g. 1..60)
        topic        - Healthcare / Middle East / Immigration / Tax Policy
        title        - article title (NOT sent to the model)
        text         - the full article body to summarize

OUTPUT
------
    summaries_output.csv  with one row per (article x model x run).

USAGE
-----
    python run_summaries.py --input articles.csv --output summaries_output.csv
    python run_summaries.py --providers claude openai      # subset
    python run_summaries.py --limit 3                      # quick smoke test
"""

import argparse
import csv
import os
import sys
import time
import datetime as dt

# ----------------------------------------------------------------------------
# CONFIGURATION  -- edit these to match your final study design
# ----------------------------------------------------------------------------

DETERMINISM_MODE = "older_models"   # "older_models" or "newest_models"

# Model strings per provider for each mode. Update these to the exact models you
# decide to use; verify availability in each provider's console before running.
MODELS = {
    "older_models": {
        # These accept temperature=0 -> maximum determinism & comparability.
        "claude": "claude-sonnet-4-5",
        "openai": "gpt-4o",
        "grok":   "grok-4.3",          # xAI still accepts temperature
    },
    "newest_models": {
        # Latest flagships; temperature is omitted for claude/openai.
        "claude": "claude-opus-4-8",
        "openai": "gpt-5",
        "grok":   "grok-4.3",
    },
}

# The single shared prompt. {article} is filled with the article body.
# Note: intentionally NEUTRAL -- no mention of balance, fairness, or "both sides".
PROMPT_TEMPLATE = (
    "Summarize the following news article in 150-200 words. "
    "Write in neutral, third-person prose. "
    "Do not add commentary, headings, or a title.\n\n"
    "ARTICLE:\n{article}"
)

# Optional system prompt. Keep identical across providers, or set to None.
# Using None (no system prompt) is the cleanest, most defensible choice.
SYSTEM_PROMPT = None

N_RUNS = 1            # replicates per article per model
MAX_TOKENS = 400     # comfortably above 200 words so output is never truncated
TEMPERATURE = 0.0    # used ONLY in "older_models" mode / for Grok

# Retry settings for transient API errors / rate limits.
MAX_RETRIES = 5
BACKOFF_BASE = 2.0   # seconds; exponential: 2, 4, 8, 16, 32

# ----------------------------------------------------------------------------
# Provider client wrappers
# ----------------------------------------------------------------------------

def _supports_temperature(provider: str) -> bool:
    """Whether to send a temperature param for this provider in the chosen mode."""
    if DETERMINISM_MODE == "older_models":
        return True  # all chosen older models accept temperature
    # newest_models mode: only Grok accepts custom temperature
    return provider == "grok"


class ClaudeClient:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = MODELS[DETERMINISM_MODE]["claude"]

    def summarize(self, article_text):
        kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user",
                       "content": PROMPT_TEMPLATE.format(article=article_text)}],
        )
        if SYSTEM_PROMPT:
            kwargs["system"] = SYSTEM_PROMPT
        if _supports_temperature("claude"):
            kwargs["temperature"] = TEMPERATURE
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text.strip(), resp.usage.input_tokens, resp.usage.output_tokens


class OpenAICompatibleClient:
    """Works for both OpenAI and Grok (Grok = OpenAI SDK + xAI base_url)."""
    def __init__(self, provider):
        from openai import OpenAI
        self.provider = provider
        if provider == "openai":
            self.client = OpenAI()  # reads OPENAI_API_KEY
        elif provider == "grok":
            self.client = OpenAI(
                api_key=os.environ["XAI_API_KEY"],
                base_url="https://api.x.ai/v1",
            )
        else:
            raise ValueError(provider)
        self.model = MODELS[DETERMINISM_MODE][provider]

    def summarize(self, article_text):
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user",
                         "content": PROMPT_TEMPLATE.format(article=article_text)})
        kwargs = dict(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        if _supports_temperature(self.provider):
            kwargs["temperature"] = TEMPERATURE
        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text.strip(), usage.prompt_tokens, usage.completion_tokens


def build_client(provider):
    if provider == "claude":
        return ClaudeClient()
    return OpenAICompatibleClient(provider)


# ----------------------------------------------------------------------------
# Retry wrapper
# ----------------------------------------------------------------------------

def with_retries(fn, *args):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - we want to retry on any transient error
            last_err = e
            wait = BACKOFF_BASE ** attempt
            print(f"    ! error ({type(e).__name__}): {e} -- retry {attempt+1}/{MAX_RETRIES} in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def load_articles(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"article_id", "topic", "title", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"Input CSV is missing required columns: {missing}")
        return [r for r in reader if (r.get("text") or "").strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="articles.csv")
    ap.add_argument("--output", default="summaries_output.csv")
    ap.add_argument("--providers", nargs="+",
                    default=["claude", "openai", "grok"],
                    choices=["claude", "openai", "grok"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N articles (smoke test).")
    args = ap.parse_args()

    articles = load_articles(args.input)
    if args.limit:
        articles = articles[: args.limit]

    print(f"Mode: {DETERMINISM_MODE}")
    print(f"Articles: {len(articles)} | Providers: {args.providers} | Runs each: {N_RUNS}")
    print(f"Total API calls planned: {len(articles) * len(args.providers) * N_RUNS}\n")

    clients = {p: build_client(p) for p in args.providers}

    fieldnames = [
        "article_id", "topic", "title",
        "provider", "model", "run_index",
        "summary", "summary_word_count",
        "input_tokens", "output_tokens",
        "temperature_sent", "system_prompt_used",
        "timestamp_utc",
    ]

    # Stream rows to disk as we go so a crash never loses completed work.
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for art in articles:
            aid = art["article_id"]
            for provider in args.providers:
                client = clients[provider]
                temp_sent = TEMPERATURE if _supports_temperature(provider) else ""
                for run in range(1, N_RUNS + 1):
                    print(f"[{aid:>3}] {provider:<7} run {run}/{N_RUNS} ...", end="", flush=True)
                    summary, in_tok, out_tok = with_retries(client.summarize, art["text"])
                    writer.writerow({
                        "article_id": aid,
                        "topic": art["topic"],
                        "title": art["title"],
                        "provider": provider,
                        "model": client.model,
                        "run_index": run,
                        "summary": summary,
                        "summary_word_count": len(summary.split()),
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "temperature_sent": temp_sent,
                        "system_prompt_used": bool(SYSTEM_PROMPT),
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    })
                    f.flush()
                    print(f" done ({len(summary.split())} words)")

    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
