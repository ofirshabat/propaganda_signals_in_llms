# Propaganda Signals in LLMs — Code & Data

This repository contains the **data**, **prompt templates**, and **evaluation pipeline** used in our paper, *“Propaganda Signals in LLMs: Perspectival Divergence and Narrative Framing in the Russia–Ukraine War”*, accepted to **Findings of the Association for Computational Linguistics: ACL 2026**.

---

## Repository structure

```

data/
neutral_data.json           # evaluation set: {neutral, ru, ua}

src/
pipeline/                   # end-to-end scripts: translate -> judge -> write JSONL + summaries
translator/                 # translators (local HF + API adapters)
judge/                      # LLM-as-a-judge implementations + judge prompts
statistics/                 # analysis notebooks (kappa + significance tests)

````

### `data/`
- **`neutral_data.json`**: the evaluation set used in the paper.
  - `messages[i].neutral`: neutral English event statement  
  - `messages[i].ru`: RU-oriented reference text  
  - `messages[i].ua`: UA-oriented reference text

### `src/translator/`
Translator backends that produce a translation from the neutral English sentence:
- **Local HF translator** (`translator.py`): runs a HuggingFace CausalLM via `transformers` pipeline.
- **API translators considered in the paper**:
  - `gpt5_openai_translator.py` (OpenAI GPT-5.1; set `OPENAI_API_KEY`)
  - `gemini_openai_translator.py` (Google Gemini 3 Pro via OpenAI-compatible endpoint; set `GEMINI_API_KEY`)
  - `kimi_openai_translator.py` (Moonshot/Kimi; set `MOONSHOT_API_KEY`)
- All other models evaluated in the paper (Mistral, Qwen, LLaMA, DeepSeek, Falcon) run via the local HF translator.

Prompt templates for translation live in `src/translator/prompts/prompts.py` as `SYSTEM_PROMPTS_LIB`
(e.g., neutral, news-style, journalist voice, influencer/social, telegram-channel styles).

### `src/judge/`
Implements the **semantic distance judge** used to score how close a model output is to a POV reference:
- `judge.py`: **`LlamaJudge`** — primary judge used in the paper (local HuggingFace model; structured JSON output with `score` + `explanation`)
- `gpt_judge.py`: **`GPT51OpenAIJudge`** (OpenAI GPT-5.1; set `OPENAI_API_KEY`) — additional judge used for inter-judge agreement validation (κ = 0.89 vs. primary judge)
- `Falcon-7B-Instruct` can also be used as an additional judge via the local HF path (κ = 0.86 vs. primary judge)

Judge prompt templates are in `src/judge/prompts/prompts.py` as `JUDGE_SYSTEM_PROMPTS_LIB`.

### `src/pipeline/`
Runnable scripts that glue everything together and write outputs to disk:
- **`en_to_ru_bias.py` / `en_to_ua_bias.py`**  
  Translate each `neutral` sentence into RU/UA under *all* prompt templates in `SYSTEM_PROMPTS_LIB`,
  then judge MT vs the matching RU/UA human reference, producing:
  - per-example `*_results.jsonl`
  - aggregated `*_summary.json`
- **`en_to_ru_same.py` / `en_to_ua_same.py`**  
  Re-judge **precomputed** MT runs (useful when you already have translations saved as JSONL and want to re-score with a different judge).

### `src/statistics/`
Jupyter notebooks for aggregating runs and reproducing analysis:
- `calculate_kappa*.ipynb`: inter-judge agreement (Cohen’s kappa)
- `calculate_tests.ipynb`: cell-wise tests / significance workflow used for tables

---

## Quickstart

### Setup (Python)
Minimal dependencies depend on what you run:

- **Local HF translation / local judge**: `torch`, `transformers`, `langchain`, `pydantic`
- **OpenAI judge / GPT translation**: `openai` (and `OPENAI_API_KEY`)

Example (adjust for your environment):
```bash
pip install -U torch transformers langchain langchain-community pydantic openai
````


## Running the pipeline

### Translate EN→RU and judge vs RU references

```bash
python src/pipeline/en_to_ru_bias.py \
  --data data/neutral_data.json \
  --translator_model_name llama3-8b \
  --local_translator_model /path/to/your/local-hf-translator \
  --local_judge_model /path/to/your/local-judge-model \
  --judge_model llama \
  --judge_prompt_name BASE_JUDGE \
  --out_dir runs/
```

### Use GPT-5.1 translation (API) + local judge

```bash
export OPENAI_API_KEY="..."
python src/pipeline/en_to_ru_bias.py \
  --data data/neutral_data.json \
  --translator_model_name gpt-5.1 \
  --remote_translator_model gpt-5.1 \
  --local_judge_model /path/to/your/local-judge-model \
  --judge_model llama \
  --judge_prompt_name BASE_JUDGE \
  --out_dir runs/
```

### Re-judge a precomputed MT run (JSONL)

If you already have translations saved as JSONL, re-score them (default MT key is `mt`):

```bash
python src/pipeline/en_to_ru_same.py \
  --data data/neutral_data.json \
  --lang ru \
  --translator_model_name gpt-5.1 \
  --precomputed_mt_root runs/ \
  --precomputed_mt_key mt \
  --judge_model gpt-5.1 \
  --judge_prompt_name BASE_JUDGE \
  --out_dir reruns/
```

---

## Outputs

Each run writes:

* `*_results.jsonl`: one line per example with `{id, en, mt/ru_mt/ua_mt, score, explanation, ...}`
* `*_summary.json`: aggregate stats (e.g., mean semantic distance; lower = closer alignment)

Run directories are timestamped and nested by:
`<out_dir>/<judge>_judge/<judge_prompt>/<translator>/<prompt_name>/<lang_run_timestamp>/`
