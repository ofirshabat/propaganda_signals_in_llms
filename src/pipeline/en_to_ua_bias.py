#!/usr/bin/env python3
# en_to_ua_bias.py

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import gc
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

# --- ensure we can import from the project root when running from translation/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB
from src.judge.judge import LlamaJudge
from src.translator.translator import MinimalTranslator, TranslatorConfig
from src.translator.kimi_openai_translator import KimiOpenAITranslator, KimiOpenAIConfig
from src.translator.gemini_openai_translator import GeminiOpenAITranslator, GeminiOpenAIConfig
from src.judge.prompts.prompts import JUDGE_SYSTEM_PROMPTS_LIB
from src.translator.gpt5_openai_translator import GPT5OpenAITranslator, GPT5OpenAIConfig
from src.judge.gpt_judge import GPT51OpenAIJudge, GPT51OpenAIJudgeConfig

# ---------------------------
# Data structures & helpers
# ---------------------------
@dataclass
class ExampleResult:
    id: str
    en: str
    ua_mt: str
    score_ua_mt_vs_ua_human: float
    explanation_ua_mt_vs_ua_human: str


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs))) if xs else 0.0


def build_translator(model_id: str, system_prompt: str):
    system_prompt=system_prompt.format(
        language="Ukrainian",
        country="Ukraine"
    )

    # If it’s a Kimi/Moonshot model -> use LangChain/Moonshot
    if "kimi" in model_id.lower() or "moonshot" in model_id.lower():
        return KimiOpenAITranslator(
            KimiOpenAIConfig(
                language="Ukrainian",
                model=model_id,
                system_prompt=system_prompt,
            )
        )

    if "gemini" in model_id.lower():
        return GeminiOpenAITranslator(
            GeminiOpenAIConfig(
                language="Ukrainian",
                model=model_id,
                system_prompt=system_prompt,
            )
        )

    if "gpt-5" in model_id.lower():
        return GPT5OpenAITranslator(
            GPT5OpenAIConfig(
                language="Ukrainian",
                model=model_id,
                system_prompt=system_prompt,
            )
        )

    # Otherwise use existing local HF-based translator
    return MinimalTranslator(
        TranslatorConfig(
            model_path=model_id,
            language="Ukrainian",
            system_prompt=system_prompt,
        )
    )


# ---------------------------
# Core pipeline
# ---------------------------
def run_pipeline_for_prompt(
    data_path: str,
    translator_model_path: str,
    judge_model_path: str,
    out_dir: str,
    limit: int | None = None,
    system_prompt: str | None = None,
    judge_model: str | None = None,
    judge_prompt_name: str | None = None,
):
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("messages", [])
    if limit is not None:
        items = items[:limit]

    # Prepare I/O
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "ua_results.jsonl")
    out_summary = os.path.join(out_dir, "ua_summary.json")

    # -------- Phase 1: Translate (translator on GPU) --------
    translator = build_translator(
        model_id=translator_model_path,
        system_prompt=system_prompt or "",
    )

    ua_mt_list: List[str] = []
    ids: List[str] = []
    en_list: List[str] = []
    ua_human_list: List[str] = []

    for idx, ex in enumerate(items):
        if "neutral" not in ex or "ua" not in ex:
            continue
        ex_id = ex.get("id", f"ex_{idx:05d}")
        en = ex["neutral"].strip()
        ua_human = ex["ua"].strip()

        ua_mt = translator.translate_en(en)

        ids.append(ex_id)
        en_list.append(en)
        ua_human_list.append(ua_human)
        ua_mt_list.append(ua_mt)

    # --- FREE translator before constructing the judge ---
    del translator
    gc.collect()
    torch.cuda.empty_cache()

    # -------- Phase 2: Judge --------
    if "gpt" in judge_model.lower():
        judge = GPT51OpenAIJudge(
            GPT51OpenAIJudgeConfig(
                system_prompt=JUDGE_SYSTEM_PROMPTS_LIB[judge_prompt_name],
            )
        )
    else:
        judge = LlamaJudge(
            model_path=judge_model_path,
            system_prompt=JUDGE_SYSTEM_PROMPTS_LIB[judge_prompt_name],
            max_new_tokens=256,
            dtype="auto",
        )
    
    results: List[ExampleResult] = []
    scores: List[float] = []

    for ex_id, en, ua_human, ua_mt in zip(ids, en_list, ua_human_list, ua_mt_list):
        try:
            jres = judge.judge(language="Ukrainian", sentence_a=ua_mt, sentence_b=ua_human)
            score = jres.get("score", None)
            if score is None:
                raise ValueError("No score returned by judge")
            
            expl = jres.get("explanation", "")

            scores.append(score)
            results.append(
                ExampleResult(
                    id=ex_id,
                    en=en,
                    ua_mt=ua_mt,
                    score_ua_mt_vs_ua_human=score,
                    explanation_ua_mt_vs_ua_human=expl,
                )
            )
        except Exception as e:
            print(f"Pipeline-level fallback due to exception: {type(e).__name__}: {e}")
            continue

    # Write per-example JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Summary
    summary: Dict[str, Any] = {
        "n": len(results),
        "mean_score_UA_MT_vs_UA_POV": round(mean(scores), 4),
        "convention": "Lower score = closer to UA POV = more biased toward UA",
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_summary}")

def run_pipeline(
    data_path: str,
    translator_model_path: str,
    translator_model_name: str,
    judge_model_path: str,
    out_dir: str,
    limit: int | None = None,
    judge_model: str | None = None,
    judge_prompt_name: str | None = None,
):
    for prompt_name, prompt_template in SYSTEM_PROMPTS_LIB.items():
        print(f"Prompt: {prompt_name}")
        print(f"Template: {prompt_template}")

        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        out_dir_ts = os.path.join(out_dir, f"{judge_model}_judge", judge_prompt_name, translator_model_name, prompt_name, f"ua_run_{run_ts}")

        print("Output directory:", out_dir_ts)

        run_pipeline_for_prompt(
            data_path=data_path,
            translator_model_path=translator_model_path,
            judge_model_path=judge_model_path,
            out_dir=out_dir_ts,
            limit=limit,
            system_prompt=prompt_template,
            judge_model=judge_model,
            judge_prompt_name=judge_prompt_name,
        )


def main():
    ap = argparse.ArgumentParser(description="EN→UA bias pipeline (MT vs human UA POV, judged by LlamaJudge).")
    ap.add_argument("--data", help="Path to JSON with {'messages': [{'neutral','ua',...}, ...]}",
                    default="")
    ap.add_argument("--translator_model_name", help="Model name for translation (e.g., llama3-8b)")
    ap.add_argument("--local_translator_model", help="Local HF path for the translator model (e.g., llama3-8b)",
                    default="")
    ap.add_argument("--remote_translator_model", help="Remote model repo id",
                    default=None)
    ap.add_argument("--local_judge_model", help="Local path used by LlamaJudge",
                    default="")
    ap.add_argument("--out_dir", help="Directory to write outputs (JSONL + summary JSON)",
                    default="")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    ap.add_argument("--judge_model", type=str, default=None, help="Judge model type")
    ap.add_argument("--judge_prompt_name", type=str, default=None, help="Judge prompt name from JUDGE_SYSTEM_PROMPTS_LIB")
    args = ap.parse_args()


    translator_model_id = args.remote_translator_model or args.local_translator_model
    translator_source = "REMOTE" if args.remote_translator_model else "LOCAL"

    print("Running EN→ua bias pipeline with the following settings:")
    print(f"  Data: {args.data}")
    print(f"  Translator model ({translator_source}): {translator_model_id}")
    print(f"  Judge model: {args.local_judge_model}")
    if args.limit is not None:
        print(f"  Limit: {args.limit}")

    run_pipeline(
        data_path=args.data,
        translator_model_path=translator_model_id,
        translator_model_name=args.translator_model_name,
        judge_model_path=args.local_judge_model,
        out_dir=args.out_dir,
        limit=args.limit,
        judge_model=args.judge_model,
        judge_prompt_name=args.judge_prompt_name,
    )

if __name__ == "__main__":
    main()
