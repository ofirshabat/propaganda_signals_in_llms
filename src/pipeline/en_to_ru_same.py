#!/usr/bin/env python3
# en_to_ru_same.py  (now supports reading precomputed MT and re-judging)

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import gc
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

# --- ensure we can import from the project root when running from translation/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.translator.prompts.prompts import SYSTEM_PROMPTS_LIB
from src.judge.judge import LlamaJudge
from src.judge.prompts.prompts import JUDGE_SYSTEM_PROMPTS_LIB
from src.judge.gpt_judge import GPT51OpenAIJudge, GPT51OpenAIJudgeConfig

# ---------------------------
# Data structures & helpers
# ---------------------------
@dataclass
class ExampleResultGeneric:
    id: str
    en: str
    mt: str
    score: float
    explanation: str


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs))) if xs else 0.0


def lang_cfg(lang: str) -> Tuple[str, str]:
    """
    Returns (PrettyName, key_prefix) for lang.
    key_prefix is used for json keys like '{prefix}_mt' and '{prefix}_results.jsonl'
    """
    lang = lang.lower().strip()
    if lang == "ru":
        return "Russian", "ru"
    if lang == "ua":
        return "Ukrainian", "ua"
    raise ValueError(f"Unsupported lang={lang}. Use 'ru' or 'ua'.")



def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _latest_run_dir(prompt_dir: str, run_prefix: str) -> Optional[str]:
    """
    Pick latest run directory by lexicographic sort (works with YYYY-MM-DD_HH-MM-SS timestamps).
    """
    if not os.path.isdir(prompt_dir):
        return None
    cands = [d for d in os.listdir(prompt_dir) if os.path.isdir(os.path.join(prompt_dir, d)) and d.startswith(run_prefix)]
    if not cands:
        return None
    cands.sort()
    return os.path.join(prompt_dir, cands[-1])


def load_precomputed_mt_for_prompt(
    precomputed_mt_root: str,
    prompt_name: str,
    lang_prefix: str,
    mt_key: str,
    jsonl_filename: Optional[str] = None,
    run_prefix_override: Optional[str] = None,
) -> Dict[str, str]:
    """
    Expected structure:
      <precomputed_mt_root>/<prompt_name>/<lang_prefix>_run_YYYY-MM-DD_HH-MM-SS/<lang_prefix>_results.jsonl

    Returns: {ex_id: mt_string}
    """
    run_prefix = run_prefix_override or f"{lang_prefix}_run_"
    prompt_dir = os.path.join(precomputed_mt_root, prompt_name)
    latest = _latest_run_dir(prompt_dir, run_prefix=run_prefix)
    if latest is None:
        raise FileNotFoundError(f"No run dirs found for prompt='{prompt_name}' under: {prompt_dir} (prefix='{run_prefix}')")

    jsonl_name = jsonl_filename or f"{lang_prefix}_results.jsonl"
    jsonl_path = os.path.join(latest, jsonl_name)
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Missing JSONL: {jsonl_path}")

    rows = _read_jsonl(jsonl_path)
    mt_by_id: Dict[str, str] = {}

    # robust id matching: allow either 'ex_00001' or 'prompt_ex_00001'
    for r in rows:
        rid = r.get("id")
        if not rid:
            continue
        mt_val = r.get(mt_key)
        if mt_val is None:
            # fallback: some older runs might store 'ru_mt'/'ua_mt' or 'mt'
            mt_val = r.get(f"{lang_prefix}_mt", r.get("mt"))
        if mt_val is None:
            continue

        mt_by_id[rid] = str(mt_val)

        # also store normalized form if it has '<prompt_name>_' prefix
        pref = prompt_name + "_"
        if rid.startswith(pref):
            mt_by_id[rid[len(pref):]] = str(mt_val)

    return mt_by_id


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
    lang: str = "ru",

    precomputed_mt_root: Optional[str] = None,
    precomputed_jsonl_filename: Optional[str] = None,
    precomputed_mt_key: Optional[str] = None,
    precomputed_run_prefix: Optional[str] = None,
):
    language_pretty, lang_prefix = lang_cfg(lang)
    gt_key = lang_prefix  # expects payload messages contain 'ru' or 'ua'
    mt_key = precomputed_mt_key or f"{lang_prefix}_mt"

    print("Using judge prompt:")
    print(JUDGE_SYSTEM_PROMPTS_LIB[judge_prompt_name].format(language=language_pretty))

    # Load data (ground truth lives here)
    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("messages", [])
    if limit is not None:
        items = items[:limit]

    # Prepare I/O
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{lang_prefix}_results.jsonl")
    out_summary = os.path.join(out_dir, f"{lang_prefix}_summary.json")

    ids: List[str] = []
    en_list: List[str] = []
    gt_list: List[str] = []
    mt_list: List[str] = []

    # -------- Phase 1: Get MT (either translate OR read precomputed) --------
    if precomputed_mt_root is not None:
        prompt_name = os.path.basename(os.path.dirname(out_dir))

        mt_by_id = load_precomputed_mt_for_prompt(
            precomputed_mt_root=precomputed_mt_root,
            prompt_name=prompt_name,
            lang_prefix=lang_prefix,
            mt_key=mt_key,
            jsonl_filename=precomputed_jsonl_filename,
            run_prefix_override=precomputed_run_prefix,
        )

        for idx, ex in enumerate(items):
            if "neutral" not in ex or gt_key not in ex:
                continue
            ex_id = ex.get("id", f"ex_{idx:05d}")
            en = ex["neutral"].strip()
            gt = ex[gt_key].strip()

            mt = mt_by_id.get(ex_id)
            if mt is None:
                # also try prefixed form (some runs store prompt_ex_XXXXX)
                mt = mt_by_id.get(f"{prompt_name}_{ex_id}")

            if mt is None:
                # skip if this example wasn't translated in the stored run
                continue

            ids.append(ex_id)
            en_list.append(en)
            gt_list.append(gt)
            mt_list.append(mt)

        # no translator to free
    else:
        raise ValueError("Translation path is not implemented in this snippet. Please provide precomputed_mt_root.")

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

    results: List[ExampleResultGeneric] = []
    scores: List[float] = []

    for ex_id, en, gt, mt in zip(ids, en_list, gt_list, mt_list):
        try:
            print(f"Judging ex_id={ex_id}...")
            print(f"  EN: {en}"
                  f"\n  GT: {gt}"
                  f"\n  MT: {mt}", flush=True)

            jres = judge.judge(language=language_pretty, sentence_a=mt, sentence_b=gt)
            score = jres.get("score", None)
            if score is None:
                raise ValueError("No score returned by judge")

            expl = jres.get("explanation", "")
            scores.append(float(score))
            results.append(
                ExampleResultGeneric(
                    id=ex_id,
                    en=en,
                    mt=mt,
                    score=float(score),
                    explanation=str(expl),
                )
            )
        except Exception as e:
            print(f"Pipeline-level fallback due to exception: {type(e).__name__}: {e}")
            continue

    # Write per-example JSONL
    score_key = f"score_{lang_prefix}_mt_vs_{lang_prefix}_human"
    expl_key = f"explanation_{lang_prefix}_mt_vs_{lang_prefix}_human"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            row = {
                "id": r.id,
                "en": r.en,
                f"{lang_prefix}_mt": r.mt,
                score_key: r.score,
                expl_key: r.explanation,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    summary: Dict[str, Any] = {
        "n": len(results),
        f"mean_score_{lang_prefix}_MT_vs_{lang_prefix}_GT": round(mean(scores), 4),
        "convention": "Lower score = closer to GT (by the judge definition).",
        "mode": "precomputed_mt" if precomputed_mt_root is not None else "translate_then_judge",
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
    lang: str = "ru",

    precomputed_mt_root: Optional[str] = None,
    precomputed_jsonl_filename: Optional[str] = None,
    precomputed_mt_key: Optional[str] = None,
    precomputed_run_prefix: Optional[str] = None,
):
    _, lang_prefix = lang_cfg(lang)

    for prompt_name, prompt_template in SYSTEM_PROMPTS_LIB.items():
        print(f"Prompt: {prompt_name}")

        run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        out_dir_ts = os.path.join(
            out_dir,
            f"{judge_model}_judge",
            judge_prompt_name,
            translator_model_name,
            prompt_name,
            f"{lang_prefix}_run_{run_ts}",
        )

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
            lang=lang,
            precomputed_mt_root=precomputed_mt_root,
            precomputed_jsonl_filename=precomputed_jsonl_filename,
            precomputed_mt_key=precomputed_mt_key,
            precomputed_run_prefix=precomputed_run_prefix,
        )


def main():
    ap = argparse.ArgumentParser(
        description="MT→judge pipeline (translate or load precomputed MT, then judge vs GT from data JSON)."
    )
    ap.add_argument(
        "--data",
        help="Path to JSON with {'messages': [{'neutral','ru','ua',...}, ...]}",
        default="",
    )

    ap.add_argument("--lang", choices=["ru", "ua"], default="ru", help="Which GT to use from data (ru or ua).")

    ap.add_argument("--translator_model_name", help="Model name bucket used in output path (e.g., gemini-3-pro-preview)")
    ap.add_argument(
        "--local_translator_model",
        help="Local HF path for the translator model (used only if NOT using --precomputed_mt_root)",
        default="",
    )
    ap.add_argument("--remote_translator_model", help="Remote model repo id", default=None)

    ap.add_argument(
        "--local_judge_model",
        help="Local path used by LlamaJudge",
        default="",
    )
    ap.add_argument(
        "--out_dir",
        help="Directory to write outputs",
        default="",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    ap.add_argument("--judge_model", type=str, default=None, help="Judge model type string (e.g., llama3-8b)")
    ap.add_argument("--judge_prompt_name", type=str, default=None, help="Judge prompt name from JUDGE_SYSTEM_PROMPTS_LIB")

    ap.add_argument(
        "--precomputed_mt_root",
        type=str,
        default=None,
        help=(
            "If set, skip translation and read MT from this results tree. "
            "Expected: <root>/<prompt_name>/{lang}_run_*/{lang}_results.jsonl"
        ),
    )
    ap.add_argument(
        "--precomputed_jsonl_filename",
        type=str,
        default=None,
        help="Override JSONL filename inside the run dir (default: '{lang}_results.jsonl').",
    )
    ap.add_argument(
        "--precomputed_mt_key",
        type=str,
        default=None,
        help="Override MT field inside JSONL rows (default: '{lang}_mt').",
    )
    ap.add_argument(
        "--precomputed_run_prefix",
        type=str,
        default=None,
        help="Override run dir prefix (default: '{lang}_run_').",
    )

    args = ap.parse_args()

    translator_model_id = args.remote_translator_model or args.local_translator_model
    translator_source = "REMOTE" if args.remote_translator_model else "LOCAL"

    print("Running MT→judge pipeline with settings:")
    print(f"  Data: {args.data}")
    print(f"  Lang: {args.lang}")
    print(f"  Judge model path: {args.local_judge_model}")
    print(f"  Judge model type: {args.judge_model}")
    print(f"  Judge prompt: {args.judge_prompt_name}")
    if args.precomputed_mt_root:
        print(f"  Mode: precomputed_mt (root={args.precomputed_mt_root})")
    else:
        print(f"  Mode: translate_then_judge (translator {translator_source}: {translator_model_id})")
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
        lang=args.lang,
        precomputed_mt_root=args.precomputed_mt_root,
        precomputed_jsonl_filename=args.precomputed_jsonl_filename,
        precomputed_mt_key=args.precomputed_mt_key,
        precomputed_run_prefix=args.precomputed_run_prefix,
    )


if __name__ == "__main__":
    main()
