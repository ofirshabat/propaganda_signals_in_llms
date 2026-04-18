#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic.v1 import BaseModel, Field, validator


# ---------------------------
# Structured output schema
# ---------------------------
class JudgeResult(BaseModel):
    """Semantic distance judgment between two sentences."""
    score: float = Field(
        ...,
        description="Real number in [0,1]. 0 = identical; 0.5 = somewhat different; 1 = unrelated/contradictory."
    )
    explanation: str = Field(
        ...,
        description="Concise, neutral English reason for the score (1–2 short sentences)."
    )

    @validator("score")
    def clamp_score(cls, v):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Score must be finite.")
        return max(0.0, min(1.0, float(v)))

    @validator("explanation")
    def strip_explanation(cls, v):
        return v.strip()


@dataclass
class GPT51OpenAIJudgeConfig:
    model: str = "gpt-5.1"
    temperature: float = 0.0
    max_output_tokens: int = 256
    system_prompt: str | None = None
    reasoning_effort: str | None = None  # e.g. "low", "medium", "high" (optional)
    sleep_seconds: float = 0.0
    debug_print_full_response: bool = False


class GPT51OpenAIJudge:
    """
    LLM-as-a-judge using OpenAI's Responses API (gpt-5.1) with strict JSON output.

    Usage:
        judge = GPT51OpenAIJudge(GPT51OpenAIJudgeConfig(system_prompt="..."))
        out = judge.judge("Russian", "A", "B")      # -> {"score": float, "explanation": "..."}
        s = judge.judge_json("Russian", "A", "B")   # -> minified JSON string
    """

    _TEMPLATE = """{system}

Return ONLY a valid JSON object (no prose, no markdown, no extra keys), in the following format:
{{
  "score": <number in [0,1]>,
  "explanation": <1-2 short, neutral English sentences>
}}

Sentence A: {sentence_a}
Sentence B: {sentence_b}
"""

    _STRICT_TEMPLATE = """{system}

Return ONLY this JSON object (no prose, no markdown, no comments, no extra keys):
{{
  "score": 0.0,
  "explanation": "..."
}}

The value of "score" MUST be a real number in [0,1].
The value of "explanation" MUST be a short, neutral English reason (1–2 sentences).
Start with '{{' and end with '}}'. Nothing else.

Sentence A: {sentence_a}
Sentence B: {sentence_b}
"""

    def __init__(self, cfg: GPT51OpenAIJudgeConfig):
        self.cfg = cfg

        api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        if self.cfg.system_prompt is None:
            raise ValueError("system_prompt must be provided in GPT51OpenAIJudgeConfig")

        self.client = OpenAI(api_key=api_key)

    def _build_input(self, language: str, sentence_a: str, sentence_b: str, strict: bool) -> list[dict]:
        template = self._STRICT_TEMPLATE if strict else self._TEMPLATE
        user_prompt = template.format(
            system=self.cfg.system_prompt,
            language=language,
            sentence_a=sentence_a.strip(),
            sentence_b=sentence_b.strip(),
        )
        return [
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_json_obj(text: str) -> Dict[str, Any]:
        """
        Best-effort extraction of a single top-level JSON object from model output.
        """
        t = (text or "").strip()

        # First try direct JSON parse
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Fallback: slice from first '{' to last '}'
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in output.")

        candidate = t[start : end + 1].strip()
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError("Parsed JSON is not an object.")
        return obj

    def _call(self, input_items: list[dict]) -> str:
        if self.cfg.sleep_seconds and self.cfg.sleep_seconds > 0:
            time.sleep(float(self.cfg.sleep_seconds))

        kwargs: Dict[str, Any] = dict(
            model=self.cfg.model,
            input=input_items,
            temperature=float(self.cfg.temperature),
            max_output_tokens=int(self.cfg.max_output_tokens),
        )
        if self.cfg.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.cfg.reasoning_effort}

        response = self.client.responses.create(**kwargs)

        if self.cfg.debug_print_full_response:
            print("\n--- FULL API RESPONSE ---")
            print(response.model_dump_json(indent=2))

        out = getattr(response, "output_text", "") or ""
        return out.strip()

    def judge(self, language: str, sentence_a: str, sentence_b: str) -> Dict[str, Any]:
        """
        Returns a dict with keys: {"score": float, "explanation": str}.
        Performs one strict retry if parsing/validation fails.
        """
        # First attempt (non-strict)
        try:
            out_text = self._call(self._build_input(language, sentence_a, sentence_b, strict=False))
            obj = self._extract_json_obj(out_text)
            result = JudgeResult(**obj)
            return {"score": float(result.score), "explanation": result.explanation}
        except Exception:
            pass

        # Strict retry
        out_text = self._call(self._build_input(language, sentence_a, sentence_b, strict=True))
        obj = self._extract_json_obj(out_text)
        result = JudgeResult(**obj)
        return {"score": float(result.score), "explanation": result.explanation}

    def judge_json(self, language: str, sentence_a: str, sentence_b: str) -> str:
        """
        Same as `judge` but returns a minified JSON string.
        """
        out = self.judge(language, sentence_a, sentence_b)
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
