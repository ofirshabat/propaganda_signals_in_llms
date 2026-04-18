from dataclasses import dataclass
import os
import time
import sys
from openai import OpenAI


@dataclass
class GPT5OpenAIConfig:
    language: str
    model: str = "gpt-5"
    temperature: float = 0.4
    max_output_tokens: int = 8192
    system_prompt: str | None = None
    reasoning_effort: str | None = None  # e.g. "low", "medium", "high" (optional)


class GPT5OpenAITranslator:
    """
    Translator that uses OpenAI's Responses API.
    API-compatible with MinimalTranslator: translate_en(text) -> str.
    """

    def __init__(self, cfg: GPT5OpenAIConfig):
        self.cfg = cfg

        api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)

        if self.cfg.system_prompt is None:
            raise ValueError("system_prompt must be provided in GPT5OpenAIConfig")

    def _build_input(self, en_sentence: str):
        user_prompt = (
            f"Translate the following text into {self.cfg.language}.\n"
            f"Text: {en_sentence.strip()}\n"
            "Output only the translation.\n"
        )

        return [
            {"role": "developer", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def translate_en(self, en_sentence: str) -> str:
        time.sleep(7)
        input_items = self._build_input(en_sentence)

        kwargs = dict(
            model=self.cfg.model,
            input=input_items,
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_output_tokens,
        )
        if self.cfg.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.cfg.reasoning_effort}

        response = self.client.responses.create(**kwargs)

        print("\n--- FULL API RESPONSE ---")
        print(response.model_dump_json(indent=2))
        sys.stdout.flush()

        out = getattr(response, "output_text", "") or ""
        print("\n--- TRANSLATION OUTPUT ---")
        print(out)
        sys.stdout.flush()
        return out.strip().strip('"').strip()
