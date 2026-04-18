from dataclasses import dataclass
import os
import time
from openai import OpenAI

import sys

@dataclass
class GeminiOpenAIConfig:
    language: str
    model: str = "gemini-3-pro-preview"
    temperature: float = 0.4
    max_tokens: int = 8192
    system_prompt: str | None = None


class GeminiOpenAITranslator:
    """
    Translator that uses Gemini's OpenAI-compatible API.
    API-compatible with MinimalTranslator: translate_en(text) -> str.
    """

    def __init__(self, cfg: GeminiOpenAIConfig):
        self.cfg = cfg

        api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        if self.cfg.system_prompt is None:
            raise ValueError("system_prompt must be provided in GeminiOpenAIConfig")

    def _build_messages(self, en_sentence: str):
        user_prompt = (
            f"Translate the following text into {self.cfg.language}.\n"
            f"Text: {en_sentence.strip()}\n"
            "Output only the translation.\n"
        )
        return [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def translate_en(self, en_sentence: str) -> str:
        time.sleep(3)
        messages = self._build_messages(en_sentence)

        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        print("\n--- FULL API RESPONSE ---")
        print(response.model_dump_json(indent=2))
        sys.stdout.flush() # Force everything above to appear now


        out = response.choices[0].message.content or ""
        return out.strip().strip('"').strip()
