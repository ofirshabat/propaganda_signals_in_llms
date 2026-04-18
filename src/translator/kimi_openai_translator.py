from dataclasses import dataclass
import os
from openai import OpenAI
import time


@dataclass
class KimiOpenAIConfig:
    language: str
    model: str = "moonshot-v1-8k"
    temperature: float = 0.4
    max_tokens: int = 8192
    system_prompt: str | None = None
    user_prompt: str | None = None


class KimiOpenAITranslator:
    """
    Translator that uses Moonshot's OpenAI-compatible API.
    API-compatible with MinimalTranslator: translate_en(text) -> str.
    """

    def __init__(self, cfg: KimiOpenAIConfig):
        self.cfg = cfg

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY is not set")

        # Create the Moonshot OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
        )

        if self.cfg.system_prompt is None:
            raise ValueError("system_prompt must be provided in KimiOpenAIConfig")

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
        time.sleep(6)
        messages = self._build_messages(en_sentence)

        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )

        out = response.choices[0].message.content
        return out.strip().strip('"').strip()
