from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Minimal translator
# ---------------------------
@dataclass
class TranslatorConfig:
    model_path: str
    language: str
    dtype: str = "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.4
    top_p: float = 0.85
    top_k: int = 50
    system_prompt: str | None = None
    user_prompt: str | None = None

class MinimalTranslator:
    """
    Deterministic translator with minimal prompts to avoid adding framing/context.
    """
    def __init__(self, cfg: TranslatorConfig):
        self.language = cfg.language
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # No accelerate device_map so we can control device explicitly
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=(None if cfg.dtype == "auto" else getattr(torch, cfg.dtype)),
            device_map="cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        if cfg.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
        else:
            self.model.to("cpu")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            #device=0 if (cfg.device == "cuda" and torch.cuda.is_available()) else -1,
        )
        self.has_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    def _build_prompt(self, en_sentence: str) -> str:
        user = (
            f"Translate the following text into {self.language}.\n\n"
            f"Text: {en_sentence.strip()}\n"
            "Output only the translation.\n"
        )

        if self.has_chat_template:
            messages = [
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        raise NotImplementedError(
            "Non-chat templates are not implemented."
        )

    def translate_en(self, en_sentence: str) -> str:
        prompt = self._build_prompt(en_sentence)
        out = self.pipe(
            prompt,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
            use_cache=False if "internlm2" in self.cfg.model_path else True,
        )[0]["generated_text"]

        if out.startswith(prompt):
            out = out[len(prompt):]
        return out.strip().strip('“”"').strip()

