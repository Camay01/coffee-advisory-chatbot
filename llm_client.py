"""
llm_client.py — Thin wrapper around Ollama for all LLM calls.
Single point of control for model, temperature, and error handling.
"""

import ollama
from config import OLLAMA_MODEL


def llm_call(system: str, user: str, num_predict: int = 512) -> str:
    """
    Execute a single Ollama chat call.
    Always temperature=0 for deterministic, reproducible responses.
    Returns the model's reply string, or an ERROR: ... string on failure.
    """
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": 0, "num_predict": num_predict},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"
