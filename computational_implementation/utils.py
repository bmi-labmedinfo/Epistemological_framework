from typing import List, Optional, Dict, Any, Literal, TypedDict
import json
import pandas as pd
from json import JSONDecodeError
import ollama
import re

class Config:
    #GLOBAL
    BASE_DIR = ".../computational_implementation/results"
    RESULT_DIR = ""
    TOP_K = 10
    CASES = pd.read_excel(".../computational_implementation/cases/Cases_NEMJ.xlsx")
    MODEL = "gpt-oss:20b"
    MODEL_DIR = "gpt"
    SUFFIX = " (plausible)"

    def set_result_dir(self, result_dir):
        self.RESULT_DIR = result_dir

    def set_top_k(self, k):
        self.TOP_K = k

    def set_cases(self, cases):
        self.CASES = cases

    def set_model(self, model_name):
        self.MODEL = model_name

    def set_model_dir(self, model_dir):
        self.MODEL_DIR = model_dir


def call_llm(
    messages: List[Dict[str, Any]],
    model: str,
    response_format,
    max_retries: int = 10
):
    """
    Provider-agnostic wrapper to call a Large Language Model.
    - Default implementation: Ollama
    - Extendable: allows users to implement any provider by creating a custom function
    - Output: JSON conforming to the `response_format` schema
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
             response = ollama.chat(
                 model=model,
                 messages=messages,
                 format=response_format.model_json_schema()
             )
             return json.loads(response.message.content)

        except JSONDecodeError as e:
            last_exception = e
            print(f"[Attempt {attempt}/{max_retries}] invalid JSON, retry...")
            continue

        except Exception as e:
            raise e

    raise RuntimeError(
        f"Failed after {max_retries} retries: invalid JSON"
    ) from last_exception


def _norm_diag(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip().lower())


def reorder_hypotheses_like(input_diagnoses, output_hypotheses, key="diagnosis"):
    """
    Returns output_hypotheses reordered to match input_diagnoses.
    """
    by_diag = { _norm_diag(h.get(key, "")): h for h in (output_hypotheses or []) }

    ordered = []
    for d in input_diagnoses:
        k = _norm_diag(d)
        if k in by_diag:
            ordered.append(by_diag.pop(k))

    # Appends any extra hypotheses that the LLM may have invented
    ordered.extend(by_diag.values())
    return ordered