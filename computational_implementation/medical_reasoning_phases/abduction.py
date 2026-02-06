import json
import random
import datetime
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from computational_implementation.utils import call_llm, Config

config = Config()

# OUTPUT ABDUCTION
class DiagnosticHypothesis(BaseModel):
    diagnosis: str = Field(..., description="Short name of the diagnostic hypothesis or disease entity.")
    explanation: str = Field(...,description="Short rationale connecting this hypothesis to the abstracted features and the clinical context.")
    supporting_features: List[str] = Field(..., description="List of feature.finding strings used as positive evidence for this hypothesis.")

class AbductionHypothesesList(BaseModel):
    hypotheses: List[DiagnosticHypothesis] = Field(
        description="Set of diagnostic hypotheses generated through abduction phase")


def query_abduction_unfocused(case: str, abstraction_output: List[str], model: str, max_unfocused: int = config.TOP_K, iteration:int=0) -> Dict[str, Any]:
    """
    Unfocused abduction phase (initial hypothesis space):
    - Output: AbductionHypothesesList with ONLY unfocused hypotheses
    """
    features = abstraction_output.copy()

    features_lines = []
    for i, f in enumerate(features, start=1):
        finding = f.get("finding", "")
        explanation = f.get("explanation", "")
        features_lines.append(f"{i}. {finding} — {explanation}")
    features_block = "\n".join(features_lines)

    # PROMPT ABDUCTION
    # UNFOCUSED
    system_prompt_abduction_unfocused = (
    "You are a **clinical reasoning system** specialized in **unfocused abduction**.\n"
    "Your task is to propose an initial, broad **diagnostic hypothesis space** from:\n"
    "1) the full clinical narrative/event, and\n"
    "2) a structured set of **abstracted clinical features**.\n\n"

    "## Objective\n"
    f"Generate a **concise, structured list of up to {max_unfocused} broad diagnostic hypotheses**.\n"
    "These hypotheses should represent **diagnostic families** (organ/system categories, syndrome groupings, major disease classes) that could plausibly "
    "explain the problem features.\n\n"

    "## Output (STRICT)\n"
    "- Return output **only** in the provided schema (exact keys, nesting, and types). No extra prose.\n"
    "- Each hypothesis MUST include:\n"
    "  - a short family-level label\n"
    "  - a brief rationale\n"
    "  - an explicit list of **supporting abstracted features** (by reference/ID or exact feature text as required by the schema)\n\n"

    "## Hypothesis construction rules\n"
    "- Use the abstracted features as **problem features to be explained**.\n"
    "- Perform **unfocused abduction**: propose broad, high-coverage families rather than specific named diseases.\n"
    "- Avoid over-fragmentation: prefer fewer, higher-level buckets that cover multiple related etiologies.\n"
    "- Exclude hypotheses that are clearly trivial, redundant, or clinically irrelevant.\n\n"

    "## Handling existing hypotheses\n"
    "- If the input already includes one or more hypotheses, propose **additional hypotheses that are meaningfully distinct**.\n"
    "- Do **not** repeat, rephrase, or minimally vary existing items.\n"
    "- Expand laterally into alternative organ systems, mechanism classes, exposures, or syndrome families.\n\n"

    "## Specificity constraint\n"
    "- Do **not** output detailed or highly specific diagnoses here.\n"
    "- Reserve specific disease naming for the **focused abduction** phase.\n\n"

    "## Evidence & uncertainty\n"
    "- Use **only** information present in the narrative and abstracted features.\n"
    "- Do **not** invent, modify, or reinterpret data.\n"
    "- Treat outputs as **hypotheses**, not conclusions; keep uncertainty explicit.\n"
    "- Do not claim confirmation (no definitive language like \"this is\").\n\n"

    "## Feature linking requirements\n"
    "- Every hypothesis must cite the subset of abstracted features that support it.\n"
    "- Do not cite features that do not actually support the hypothesis.\n"
    )

    user_content = (
        "Clinical case narrative:\n"
        "------------------------\n"
        f"{case}\n\n"
        "Abstracted clinical features (from Abstraction phase):\n"
        "------------------------------------------------------\n"
        f"{features_block}\n\n"
        "Task: Using ONLY the abstracted features above as clinical evidences, "
        "perform the **unfocused abduction** phase.\n"
        f"- You must generate {max_unfocused} **broad** diagnostic hypotheses.\n"
        "- For each hypothesis, fill the JSON fields according to the schema (diagnosis, explanation, supporting_features).\n"
        "- Prefer broad diagnostic families and syndrome-level categories rather than very specific diseases.\n"
        "- In supporting_features, reference the *finding* strings exactly as given above.\n\n"
        "Return ONLY valid JSON, with no additional commentary."
    )

    messages = [
        {"role": "system", "content": system_prompt_abduction_unfocused},
        {"role": "user", "content": user_content},
    ]
    answer = call_llm(messages=messages, model=model, response_format=AbductionHypothesesList)

    return answer


def query_abduction_focused(case: str, abstraction_output: List[Any], unfocused_output: Dict[str, Any], model: str, max_focused: int = config.TOP_K, iteration:int=0) -> Dict[str, Any]:
    """
    Focused abduction phase (refinements / competitors / complements):
    - Input: case, features, and unfocused hypotheses
    - Output: AbductionHypothesesList with ONLY focused hypotheses
    """
    # Features block
    features = abstraction_output.copy()
    features_lines = []
    for i, f in enumerate(features, start=1):
        finding = f.get("finding", "")
        explanation = f.get("explanation", "")
        features_lines.append(f"{i}. {finding} — {explanation}")
    features_block = "\n".join(features_lines)

    # Unfocused hypotheses block
    unfocused_hyp = unfocused_output.get("hypotheses", [])
    unfocused_lines = []
    for i, h in enumerate(unfocused_hyp, start=1):
        diag = h.get("diagnosis", "")
        supp = h.get("supporting_features", [])
        unfocused_lines.append(
            f"U{i}: diagnosis = '{diag}'\n"
            f"    supporting_features: {supp}\n"
        )
    unfocused_block = "\n".join(unfocused_lines)
    user_content = (
        "Clinical case narrative:\n"
        "------------------------\n"
        f"{case}\n\n"
        "Abstracted clinical features (from Abstraction phase):\n"
        "------------------------------------------------------\n"
        f"{features_block}\n\n"
        "Unfocused diagnostic hypotheses (broad families from previous phase):\n"
        "---------------------------------------------------------------------\n"
        f"{unfocused_block}\n\n"
        "Task: Using ONLY the narrative, the abstracted features, and the unfocused hypotheses above, "
        "perform the **focused abduction** phase.\n"
        f"- You must generate {max_focused} **focused** diagnostic hypotheses.\n"
        "- Focused hypotheses should be more specific diagnoses that refine or complement the unfocused ones.\n"
        "- For each hypothesis, fill the JSON fields according to the schema "
        "(diagnosis, explanation, hypothesis_type, supporting_features).\n"
        "- In supporting_features reference the *finding* strings exactly as given above.\n\n"
        "Return ONLY valid JSON, with no additional commentary."
    )
    # FOCUSED
    system_prompt_abduction_focused = (
    "You are a **clinical reasoning system** focused on **focused abduction of diagnostic hypotheses**.\n\n"

    "## Inputs\n"
    "You will be given:\n"
    "1) A clinical presentation (free text)\n"
    "2) A **structured set of abstracted clinical features** (use wording exactly as provided)\n"
    "3) A set of **broad (unfocused) diagnostic hypotheses**\n\n"

    "## Task\n"
    f"Generate a **structured set of up to {max_focused} focused diagnostic hypotheses**.\n"
    "Focused hypotheses must be **more specific diagnostic entities** (or specific combinations) that:\n"
    "- **Refine** a broad hypothesis (a narrower subtype/etiology/mechanism)\n"
    "- **Compete** with a broad hypothesis (a specific alternative explanation within that space)\n"
    "- **Complement** a broad hypothesis (a specific co-existing diagnosis that also explains features)\n\n"

    "## Constraints (STRICT)\n"
    "- Output must **exactly match the provided schema** (keys, nesting, types). Output **only** the structured result.\n"
    "- Do **not** repeat broad categories already present in the unfocused hypotheses.\n"
    "- Do **not** rank, score, or order by likelihood.\n"
    "- Do **not** invent, infer, or modify clinical data. Use only what is present in the inputs.\n"
    "- Hypotheses are **not conclusions**; keep uncertainty explicit.\n\n"

    "## How to generate hypotheses\n"
    "- Treat abstracted features as **problem features to be explained**.\n"
    "- For each broad hypothesis, propose focused diagnoses that plausibly explain **some or all** of the problem features.\n"
    "- Prefer clinically recognized diagnostic labels (avoid vague descriptors like “infection” unless made specific).\n"
    "- If multiple focused diagnoses are essentially the same concept, **merge** into one best term rather than duplicating.\n\n"

    "## Evidence linkage requirement\n"
    "- For **each focused hypothesis**, explicitly list which abstracted features support it.\n"
    "- Use the feature findings **exactly as provided** (verbatim strings/IDs per schema).\n"
    "- If key evidence is absent or contradictory, represent that as uncertainty (per schema, or explicitly in the explanation field).\n\n"

    "## Coverage guidance\n"
    "- Aim for breadth across plausible focused diagnoses **within** the spaces opened by the broad hypotheses.\n"
    "- Avoid overly rare “zebra” diagnoses unless the provided features specifically suggest them.\n"
    )
    messages = [
        {"role": "system", "content": system_prompt_abduction_focused},
        {"role": "user", "content": user_content},
    ]

    answer = call_llm(messages=messages, model=model, response_format=AbductionHypothesesList)
    # Random choice of n=new_hypotheses abduction_focused_2 hypotheses
    if len(answer["hypotheses"]) > max_focused:
        answer = {"hypotheses":random.sample(answer["hypotheses"], max_focused)}

    return answer