import json
import datetime
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from computational_implementation.utils import call_llm
from typing_extensions import Annotated

# OUTPUT DEDUCTION
class ExpectedFinding(BaseModel):
    description: str = Field(...,
        description="A clinical consequence expected if the hypothesis is true (symptom/sign, test result, or evolution of the clinical course).",
    )
    kind: Literal["manifestation", "test_result", "clinical_course"] = Field(...,
        description="Type of expected consequence: 'manifestation' for signs/symptoms, 'test_result' for laboratory/imaging findings,'clinical_course' for expected temporal evolution.",
    )
    priority: Literal["high", "medium", "low"] = Field(...,
        description="Clinical priority of verifying this expected finding for testing the hypothesis (higher priority = more informative and/or urgent to check).",
    )

def deduct_diagnoses(diagnoses):
    DiagnosisEnum = Enum(
        "DiagnosisEnum",
        {f"{i}": dx for i, dx in enumerate(diagnoses)}
    )
    class HypothesisDeduction(BaseModel):
        diagnosis: DiagnosisEnum = Field(..., description="Short name of the diagnostic hypothesis or disease entity.")
        predicted_consequences: List[ExpectedFinding] = Field(..., description="List of expected clinical consequences (predictions) if this hypothesis is true.")

    class DeductionPlan(BaseModel):
        hypotheses: Annotated[
            List[HypothesisDeduction],
            Field(min_length=10, max_length=10)
            ] = Field(
            description="Set of hypotheses with their expected consequences, as produced by the deduction (prediction) phase.",
        )
    return DeductionPlan


def query_deduction(ranking_output: Dict[str, Any], model: str, iteration:int=0) -> Dict[str, Any]:
    """
    Deduction phase of the STModel (prediction of expected findings):
    - Input:
        - case: raw clinical case narrative
        - ranking_output: dict with key 'hypotheses' (RankedHypothesesList)
    - Output:
        - DeductionPlan: list of hypotheses with their predicted consequences
    """

    ranked_hypotheses = ranking_output.get("hypotheses", [])
    ranked_hypotheses = sorted(
        ranked_hypotheses,
        key=lambda h: h.get("position", {}).get("rank", 10**9)
    )
    input_order = [h.get("diagnosis", "") for h in ranked_hypotheses]


    # Hypotheses block
    hyp_lines = []
    for i, h in enumerate(ranked_hypotheses, start=1):
        diag = h.get("diagnosis", "")
        score = h.get("score", None)
        rank = h.get("rank", None)
        reas = h.get("reasoning_score", [])

        hyp_lines.append(
            f"H{i}: diagnosis = '{diag}', score = {score}, rank = {rank}\n"
            f"    reasoning_score: {reas}\n"
        )
    hypotheses_block = "\n".join(hyp_lines)

    system_prompt_deduction = (
    "You are a **clinical reasoning system** specialized in **deducing expected clinical consequences** from given diagnostic hypotheses.\n\n"

    "## Input\n"
    "- You will receive a list of **ranked diagnostic hypotheses** (already ranked).\n\n"

    "## Task\n"
    "For **each** hypothesis, produce a compact set of **expected findings/consequences** that would be likely to occur **if that hypothesis were true**.\n"
    "These expected consequences may include:\n"
    "- Symptoms/signs and clinical trajectory\n"
    "- Vital sign patterns\n"
    "- Targeted physical exam findings\n"
    "- Key lab abnormalities (include directionality: ↑/↓)\n"
    "- Imaging/ECG patterns\n"
    "- Microbiology/pathology results\n"
    "- Response (or lack of response) to specific treatments\n"
    "- Short-term complications relevant to safety/management\n\n"

    "## Output (STRICT)\n"
    "- Output must **exactly match the provided schema** (field names, nesting, data types).\n"
    "- Output **only** the structured result—no explanation, no extra keys, no reformatting.\n"
    "- Do **not** add, remove, rewrite, merge, or re-rank hypotheses.\n\n"

    "## Selection rules (keep it small and useful)\n"
    "For each hypothesis, include a **small, focused** set of expected consequences (typically 5–10), prioritizing:\n"
    "- Findings that are **highly discriminative** (help confirm/refute)\n"
    "- Findings that are **clinically actionable** (affect immediate safety/management)\n"
    "- Findings that are commonly observed and/or time-linked (early vs late when relevant)\n\n"

    "## Clinical grounding\n"
    "- Use medically plausible deductions consistent with standard clinical knowledge.\n"
    "- Do **not** invent new diagnoses or propose alternative hypotheses.\n"
    "- Do **not** perform any re-scoring or ranking.\n\n"

    "## Expression guidelines\n"
    "- Use **ALL** the hypotheses provided in the input.\n"
    "- Mention the hypotheses only once.\n"
    "- Phrase consequences as **testable expectations** (e.g., \"elevated troponin\", \"unilateral leg swelling\", \"CXR infiltrate\").\n"
    "- Include directionality (↑/↓), laterality, and timing when helpful.\n"
    "- Avoid vague statements (e.g., \"patient feels unwell\") unless the hypothesis truly implies it.\n"
    )

    user_content = (
        "Ranked diagnostic hypotheses (from Ranking phase):\n"
        "--------------------------------------------------\n"
        f"{hypotheses_block}\n\n"
        "Task: Using ONLY the information above, perform the **deduction** phase.\n"
        "- For each focused hypothesis, derive a small set of expected clinical consequences "
        "  (symptoms/signs, test results, or clinical course elements) that should hold if the hypothesis is true.\n"
        "- Use the JSON schema (DeductionPlan -> HypothesisDeduction -> ExpectedFinding).\n"
        "- Include both expectations that are already observed in the case and expectations that are clinically important "
        "  but not yet observed (these will correspond to possible further tests).\n"
        "\n"
        "Return ONLY valid JSON, with no additional commentary.\n"
        "Use and report **ALL** the hypotheses provided in the input.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt_deduction},
        {"role": "user", "content": user_content},
    ]

    answer = call_llm(messages=messages, model=model,
                      response_format=deduct_diagnoses([x["diagnosis"] for x in ranked_hypotheses]),)

    return answer
