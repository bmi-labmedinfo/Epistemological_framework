import json
import datetime
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from computational_implementation.utils import call_llm
from typing_extensions import Annotated



# OUTPUT RANKING
class Rank(BaseModel):
    rank: int = Field(...,description="Ranking position of the hypothesis: 1 = most plausible, 2 = second most plausible, etc. No ties.")
    parsimony : str = Field(...,description="Prefer hypotheses that require minimal therapeutic intervention rather than multiple interventions, provided they achieve equivalent therapeutic outcomes.")
    danger: str = Field(...,description="Assign higher priority to hypotheses where delayed diagnosis or treatment would result in greater risk or harm to the patient.")
    cost: str = Field(..., description="Prefer hypotheses whose confirmation and management require fewer amount of money, time, labor, or other expense.")
    curability: str = Field(..., description="Prefer hypothesis such that curing or healing is possible.")

def ranked_diagnoses(diagnoses):
    DiagnosisEnum = Enum(
        "DiagnosisEnum",
        {f"{i}": dx for i, dx in enumerate(diagnoses)}
    )
    class RankedDiagnosticHypothesis(BaseModel):
        diagnosis: DiagnosisEnum = Field(..., description="The diagnosis.")
        position: Rank = Field(..., description="The ranking position of the hypothesis.")

    class RankedHypothesesList(BaseModel):
        hypotheses: Annotated[
            List[RankedDiagnosticHypothesis],
            Field(min_length=10, max_length=10)
            ] = Field(
        description="List of diagnostic hypotheses ranked by plausibility (rank 1 = most plausible).")
    return RankedHypothesesList


def summarize_evaluated_findings(findings):
    confirmed = []
    contradicted = []
    not_observed = []

    for f in findings or []:
        desc = f.get("description", "")
        comm = f.get("comment", "")
        line = f"- {desc}" + (f" :: {comm}" if comm else "")

        st = (f.get("status") or "").lower()
        if st == "confirmed":
            confirmed.append(line)
        elif st == "contradicted":
            contradicted.append(line)
        else:  # not_observed / unknown / other
            not_observed.append(line)

    return confirmed, contradicted, not_observed


def query_rank_hypotheses(
    case: str,
    abduction_output: Dict[str, Any],
    model: str,
    iteration: int = 0,
    ):
    hypotheses = abduction_output.get("hypotheses", [])

    # Detect PRE vs POST induction
    is_post_induction = any(("evaluation" in h) or ("evaluated_findings" in h) for h in hypotheses)
    ranking_context = "POST_INDUCTION" if is_post_induction else "PRE_INDUCTION"

    focused_lines = []
    for i, h in enumerate(hypotheses, start=1):
        diag = h.get("diagnosis", "")

        if is_post_induction:
            evaluation = h.get("evaluation", "unknown")
            term = h.get("termination_recommendation", "unknown")
            expl = h.get("explanation", "")
            findings = h.get("evaluated_findings", [])

            conf, contr, miss = summarize_evaluated_findings(findings)

            focused_lines.append(
                f"H{i}: diagnosis = '{diag}'\n"
                f"    evaluation: {evaluation}\n"
                f"    termination_recommendation: {term}\n"
                f"    explanation: {expl}\n"
                f"    evaluated_findings_summary:\n"
                f"      confirmed ({len(conf)}):\n" + ("\n".join(conf) if conf else "      - (none)") + "\n"
                f"      contradicted ({len(contr)}):\n" + ("\n".join(contr) if contr else "      - (none)") + "\n"
                f"      not_observed ({len(miss)}):\n" + ("\n".join(miss) if miss else "      - (none)") + "\n"
            )
        else:
            supp = h.get("supporting_features", [])
            focused_lines.append(
                f"H{i}: diagnosis = '{diag}'\n"
                f"    supporting_features: {supp}\n"
            )

    focused_block = "\n".join(focused_lines)

    system_prompt_ranking = (
    "You are a **clinical reasoning and triage-ranking system**. Your task is to **rank an existing list of diagnostic hypotheses** produced by prior steps.\n\n"

    "## Inputs\n"
    "1) The full clinical presentation (note/event narrative)\n"
    "2) A list of diagnostic hypotheses. Each hypothesis may include:\n"
    "   - Abduction-style fields (e.g., supporting_features)\n"
    "   - Induction-style evidence fields (e.g., evaluation, explanation, termination_recommendation, evaluated_findings)\n\n"

    "## Output (STRICT)\n"
    "- Produce output that **exactly matches the provided schema** (field names, nesting, data types).\n"
    "- Output **only** the structured result—no prose, no extra keys, no commentary.\n"
    "- You must include **every hypothesis** from the provided list (none omitted, none added).\n"
    "- Do **not** alter hypothesis wording/content except as strictly required by the schema.\n\n"

    "## Core task\n"
    "- Assign each hypothesis an **ordinal rank**: 1 = highest priority / best-supported and/or most urgent to test, 2 = next, etc.\n"
    "- Provide, for each hypothesis, **criterion-specific justifications** for:\n"
    "  - **Parsimony**\n"
    "  - **Danger**\n"
    "  - **Cost**\n"
    "  - **Curability**\n\n"

    "## Ranking criteria (apply ALL, explicitly)\n"
    "Use the clinical presentation to evaluate each hypothesis under these preferences:\n"
    "- **Parsimony**: prefer simpler explanations/management when outcomes are equivalent.\n"
    "- **Danger**: prioritize hypotheses where delay could cause greater harm.\n"
    "- **Cost**: prefer hypotheses whose confirmation/management require fewer tests/time/money.\n"
    "- **Curability**: prefer hypotheses where cure/healing is feasible (all else equal).\n\n"

    "## Evidence integration rule\n"
    "- If **induction evidence fields** are provided (evaluation / evaluated_findings / termination_recommendation), you MUST incorporate them.\n"
    "- Hypotheses marked **plausible** with multiple **confirmed** findings should generally rank higher than those marked **refuted** or with **contradicted** findings,\n"
    "  unless **Danger** strongly overrides.\n"
    "- If supports are similar, break ties using **Danger first**, then **Parsimony**, then **Cost**, then **Curability**.\n\n"

    "## Prohibited actions\n"
    "- Do not generate free text outside the schema.\n"
    "- Do not introduce new data, new features, or new hypotheses.\n"
    )

    user_content = (
        f"RANKING_CONTEXT: {ranking_context}\n"
        "Clinical case narrative:\n"
        "------------------------\n"
        f"{case}\n\n"
        "Provided diagnostic hypotheses:\n"
        "------------------------------\n"
        f"{focused_block}\n\n"
        "Task: Using ONLY the clinical presentation and the hypotheses above, rank each hypothesis.\n"
        "- Assign an ordinal rank (1 = highest priority).\n"
        "- Provide parsimony, danger, cost, and curability explanations.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt_ranking},
        {"role": "user", "content": user_content},
    ]

    focused_diagnoses = [h.get("diagnosis", "") for h in hypotheses]

    answer = call_llm(
        messages=messages,
        model=model,
        response_format=ranked_diagnoses(focused_diagnoses),
    )

    ranked_answer = {"hypotheses": sorted(answer["hypotheses"], key=lambda d: d["position"]["rank"])}

    return ranked_answer