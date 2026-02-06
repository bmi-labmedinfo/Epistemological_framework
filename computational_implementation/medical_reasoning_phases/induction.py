import json
import datetime
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from computational_implementation.utils import call_llm
from typing_extensions import Annotated


# OUTPUT INDUCTION
class EvaluatedExpectedFinding(BaseModel):
    description: str = Field(..., description="The description of an expected finding, copied from the deduction phase.")
    status: Literal["confirmed", "not_observed", "contradicted"] = Field(...,
        description=(
            "'confirmed' if the expected finding clearly matches one or more abstracted features; "
            "'not_observed' if it is not mentioned in the case and would require additional tests; "
            "'contradicted' if the case contains evidence that clearly goes in the opposite direction."
        ),
    )
    comment: str = Field(...,description="Short explanation for the assigned status.")


def induct_diagnoses(diagnoses):
    DiagnosisEnum = Enum(
        "DiagnosisEnum",
        {f"{i}": dx for i, dx in enumerate(diagnoses)}
    )
    class EvaluatedHypothesis(BaseModel):
        diagnosis: DiagnosisEnum = Field(..., description="Short name of the diagnostic hypothesis or disease entity.")
        evaluation: Literal["plausible", "refuted"] = Field(...,
            description="Global inductive judgement on the hypothesis after matching expectations with observed findings: 'plausible' or 'refuted'",
        )
        explanation: str = Field(...,
            description="Short explanation for why the hypothesis is plausible or refuted, based on the match between expected and observed findings.",
        )
        termination_recommendation: Literal["sufficient_explanation_reached", "continue_testing", "reopen_abduction"] = Field(...,
            description=(
                "Recommendation about the next step for this hypothesis: "
                "'sufficient_explanation_reached' if the diagnostic process could reasonably stop here; "
                "'continue_testing' if further tests on this hypothesis are warranted; "
                "'reopen_abduction' if this hypothesis is weak and a new abduction step should be considered."
            )
        )
        evaluated_findings: List[EvaluatedExpectedFinding] = Field(...,
            description="List of expected findings for this hypothesis, each evaluated against the observed data."
        )

    class InductionResult(BaseModel):
        hypotheses: Annotated[
            List[EvaluatedHypothesis],
            Field(min_length=10, max_length=10)
            ] = Field(
            description="Set of diagnostic hypotheses evaluated by induction after the deduction phase (matching expected vs observed findings and deciding on next steps).",
        )
    return InductionResult



def query_induction(case: str, deduction_output: Dict[str, Any], model: str,iteration:int=0) -> tuple[Any,Any]:
    """
    Induction phase of the STModel (testing hypotheses):
    - Input:
        - case: raw clinical case narrative
        - deduction_output: dict with key 'hypotheses' (DeductionPlan)
    - Output:
        - InductionResult: hypotheses evaluated as plausible or refuted,
          with evaluated expected findings and a termination recommendation.
    """
    # --- Deduction (expected consequences) block ---
    ded_hypotheses = deduction_output.get("hypotheses", [])
    ded_lines = []
    for i, h in enumerate(ded_hypotheses, start=1):
        diag = h.get("diagnosis", "")

        consequences = h.get("predicted_consequences", [])
        inner = []
        for j, c in enumerate(consequences, start=1):
            desc = c.get("description", "")
            kind = c.get("kind", "")
            priority = c.get("priority", "")
            inner.append(
                f"    E{j}: {desc} [kind={kind}, priority={priority}]"
            )
        inner_block = "\n".join(inner)
        ded_lines.append(
            f"D{i}: hypothesis = '{diag}'\n{inner_block}"
        )
    deduction_block = "\n".join(ded_lines)

    system_prompt_induction = (
    "You are a **clinical reasoning system** specialized in the **induction phase**: evaluating diagnostic hypotheses by testing predicted consequences against observed clinical data.\n\n"

    "## Inputs you will receive\n"
    "1) The **clinical presentation** (and/or its abstracted feature list).\n"
    "2) A **deduction plan** that lists, for each hypothesis, its **expected consequences/findings**.\n\n"

    "## Task\n"
    "For each hypothesis:\n"
    "- Compare every expected consequence to the observed data.\n"
    "- Decide whether the hypothesis remains plausible or is refuted.\n"
    "- Recommend whether the diagnostic process can stop or should continue.\n\n"

    "## Evidence matching rules (per expected consequence)\n"
    "Assign exactly one status:\n"
    "- **confirmed**: clearly supported by one or more observed/abstracted features.\n"
    "- **not_observed**: not mentioned / not assessed; would require further history/exam/tests to evaluate.\n"
    "- **contradicted**: clear evidence in the presentation that argues against the expected consequence.\n\n"

    "## Hypothesis-level inductive evaluation\n"
    "After scoring all expected consequences, assign a global verdict:\n"
    "- **plausible**: most *key* expectations are confirmed AND no major contradictions exist.\n"
    "- **refuted**: one or more *key* expectations are contradicted OR there is a pattern of absence/contradiction that strongly argues against it.\n"
    "If the schema allows only these options, choose the best fit; otherwise use the schema’s uncertainty mechanism.\n\n"

    "## Termination recommendation (per hypothesis)\n"
    "Choose exactly one:\n"
    "- **sufficient_explanation_reached**: this hypothesis (alone or in combination already provided) satisfactorily explains the case.\n"
    "- **continue_testing**: hypothesis remains viable but needs additional targeted data/tests.\n"
    "- **reopen_abduction**: hypothesis appears weak; expand/reconsider the hypothesis set.\n\n"

    "## Constraints\n"
    "- Use **ALL** the hypotheses provided in the input.\n"
    "- Mention the hypotheses only once.\n"
    "- **Do not introduce** new clinical facts, tests, or hypotheses.\n"
    "- Use only what is present in the clinical presentation/abstracted features and the provided deduction plan.\n"
    "- Preserve negations and uncertainty exactly as evidenced in the text.\n\n"

    "## Output (STRICT)\n"
    "- Return **only** JSON that **exactly** matches the provided schema (no extra keys, no prose, no markdown).\n"
    )

    user_content = (
        "Clinical case narrative:\n"
        "------------------------\n"
        f"{case}\n\n"
        "Expected consequences from Deduction phase:\n"
        "------------------------------------------\n"
        f"{deduction_block}\n\n"
         "Task: Using ONLY the information above, perform the **induction** phase.\n"
         "- For each hypothesis and each expected finding, decide whether it is:\n"
         "  * 'confirmed' (clearly matches one or more abstracted features),\n"
         "  * 'not_observed' (not mentioned in the case, would require additional tests),\n"
         "  * 'contradicted' (clearly opposed by some abstracted feature).\n"
         "- Then, for each hypothesis, provide a global evaluation ('plausible', 'refuted').\n"
         "- Finally, provide a termination recommendation for each hypothesis "
         "  ('sufficient_explanation_reached', 'continue_testing', or 'reopen_abduction').\n\n"
         "Use the JSON schema (InductionResult -> EvaluatedHypothesis -> EvaluatedExpectedFinding).\n"
         "Return ONLY valid JSON, with no additional commentary."
    )

    messages = [
        {"role": "system", "content": system_prompt_induction},
        {"role": "user", "content": user_content},
    ]

    ded_hypotheses = deduction_output.get("hypotheses", [])

    answer = call_llm(messages=messages, model=model, response_format=induct_diagnoses([x["diagnosis"] for x in ded_hypotheses]))

    filtered_answer= {"hypotheses":[]}
    for diagnosis in answer["hypotheses"]:
        if diagnosis["evaluation"] == "plausible":
            filtered_answer["hypotheses"].append(diagnosis)

    return answer, filtered_answer