import json
import datetime
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from computational_implementation.utils import call_llm

# OUTPUT ABSTRACTION
class FindingCategory(str, Enum):
    #EPIC
    present_illness="History of Present Illness"
    medical_history="Past Medical History"
    medications="Current Medications"
    allergies="Allergies"
    family_history="Family History"
    physical_examination="Physical Examination"
    investigations="Investigations"
    #SDOH
    economic_stability="Economic Stability"
    education="Education Access & Quality"
    health_quality="Healthcare Access & Quality"
    environment="Neighborhood & Built Environment"
    social_context="Social & Community Context"

class Finding(BaseModel):
    finding: str = Field(...,description="A specific clinical finding that falls within the assigned category and reflects a distinct, relevant element from the clinical case.")
    explanation: str = Field(..., description="A brief explanation of the finding as it appears in the clinical text. Provide the specific excerpt or phrasing from the source clinical case where this finding is described.")

class AbstractionFeature(BaseModel):
    category: FindingCategory = Field(..., description="The category of the finding.")
    findings: List[Finding] = Field(..., description="List of findings belonging to the specified category.")

class AbstractionFeaturesList(BaseModel):
    features: List[AbstractionFeature] = Field(description="List of the features extracted from the clinical case.")


def abstraction_postprocessing(answer):
    flattened = []
    for feature in answer['features']:
        category = feature['category']
        for f in feature['findings']:
            flattened.append({
                'category': category,
                'finding': f['finding'],
                'explanation': f['explanation']
            })
    return flattened

def query_abstraction(case: str, model: str, iteration:int=0) -> List:
    """
    Abstraction phase:
    - Output: AbstractionFeaturesList
    """
    system_prompt_abstraction = (
    "You are a **clinical information extraction system**. Your job is to transform raw medical narrative into a **concise, structured set of clinically relevant features**.\n\n"

    "## Input\n"
    "- You will receive a clinical note / case description / event narrative.\n\n"

    "## Output (STRICT)\n"
    "- Produce output that **exactly matches the provided schema** (field names, nesting, data types).\n"
    "- Output **only** the structured result—no commentary, no extra keys, no prose.\n"
    "- If the schema requires a value but the text does not provide it, use the schema’s allowed representation for missing/unknown (e.g., null/\"unknown\"). Otherwise, omit.\n\n"

    "## What to extract (maximize clinical coverage)\n"
    "Capture clinically meaningful information, such as:\n"
    "- Presenting problem(s) and symptoms (with onset/timeline, severity, progression)\n"
    "- Relevant past medical/surgical history, risk factors, baseline function\n"
    "- Medications (including recent changes), allergies/adverse reactions\n"
    "- Vital signs and physical exam findings\n"
    "- Key investigations: labs, imaging, ECG, microbiology, pathology (include values/units when present)\n"
    "- Diagnoses/working impressions and clinical reasoning explicitly stated in the text\n"
    "- Treatments and interventions (drug, dose/route if present, procedures, supportive care), and response\n"
    "- Disposition and follow-up plans (if present)\n\n"

    "## Relevance filter\n"
    "- Include details that **help understand the clinical problem**, differential, severity, trajectory, or management.\n"
    "- Exclude redundant, trivial, purely administrative, or non-clinical details (unless the schema explicitly requires them).\n\n"

    "## Abstraction & de-duplication\n"
    "- **Do not duplicate** the same concept across fields. If multiple statements express the same clinical concept, **merge** into one well-abstracted feature.\n"
    "- Normalize synonymous phrasing (e.g., \"SOB\" → \"dyspnea\"), while preserving the original meaning.\n\n"

    "## Evidence & uncertainty\n"
    "- **Do not invent** or infer facts not supported by the text.\n"
    "- Preserve negations and exclusions **when clinically important** (e.g., \"no chest pain\").\n"
    "- If information is uncertain/possible/suspected, represent uncertainty using the schema’s mechanism (e.g., status/qualifier fields). If none exists, encode it explicitly in the value (e.g., \"suspected pneumonia\").\n\n"

    "## Consistency & quality checks\n"
    "- Prefer exact numbers/dates from the text over vague paraphrases.\n"
    "- Keep entries compact but complete (include temporality, laterality, and context when present).\n"
    )
    user_prompt = (
        "Clinical case narrative:\n"
        "------------------------\n"
        f"{case}\n\n")
    messages = [{"role": "system", "content": system_prompt_abstraction},
                {"role": "user", "content": user_prompt}]
    answer = call_llm(messages=messages, model=model, response_format=AbstractionFeaturesList)
    post_processed_answer = abstraction_postprocessing(answer)

    return post_processed_answer