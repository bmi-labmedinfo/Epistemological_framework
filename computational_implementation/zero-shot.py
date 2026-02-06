import json
import numpy as np
from typing import List
from pydantic import BaseModel, Field
from computational_implementation.utils import call_llm, Config
from typing_extensions import Annotated
from pathlib import Path

config = Config()

model_dir_name = "gpt"
model_name = "gpt-oss:20b"
config.set_model(model_name)

class SingleDiagnosis(BaseModel):
    diagnosis: str = Field(..., description="The most likely diagnosis.")
    explanation: str = Field(..., description="The explanation of the most likely diagnosis.")

class TenDiagnoses(BaseModel):
    diagnoses: Annotated[
            List[SingleDiagnosis],
            Field(min_length=10, max_length=10)
            ] = Field(..., description="The 10 most likely diagnoses.")


def llm_as_a_judge(case:str , model:str):
    system_prompt = ("You are an experienced physician. "
                     "Your task is to determine the most likely diagnoses based on the patient’s clinical presentation and to explain the reasoning supporting this conclusion."
                     "Rank the diagnoses from most to least plausible.")
    user_content = ("Clinical case narrative:\n"
                    "------------------------\n"
                    f"{case}\n\n")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    answer = call_llm(messages=messages, model=model, response_format=TenDiagnoses)
    return answer


for i in range(len(config.CASES)):
    CASE_ID = config.CASES.loc[i, "Case-Id"]
    CASE_TEXT = config.CASES.loc[i, "Cases"]
    config.set_model_dir(f"{config.BASE_DIR}/zero-shot/{model_dir_name}")
    config.set_result_dir(f"{config.MODEL_DIR}/{CASE_ID}")
    Path(config.RESULT_DIR).mkdir(parents=True, exist_ok=True)

    for j in range(10):
        llm_answer = llm_as_a_judge(CASE_TEXT, config.MODEL)

        with open(f"{config.RESULT_DIR}/answer_run_{j}.json", 'w') as file:
            json.dump(llm_answer, file, indent=4)
