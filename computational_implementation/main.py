import json
import numpy as np
from computational_implementation.utils import Config
from computational_implementation.logging_file_generation import *
from computational_implementation.framework_implementation import *
from pathlib import Path

config = Config()

model_dir_name = "gpt"
model_name = "gpt-oss:20b"
config.set_model(model_name)

for i in range(len(config.CASES)):
    CASE_ID = config.CASES.loc[i, "Case-Id"]
    CASE_TEXT = config.CASES.loc[i, "Cases"]
    config.set_model_dir(f"{config.BASE_DIR}/epistemological_framework/{model_dir_name}")
    Path(f"{config.MODEL_DIR}/{CASE_ID}").mkdir(parents=True, exist_ok=True)

    for j in np.arange(0,1):
        config.set_result_dir(f"{config.MODEL_DIR}/{CASE_ID}")
        Path(config.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        final_state = run_with_human_log(
            graph_definition(config),
            {
                "case_text": CASE_TEXT,
                "model": config.MODEL,
                "suffix": config.SUFFIX,
                "iter": 0,
                "top_k": 10,
            },
            log_path=f"{config.RESULT_DIR}/log_{config.MODEL}_{CASE_ID}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_run_{j}.log",
            subgraphs=False,   # metti True solo se vuoi anche i subgraph
        )
