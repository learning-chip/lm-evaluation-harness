"""
Tested with nanovllm commit:
https://github.com/GeeeekExplorer/nano-vllm/commit/4fc764f175f48ff833ecea117070d297fea5779c
"""

import torch

import lm_eval
from lm_eval import evaluator
from lm_eval.utils import make_table

run_reference = False  # run standard vllm for comparison
if run_reference:
    import os
    os.environ["VLLM_USE_V1"] = "0"
    from lm_eval.models.vllm_causallms import VLLM
else:
    from lm_eval.models.nanovllm import NanoVLLM as VLLM
    # TODO: takes too long than vllm, need to check if `generate_until` stops on time

model_path = "/workspace/weights/Qwen3-0.6B"

model = VLLM(
    pretrained=model_path,
    dtype=torch.bfloat16,
    max_model_len=2048,
    max_num_seqs=32,
    device="cuda",
    enforce_eager=True,
    tensor_parallel_size=1
)

# NOTE: can only run generative tasks as `prompt_logprobs` is not supported in nanovllm
task_list = [
    "mmlu_college_biology_generative",
    "mmlu_high_school_computer_science_generative"
    ]

results = evaluator.simple_evaluate(
    model=model,
    tasks=task_list,
    task_manager=lm_eval.tasks.TaskManager(),
    batch_size=32,
    limit=0.1,
    write_out=True,
    log_samples=False
)

print(make_table(results))
