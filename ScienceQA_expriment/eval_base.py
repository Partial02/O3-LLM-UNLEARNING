import argparse
import json
import os
import random
import re
import sys
from typing import Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except Exception:
    pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:",
        }

        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def parse_answer(output: str) -> str:
    pattern = re.compile(r"The answer is ([A-Z]).")
    match = pattern.findall(output)
    if len(match) == 1:
        return match[0]
    return "FAILED"


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="gcyzsl/O3_LLAMA2_ScienceQA")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_a = json.load(open(args.test_dataset))
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print(args.test_dataset)
    print(args.base_model)
    print(args.output_file)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=args.load_in_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if device == "cuda" and not args.load_in_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    prompter = Prompter(template_name="alpaca")
    correct = 0
    results = []
    outputs = []
    predictions = []
    labels = []
    save_every = 200

    for start_idx in tqdm(range(0, len(data_a), args.max_batch_size)):
        end_idx = min(start_idx + args.max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]
        answers = [str(example["answer"]) for example in batch]
        prompts = [prompter.generate_prompt(example["instruction"], example["input"]) for example in batch]

        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_new_tokens,
            )

        decoded = tokenizer.batch_decode(generation_output.sequences)
        output = [prompter.get_response(otp) for otp in decoded]

        for output_i, gt_answer in zip(output, answers):
            answer = parse_answer(output_i)
            predictions.append(answer)
            labels.append(gt_answer)
            results.append([answer] if answer != "FAILED" else [])
            outputs.append(output_i)

            if str(answer) == str(gt_answer):
                correct += 1
                print("correct:", str(answer), str(gt_answer))
            else:
                print("gt-ans:", str(answer), str(gt_answer))

        acc = correct / len(results) * 100

        if end_idx % save_every == 0 or end_idx == len(data_a):
            print(
                f"{len(results)}/{len(data_a)}, correct: {correct}, "
                f"acc: {round(acc, 2)}%, saving to {args.output_file}"
            )
            result_data = {
                "acc": acc,
                "correct": correct,
                "len": len(results),
                "results": results,
                "outputs": outputs,
                "predictions": predictions,
                "labels": labels,
            }
            with open(args.output_file, "w") as f:
                json.dump(result_data, f, indent=2, separators=(",", ": "))


if __name__ == "__main__":
    main()
