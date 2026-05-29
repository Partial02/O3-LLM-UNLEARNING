import os
import re
import json
import argparse
import torch
import random
from tqdm import tqdm

from src.ood_model_selector import RobertaForSelector, RobertaForSelector_inference
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from src.peft_model_hacked_o import PeftModel
import pickle
from src.modeling_llama_hacked_o import LlamaForCausalLM_ood
import math
import sys
from transformers import GenerationConfig, LlamaTokenizer, AutoConfig, AutoTokenizer
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

import json
import os.path as osp
from typing import Union


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


def knowledge_weights(gmm_scores, threshold_train):
    weight_res = []
    r = 3
    gmm_scores -= r * threshold_train
    for i in range(gmm_scores.shape[0]):
        weight_t = math.exp(gmm_scores[i]) / (1 + math.exp(gmm_scores[i]))
        weight_res.append(weight_t)
    return weight_res


def weighting_func_gmm(train_in_score, test_in_score):
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)

    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    gmm.covariances_ = np.array([[[std2 ** 2]], [[std2 ** 2]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    x0 = (mean1 + mean2) / 2
    return gmm, x0


def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)


def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)


def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)


all_w_res = []
all_w_res_dic = {}


def obtain_weights(input_x, gmm, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2

    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    if w_res > 0.9:
        w_res = 1.2
    elif w_res <= 0.4 and w_res > 0.3:
        w_res = w_res
    else:
        w_res = 0

    return w_res


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
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


def main():
    parser = argparse.ArgumentParser(description='Evaluation')

    ## --test_dataset: 평가할 데이터셋 경로 목록 (space-separated)
    ## default: biology 단일 / 4종 모두 평가하려면 아래 주석 해제
    parser.add_argument('--test_dataset', nargs='+',
                        default=["./data/scienceqa_SD_5/scienceqa_biology_test_SD.json"],
                        # default=[
                        #     "./data/scienceqa_SD_5/scienceqa_biology_test_SD.json",
                        #     "./data/scienceqa_SD_5/scienceqa_physics_test_SD.json",
                        #     "./data/scienceqa_SD_5/scienceqa_chemistry_test_SD.json",
                        #     "./data/scienceqa_SD_5/scienceqa_economics_test_SD.json",
                        # ],
                        help='평가할 테스트셋 경로(들)')
    parser.add_argument('--base_model', type=str, default="gcyzsl/O3_LLAMA2_ScienceQA",
                        help='base_model')
    parser.add_argument('--ood_base_model', type=str, default="roberta-large",
                        help='OOD 검출기 베이스 모델')
    ## --lora_weights: LoRA 체크포인트 경로 목록 (space-separated)
    ## 여러 개 지정 시 마지막 경로를 실제 로드에 사용 (base_lora)
    ## default: biology 단일 / 4종 누적 학습된 LoRA를 쓰려면 아래 주석 해제
    parser.add_argument('--lora_weights', nargs='+', type=str,
                        default=["./SCALE_0.1_seed_1_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force"],
                        # default=[
                        #     "./SCALE_0.1_seed_1_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force_chemistry_force_economics_force",
                        # ],
                        help='LoRA 체크포인트 경로(들): 마지막 경로를 기준으로 로드')
    parser.add_argument('--ood_weights', type=str,
                        default="./ood_checkpoints_scienceqa_1/",
                        help='OOD 체크포인트 디렉토리 (끝에 / 포함)')
    ## --ood_type: 망각된 과목 목록 (space-separated)
    ## default: biology 단일 / 4종 모두 처리하려면 아래 주석 해제
    parser.add_argument('--ood_type', nargs='+',
                        default=["biology"],
                        # default=["biology", "physics", "chemistry", "economics"],
                        help='망각된 과목(들): biology physics chemistry economics')
    parser.add_argument('--restore_tasks', nargs='*', type=int, default=[],
                        help='복원(OOD gate 건너뜀)할 과목 인덱스 목록 (예: 0 2)')
    parser.add_argument('--ood_setting', type=str, default="c",
                        help='ood setting')
    parser.add_argument('--ood_setting_name', type=str, default="scienceqa",
                        help='ood setting name')
    parser.add_argument('--seed', type=int, default=1, help='seed')

    args = parser.parse_args()
    set_seed(args.seed)

    base_model = args.base_model
    max_batch_size = 1

    ## Fix B: args.ood_type은 이미 리스트 → 빈 문자열만 필터링
    ood_types = [t for t in args.ood_type if len(t) > 0]

    ood_setting = args.ood_setting
    ood_setting_names = args.ood_setting_name

    print("test_dataset  :", args.test_dataset)
    print("base_model    :", args.base_model)
    print("lora_weights  :", args.lora_weights)
    print("ood_weights   :", args.ood_weights)
    print("ood_types     :", ood_types)
    print("ood_setting   :", ood_setting)
    print("ood_setting_name:", args.ood_setting_name)
    print("restore_tasks :", args.restore_tasks)

    ## 각 과목별 OOD 체크포인트 기본 경로 구성
    ood_weight_paths = []
    for topic in ood_types:
        o_p = args.ood_weights + f"{ood_setting_names}_{topic}_ood_{ood_setting_names}"
        ood_weight_paths.append(o_p)

    ## OOD 파일 suffix: train_ood.py / run_ood.py 가 저장하는 방식과 동일하게 "ocsvm"
    ood_method = "ocsvm"

    ## Fix C: lora_weights 리스트의 마지막 경로를 PeftModel 로드에 사용
    base_lora = args.lora_weights[-1] if isinstance(args.lora_weights, list) else args.lora_weights
    path = "/".join(base_lora.split("/")[:-1])
    if not os.path.exists(path):
        os.mkdir(path)
    ood_type_str = "_".join(ood_types)  ## 결과 파일명용 문자열

    ## ── 모델 로드 (테스트 루프 밖에서 1회) ────────────────────────────
    load_8bit = False
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')
    lora_target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj"
    ]
    config = AutoConfig.from_pretrained(base_model)
    config.lora_target_modules = lora_target_modules

    orthogonal_loss = False
    olora_weights = {}
    config.orthogonal_loss = orthogonal_loss
    config.orthogonal_loss_weight = 0.1
    model = LlamaForCausalLM_ood.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ## Fix C: base_lora (단일 경로 문자열)로 PeftModel 로드
    model = PeftModel.from_pretrained(
        model,
        base_lora,
        torch_dtype=torch.bfloat16,
    )
    model.init_olora(orthogonal_loss=orthogonal_loss, olora_weights=olora_weights)
    model.init_active_adapters_d(active_adapters_d=['default'])

    print(model.config.pad_token_id, tokenizer.pad_token_id)
    print(model.config.bos_token_id, tokenizer.bos_token_id)
    print(model.config.eos_token_id, tokenizer.eos_token_id)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    ## ── OOD 검출기 로드 (테스트 루프 밖에서 1회) ─────────────────────
    ood_base_model = args.ood_base_model
    ood_tokenizer = RobertaTokenizer.from_pretrained(ood_base_model)
    ood_models = []
    ood_clrs = []
    ood_thresholds = []
    ood_x0 = []
    ood_mean_lists = []
    ood_precision_lists = []
    ood_fea_lists = []
    ood_gmm_w_cls = []

    ## Fix E: 파일 suffix를 topic명 대신 ood_method("ocsvm") 사용
    ##        → train_ood.py / run_ood.py 가 저장하는 naming과 일치
    for base_path, topic in zip(ood_weight_paths, ood_types):
        roberta_path   = base_path + f"_roberta_{ood_method}"
        ocsvm_path     = base_path + f"_{ood_method}.pkl"
        threshold_path = base_path + f"_threshold_{ood_method}.json"
        mean_list_path = base_path + f"_mean_list_{ood_method}.pt"
        precision_path = base_path + f"_precision_list_{ood_method}.pt"
        fea_list_path  = base_path + f"_fea_list_{ood_method}.pt"
        gmm_w_path     = base_path + f"_gmm_w_{ood_method}.pkl"

        ood_models.append(
            RobertaForSelector_inference(ood_base_model, lora_path=roberta_path, projection_dim=100).to(device)
        )
        with open(ocsvm_path, "rb") as f:
            ood_clrs.append(pickle.load(f))
        with open(gmm_w_path, "rb") as f:
            ood_gmm_w_cls.append(pickle.load(f))
        with open(threshold_path) as f:
            threshold = json.load(f)
        ood_thresholds.append(threshold[1])
        ood_x0.append(threshold[0])
        ood_mean_lists.append(torch.load(mean_list_path, map_location=torch.device(device)))
        ood_precision_lists.append(torch.load(precision_path, map_location=torch.device(device)))
        ood_fea_lists.append(torch.load(fea_list_path, map_location=torch.device(device)))

    prompter = Prompter(template_name="alpaca")
    max_new_tokens = 128
    save_every = 200

    ## ── Fix A & D: 테스트셋 루프로 평가 전체를 감쌈 ──────────────────
    for test_file in args.test_dataset:
        print(f"\n{'='*60}")
        print(f"Evaluating: {test_file}")

        ## Fix A: data_a 로드를 루프 안으로 이동
        data_a = json.load(open(test_file, 'r', encoding='utf-8'))

        restore_tag = ("_restore" + "_".join(map(str, args.restore_tasks))) if args.restore_tasks else ""
        result_file = path + "/test_noretain_{}_seed{}_oodlora{}_{}_{}".format(
            ood_setting,
            str(args.seed),
            restore_tag,
            base_lora.split("/")[-1],
            test_file.split("/")[-1],
        )
        print(f"Result will be saved to: {result_file}")

        correct = 0
        results = []
        outputs = []
        gt = []

        for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
            end_idx = min(start_idx + max_batch_size, len(data_a))
            batch = data_a[start_idx:end_idx]
            answers = [str(example["answer"]) for example in batch]
            prompts = [prompter.generate_prompt(example['instruction'], example['input']) for example in batch]

            ood_input = ood_tokenizer(
                prompts, padding='max_length', truncation=True, max_length=512, return_tensors="pt"
            )
            max_ood = 0

            for i in range(len(ood_weight_paths)):
                if i in args.restore_tasks:
                    ## restore_tasks에 포함된 과목은 OOD gate를 건너뜀
                    ## → LoRA 미적용 → 원래 지식 복원 (relearning)
                    continue
                mah_score = ood_models[i].get_unsup_Mah_score_s(
                    ood_input, ood_mean_lists[i], ood_precision_lists[i], ood_fea_lists[i]
                )[:, 1:]
                test_score = ood_clrs[i].score_samples(mah_score)
                w_ood = obtain_weights(test_score, ood_gmm_w_cls[i], ood_x0[i])
                if w_ood > max_ood:
                    max_ood = w_ood

            all_w_res.append(max_ood)
            all_w_res_dic[str(max_ood)[:5]] = all_w_res_dic.get(str(max_ood)[:5], 0) + 1

            print("ood_weight:", [1, max_ood])
            model.init_oodweight(ood_weight=[1, max_ood])

            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            s = generation_output.sequences
            output = tokenizer.batch_decode(s)
            output = [prompter.get_response(otp) for otp in output]

            pattern = re.compile(r'The answer is ([A-Z]).')
            res = [pattern.findall(otp) for otp in output]
            for r_i in range(len(res)):
                if len(res[r_i]) == 1:
                    answer = res[r_i][0]
                else:
                    answer = "FAILED"
                results.append(res[r_i])
                outputs.append(output[r_i])
                gt.append(answers[r_i])
                if str(answer) == str(answers[r_i]):
                    correct += 1
                    print('correct:', str(answer), str(answers[r_i]))
                else:
                    print('gt-ans:', str(answer), str(answers[r_i]))

            acc = correct / len(results) * 100

            if end_idx % save_every == 0 or end_idx == len(data_a):
                print(f"{len(results)}/{len(data_a)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
                save_data = {
                    'acc': acc,
                    'correct': correct,
                    'len': len(results),
                    'results': results,
                    'outputs': outputs,
                }
                with open(result_file, 'w') as f:
                    json.dump(save_data, f, indent=2, separators=(',', ': '))

        print(f"[{test_file.split('/')[-1]}] Final acc: {round(acc, 2)}%")

    print("\nAll evaluations done.")
    if all_w_res:
        print(f"OOD weight stats — mean: {np.mean(all_w_res):.4f}, min: {min(all_w_res):.4f}, max: {max(all_w_res):.4f}")


if __name__ == "__main__":
    main()
