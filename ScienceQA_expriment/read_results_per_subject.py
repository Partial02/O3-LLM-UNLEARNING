import json
import re

SCALE       = str(0.1)
SEED        = 0
LABEL_K     = "force"
RESTORE_TASKS = []   # [] = 전부 unlearn / [1, 3] = physics+economics relearn 등
OOD_SETTING = "C"

DATASETS = ["biology", "physics", "chemistry", "economics"]

restore_tag  = ("_restore" + "_".join(map(str, RESTORE_TASKS))) if RESTORE_TASKS else ""
BASE_CKPT    = f"./SCALE_{SCALE}_seed_{SEED}_o_unlearn_lora_{LABEL_K}_checkpoints_5"
SD_PATH      = "./data/scienceqa_SD_5"
RD_PATH      = "./data/scienceqa_RD_5"

pattern = re.compile(r'The answer is ([A-Z])\.')

def parse_pred(pred_list):
    if isinstance(pred_list, list) and len(pred_list) == 1:
        return pred_list[0]
    return "FAILED"

def acc_by_topic(data, result_preds, target_topic=None):
    correct, total = 0, 0
    for sample, pred in zip(data, result_preds):
        if target_topic and sample.get('topic') != target_topic:
            continue
        gt = str(sample['answer'])
        if parse_pred(pred) == gt:
            correct += 1
        total += 1
    if total == 0:
        return None, 0
    return round(correct / total * 100, 2), total

# ─────────────────────────────────────────────────────────────
# 공통: stage별 lora 경로 누적
# ─────────────────────────────────────────────────────────────
TYPE = ""
OUTPUT_LORA = f"{BASE_CKPT}/lora_{LABEL_K}_random"
stages = []   # (TYPE, prefix) 리스트
for UNLEAN_D in DATASETS:
    OUTPUT_LORA += f"_{UNLEAN_D}_{LABEL_K}"
    TYPE        += f"_{UNLEAN_D}"
    lora_last    = OUTPUT_LORA.split("/")[-1]
    prefix       = f"{BASE_CKPT}/test_noretain_{OOD_SETTING}_seed{SEED}_oodlora{restore_tag}_{lora_last}"
    stages.append((TYPE, prefix))


# ─────────────────────────────────────────────────────────────
# [블록 1] RD — 과목별 accuracy
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RD (Retain Data) — 과목별 accuracy")
print("="*60)
for TYPE, prefix in stages:
    rd_data_path   = f"{RD_PATH}/scienceqa_not{TYPE}_test_RD.json"
    rd_result_path = f"{prefix}_scienceqa_not{TYPE}_test_RD.json"
    try:
        with open(rd_data_path,   'r', encoding='utf-8') as f: rd_data = json.load(f)
        with open(rd_result_path, 'r', encoding='utf-8') as f: rd_res  = json.load(f)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}"); continue

    print(f"\nStage {TYPE}:")
    # RD = unlearn 대상이 아닌 과목들 → topic 종류 파악
    topics_in_rd = list({s['topic'] for s in rd_data if 'topic' in s})
    for topic in sorted(topics_in_rd):
        acc, n = acc_by_topic(rd_data, rd_res['results'], target_topic=topic)
        print(f"  {topic:<15} acc={acc}%  (n={n})")
    acc_all, n_all = acc_by_topic(rd_data, rd_res['results'])
    print(f"  {'[전체]':<15} acc={acc_all}%  (n={n_all})")


# ─────────────────────────────────────────────────────────────
# [블록 2] SD_train — 과목별 accuracy
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SD_train (Subject Data train) — 과목별 accuracy")
print("="*60)
for TYPE, prefix in stages:
    sd_data_path   = f"{SD_PATH}/scienceqa{TYPE}_train_SD.json"
    sd_result_path = f"{prefix}_scienceqa{TYPE}_train_SD.json"
    try:
        with open(sd_data_path,   'r', encoding='utf-8') as f: sd_data = json.load(f)
        with open(sd_result_path, 'r', encoding='utf-8') as f: sd_res  = json.load(f)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}"); continue

    included = [s for s in DATASETS if s in TYPE]
    print(f"\nStage {TYPE}:")
    for subj in included:
        acc, n = acc_by_topic(sd_data, sd_res['results'], target_topic=subj)
        print(f"  {subj:<15} acc={acc}%  (n={n})")
    acc_all, n_all = acc_by_topic(sd_data, sd_res['results'])
    print(f"  {'[전체]':<15} acc={acc_all}%  (n={n_all})")


# ─────────────────────────────────────────────────────────────
# [블록 3] SD_test — 과목별 accuracy
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SD_test (Subject Data test) — 과목별 accuracy")
print("="*60)
for TYPE, prefix in stages:
    sd_data_path   = f"{SD_PATH}/scienceqa{TYPE}_test_SD.json"
    sd_result_path = f"{prefix}_scienceqa{TYPE}_test_SD.json"
    try:
        with open(sd_data_path,   'r', encoding='utf-8') as f: sd_data = json.load(f)
        with open(sd_result_path, 'r', encoding='utf-8') as f: sd_res  = json.load(f)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}"); continue

    included = [s for s in DATASETS if s in TYPE]
    print(f"\nStage {TYPE}:")
    for subj in included:
        acc, n = acc_by_topic(sd_data, sd_res['results'], target_topic=subj)
        print(f"  {subj:<15} acc={acc}%  (n={n})")
    acc_all, n_all = acc_by_topic(sd_data, sd_res['results'])
    print(f"  {'[전체]':<15} acc={acc_all}%  (n={n_all})")
