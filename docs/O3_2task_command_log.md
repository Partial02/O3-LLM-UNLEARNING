# O3 biology -> physics 실험 명령어 기록

작성일: 2026-05-20

이 문서는 `biology -> physics` selective switch 실험에서 어떤 결과 파일을 어떤 명령어로 만들었는지 기록하기 위한 로그입니다.

## 공통 설정

- 서버 작업 경로: `~/O3-LLM-UNLEARNING/ScienceQA_expriment`
- base model: `gcyzsl/O3_LLAMA2_ScienceQA`
- seed: `0`
- OOD weights: `./ood_checkpoints_scienceqa_0/`
- 2-task LoRA checkpoint: `./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force`
- OOD type: `_biology_physics`
- OOD setting: `C`

서버 접속 후 공통 환경 설정:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate o3
cd ~/O3-LLM-UNLEARNING/ScienceQA_expriment

mkdir -p /opt/dlami/nvme/o3/hf_cache
export HF_HOME=/opt/dlami/nvme/o3/hf_cache
export TRANSFORMERS_CACHE=/opt/dlami/nvme/o3/hf_cache
export HF_DATASETS_CACHE=/opt/dlami/nvme/o3/hf_cache/datasets
```

## 1. physics OOD detector 학습

결과 위치:

```text
ood_checkpoints_scienceqa_0/
```

실행 명령어:

```bash
python train_ood.py \
  --unlearn_dataset scienceqa_physics \
  --ood_dataset ood_scienceqa \
  --base_unlearn_path ./data/scienceqa/ \
  --base_ood_path ./data/scienceqa_RD_5/scienceqa_not_biology_physics \
  --seed 0
```

확인된 결과:

```text
ocsvm current auroc: 0.995
```

생성 파일 확인:

```bash
ls ood_checkpoints_scienceqa_0 | grep physics
```

## 2. physics-only 평가셋 생성

원본 O3 전처리는 cumulative SD 파일만 저장하므로, `biology_physics` 파일에서 `topic == physics`인 샘플만 골라 physics-only 평가셋을 만들었습니다.

결과 위치:

```text
data/scienceqa_SD_5/scienceqa_physics_train_SD.json
data/scienceqa_SD_5/scienceqa_physics_validation_SD.json
data/scienceqa_SD_5/scienceqa_physics_test_SD.json
```

실행 명령어:

```bash
python - <<'PY'
import json

base = "./data/scienceqa_SD_5"

for split in ["train", "validation", "test"]:
    src = f"{base}/scienceqa_biology_physics_{split}_SD.json"
    out = f"{base}/scienceqa_physics_{split}_SD.json"

    with open(src) as f:
        data = json.load(f)

    physics = [x for x in data if x.get("topic") == "physics"]

    with open(out, "w") as f:
        json.dump(physics, f, indent=2)

    print(out, len(physics))
PY
```

## 3. biology -> physics LoRA 이어 학습

입력 checkpoint:

```text
./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force
```

결과 checkpoint:

```text
./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force
```

실행 명령어:

```bash
python train_unlearn_lora_o.py \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --data_path ./data/scienceqa_random_force_5/scienceqa_physics_train_random_force.json \
  --output_dir ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --seed 0 \
  --batch_size 16 \
  --micro_batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force, \
  --ood_weight "1,1" \
  --orthogonal_loss_weight 0.1 \
  --train_on_inputs \
  --group_by_length \
  --add_eos_token \
  --resume_from_checkpoint ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force
```

확인된 결과:

```text
train_runtime: 164.8247
train_loss: 1.5420318809715476
epoch: 1.0
```

## 4. 2-task base 평가

기존 biology-only base 결과에서 재사용한 파일:

```bash
mkdir -p results_2task/base

cp results/base/biology_test_SD.json results_2task/base/biology_test_SD.json
cp results/base/commonqa_test.json results_2task/base/commonqa_test.json
cp results/base/openbookqa_test.json results_2task/base/openbookqa_test.json
```

### 4-1. physics base

결과 파일:

```text
results_2task/base/physics_test_SD.json
```

실행 명령어:

```bash
python eval_base.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_physics_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --output_file ./results_2task/base/physics_test_SD.json \
  --seed 0
```

결과:

```text
192/192, correct: 186, acc: 96.88%
```

### 4-2. retained RD base

결과 파일:

```text
results_2task/base/retained_biology_physics_RD.json
```

실행 명령어:

```bash
python eval_base.py \
  --test_dataset ./data/scienceqa_RD_5/scienceqa_not_biology_physics_test_RD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --output_file ./results_2task/base/retained_biology_physics_RD.json \
  --seed 0
```

결과:

```text
1635/1635, correct: 1506, acc: 92.11%
```

## 5. 2-task O3 unlearned 평가

restore option 없이 평가합니다.

### 5-1. biology unlearned

결과 파일:

```text
results_2task/o3_unlearned/biology_test_SD.json
```

실행 명령어:

```bash
mkdir -p results_2task/o3_unlearned

python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --output_file ./results_2task/o3_unlearned/biology_test_SD.json
```

결과:

```text
397/397, correct: 86, acc: 21.66%
```

### 5-2. physics unlearned

결과 파일:

```text
results_2task/o3_unlearned/physics_test_SD.json
```

실행 명령어:

```bash
python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_physics_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --output_file ./results_2task/o3_unlearned/physics_test_SD.json
```

결과:

```text
192/192, correct: 48, acc: 25.00%
```

## 6. restore biology 평가

### 6-1. restore biology / biology test

결과 파일:

```text
results_2task/restore_biology/biology_test_SD.json
```

실행 명령어:

```bash
mkdir -p results_2task/restore_biology

python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --restore_tasks biology \
  --output_file ./results_2task/restore_biology/biology_test_SD.json
```

결과:

```text
397/397, correct: 394, acc: 99.24%
```

### 6-2. restore biology / physics test

결과 파일:

```text
results_2task/restore_biology/physics_test_SD.json
```

실행 명령어:

```bash
python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_physics_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --restore_tasks biology \
  --output_file ./results_2task/restore_biology/physics_test_SD.json
```

결과:

```text
192/192, correct: 48, acc: 25.00%
```

## 7. restore physics 평가

### 7-1. restore physics / biology test

결과 파일:

```text
results_2task/restore_physics/biology_test_SD.json
```

실행 명령어:

```bash
mkdir -p results_2task/restore_physics

python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --restore_tasks physics \
  --output_file ./results_2task/restore_physics/biology_test_SD.json
```

결과:

```text
397/397, correct: 86, acc: 21.66%
```

### 7-2. restore physics / physics test

결과 파일:

```text
results_2task/restore_physics/physics_test_SD.json
```

실행 명령어:

```bash
python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_physics_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --restore_tasks physics \
  --output_file ./results_2task/restore_physics/physics_test_SD.json
```

결과:

```text
192/192, correct: 186, acc: 96.88%
```

## 8. restore biology physics 평가

### 8-1. restore both / biology test

결과 파일:

```text
results_2task/restore_biology_physics/biology_test_SD.json
```

실행 명령어:

```bash
mkdir -p results_2task/restore_biology_physics

python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_biology_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --restore_tasks biology physics \
  --output_file ./results_2task/restore_biology_physics/biology_test_SD.json
```

결과:

```text
397/397, correct: 394, acc: 99.24%
```

### 8-2. restore both / physics test

결과 파일:

```text
results_2task/restore_biology_physics/physics_test_SD.json
```

실행 명령어:

```bash
python eval_o3.py \
  --test_dataset ./data/scienceqa_SD_5/scienceqa_physics_test_SD.json \
  --base_model gcyzsl/O3_LLAMA2_ScienceQA \
  --seed 0 \
  --lora_weights ./SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_5/lora_force_random_biology_force_physics_force \
  --ood_type _biology_physics \
  --ood_setting C \
  --ood_weights ./ood_checkpoints_scienceqa_0/ \
  --restore_tasks biology physics \
  --output_file ./results_2task/restore_biology_physics/physics_test_SD.json
```

결과:

```text
192/192, correct: 186, acc: 96.88%
```

## 9. 핵심 결과 요약

```text
dataset    base    unlearned   restore biology   restore physics   restore both
biology    99.24   21.66       99.24             21.66             99.24
physics    96.88   25.00       25.00             96.88             96.88
```

해석:

```text
restore biology: biology만 복원되고 physics는 unlearned 유지
restore physics: physics만 복원되고 biology는 unlearned 유지
restore both: biology와 physics 모두 base 수준으로 복원
```
