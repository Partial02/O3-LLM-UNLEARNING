BASE_MODEL="gcyzsl/O3_LLAMA2_ScienceQA"
OOD_SETTING="C"
for SCALE in 0.1
do
  # for SEED in 0 1 2
  for SEED in 0
  do
    for LABEL_K in "force"
    do
      OUTPUT_1="./SCALE_${SCALE}_seed_${SEED}_o_unlearn_lora_${LABEL_K}_checkpoints_5/lora_${LABEL_K}_random"
      TYPE=""        ## 파일 경로 구성용 (누적 언더스코어 문자열)
      OOD_TYPES=()  ## eval_o3.py 에 space-separated 로 전달할 과목 배열

      ## biology 단일 망각 (default)
      DATASETS=("biology")
      ## 4종 모두 망각하려면 아래 주석 해제
      # DATASETS=("biology" "physics" "chemistry" "economics")

      for UNLEAN_D in "${DATASETS[@]}"
      do
        OUTPUT_1+="_${UNLEAN_D}_${LABEL_K}"
        TYPE+="_${UNLEAN_D}"        ## 파일 경로용 누적 문자열
        OOD_TYPES+=("${UNLEAN_D}") ## Python 인자용 배열 (공백 구분)

        TESTPATH_1="./data/scienceqa_RD_5/scienceqa_not${TYPE}_test_RD.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type "${OOD_TYPES[@]}" \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"

        TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_train_SD.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type "${OOD_TYPES[@]}" \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/" \
          --restore_tasks 0

        TESTPATH_1="./data/scienceqa_SD_5/scienceqa${TYPE}_test_SD.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type "${OOD_TYPES[@]}" \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/" \
          --restore_tasks 0

        TESTPATH_1="./data/commonqa/commonqa_test.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type "${OOD_TYPES[@]}" \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"

        TESTPATH_1="./data/openbookqa/openbookqa_test.json"
        python eval_o3.py \
          --test_dataset ${TESTPATH_1} \
          --base_model ${BASE_MODEL} \
          --seed ${SEED} \
          --lora_weights ${OUTPUT_1} \
          --ood_type "${OOD_TYPES[@]}" \
          --ood_setting ${OOD_SETTING} \
          --ood_weights "./ood_checkpoints_scienceqa_${SEED}/"
      done
    done
  done
done
