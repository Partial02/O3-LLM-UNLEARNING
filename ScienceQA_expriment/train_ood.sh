for SEED in 0 1 2
do
  TYPE=""
  # for UNLEAN_D in "biology" "physics" "chemistry" "economics"
  ## revised: 생물학만 v.s. 4종을 모두 망각하는 걸 주석으로 바꿀 수 있도록 개선
  DATASETS=("biology") # 생물학만을 망각시킬 경우
  # DATASETS=("biology" "physics" "chemistry" "economics") # 4종을 모두 망각시킬 경우
  
  for UNLEAN_D in "${DATASETS[@]}"
  do
      TYPE+="_${UNLEAN_D}"
      OODPATH_1="./data/scienceqa_RD_5/scienceqa_not${TYPE}"
        
      python train_ood.py \
          --unlearn_dataset "scienceqa_${UNLEAN_D}" \
          --ood_dataset "ood_scienceqa" \
          --base_unlearn_path "./data/scienceqa/" \
          --base_ood_path ${OODPATH_1} \
          --seed ${SEED}
  done
done
