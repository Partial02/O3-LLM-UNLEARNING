for SEED in 0
do
  TYPE=""
  ## revised: 생물학 only v.s. 4종을 모두 받을 수 있도록 주석으로 처리
  DATASETS=("biology")
  # DATASETS=("biology" "physics" "chemistry" "economics")
     
  # for UNLEAN_D in "biology" "physics" "chemistry"
  for UNLEAN_D in "${DATASETS[@]}"
  do
      TYPE+="_${UNLEAN_D}"
      OODPATH_1="./data/scienceqa_RD_5/scienceqa_not${TYPE}"
      python run_ood.py \
           --unlearn_dataset "scienceqa_${UNLEAN_D}" \
           --ood_dataset "ood_scienceqa" \
           --base_unlearn_path "./data/scienceqa/" \
           --base_ood_path ${OODPATH_1} \
           --seed ${SEED}
  done
done
