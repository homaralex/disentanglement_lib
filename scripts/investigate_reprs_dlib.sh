OUT_DIR=output_investigate
mkdir -p ${OUT_DIR}

for model_num in {0..2}; do
  if [[ $(python scripts/config_checker.py --model_num=0 --model_name=beta) -eq 0 ]]
  then
    continue
  fi

  # download the pretrained model
  zip_path="${OUT_DIR}/${model_num}.zip"
  wget https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/${model_num}.zip -O ${zip_path}
  unzip -o ${zip_path} -d ${OUT_DIR}
  rm ${zip_path}

  # run the script
  model_dir=${OUT_DIR}/${model_num}
  python scripts/investigate_reprs.py \
  --model_dir=${model_dir} \
  --num_points=1000 \
  --overwrite \
  --output_file=${OUT_DIR}/dlib_reprs.pkl;

  # remove the model data
  rm -r ${model_dir}
done