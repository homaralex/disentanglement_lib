ROOT_DIR=$1

for FILE_ID in {0..10799}; do
  EXTRACTED_PATH=${ROOT_DIR}/${FILE_ID}
  ZIP_PATH=${EXTRACTED_PATH}.zip
  wget https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/${FILE_ID}.zip -O ${ZIP_PATH}
  unzip ${ZIP_PATH} -d $ROOT_DIR
  rm ${ZIP_PATH}
  rm -r ${EXTRACTED_PATH}/visualizations ${EXTRACTED_PATH}/model ${EXTRACTED_PATH}/postprocessed
done



