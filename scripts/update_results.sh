# first argument is the credentials to log in to the ssh server

# directory to download the files to on the local machine
DOWNLOAD_PATH='aggregated_results/test_results.json'

ssh $1 ". activate.sh && cd sparse_dlib && dlib_aggregate_results --result_file_pattern=output/*/*/metrics/*/*/results/aggregate/evaluation.json --output_path=test_results.json"
rsync ${1}:sparse_dlib/test_results.json $DOWNLOAD_PATH