export dataset_dir=/workspace/mnt/storage/zhaozhijian@supremind.com/FS4T2/MAP_CC
export WORK_DIR=/workspace

bash run_make_pretraining_dataset.sh \
/workspace/Pai-Megatron-Patch \
${dataset_dir}/cleaned_zst/ \
llamabpe \
${dataset_dir}/baike \
../../../llama3-ckpts/Meta-Llama-3-8B \
16
#rm -rf ${dataset_dir}/cleaned_zst