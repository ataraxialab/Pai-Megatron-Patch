# 此处设置分块数为10，如数据处理慢可设置稍大
NUM_PIECE=10

dataset_dir=/workspace/mnt/storage/zhaozhijian@supremind.com/FS4T2/MAP_CC/

# 对merged_wudao_cleaned.json文件进行处理
mkdir -p ${dataset_dir}/cleaned_zst/
# 查询数据总长度，对数据进行拆分
NUM=$(sed -n '$=' ${dataset_dir}/zh_baike.jsonl)
echo "total line of dataset is $NUM, data will be split into $NUM_PIECE pieces for processing"
NUM=`expr $NUM / $NUM_PIECE`
echo "each group is processing $NUM sample"
split_dir=${dataset_dir}/split
mkdir $split_dir
split -l $NUM --numeric-suffixes --additional-suffix=.jsonl ${dataset_dir}/zh_baike.jsonl $split_dir/

# 数据压缩
o_path=${dataset_dir}/cleaned_zst/
mkdir -p $o_path
files=$(ls $split_dir/*.jsonl)
for filename in $files
do
   f=$(basename $filename)
   zstd -z $filename -o $o_path/$f.zst &
done
rm -rf $split_dir
#rm ${dataset_dir}/wudao/merged_wudao_cleaned.json