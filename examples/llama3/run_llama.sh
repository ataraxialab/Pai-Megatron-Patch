

sh run_pretrain_megatron_llama.sh \
dsw \
../../  \
8B \
1 \
4 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4  \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
/workspace/mnt/storage/zhaozhijian@supremind.com/PTcode/llama3-datasets-16parts/wudao_llama3bpe_2_content_document \
/workspace/mnt/storage/zhaozhijian@supremind.com/PTcode/llama3-ckpts/Meta-Llama-3-8B/ \
1000000000   \
10000   \
/workspace/mnt/storage/output/
