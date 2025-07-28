#目的是准备 Places2 数据集中的评估数据，包括采样图像并生成分割掩码。

# 0. folder preparation
#用于存储高分辨率图像#用于存储分割掩码。
rm -r -f datasets/Chinese/evaluation/segm_hires/
mkdir -p datasets/Chinese/evaluation/segm_hires/
mkdir -p datasets/Chinese/evaluation/random_segm_256/

# 1. sample 10000 new images运行 eval_segm_sampler.py 脚本来采样 10000 张新图像，并将输出结果存储在变量 OUT 中，并打印出来
OUT=$(python3 tools/eval_segm_sampler.py)
echo ${OUT}

echo "Preparing images..."
SEGM_FILELIST=$(cat datasets/Chinese/eval_random_segm_files.txt)
for i in $SEGM_FILELIST
do
    $(cp ${i} datasets/Chinese/evaluation/segm_hires/)
done
#读取 datasets/Chinese/eval_random_segm_files.txt 文件中列出的图像文件，并将这些图像复制到 datasets/Chinese/evaluation/segm_hires/ 文件夹中。


# 2. generate segmentation masks指定掩码配置文件。指定图像数据文件夹。指定掩码保存位置。

python3 tools/gen_random_segm_masks.py \
    training/data/configs/segm_256.yaml \
    datasets/Chinese/evaluation/segm_hires/ \
    datasets/Chinese/evaluation/random_segm_256
