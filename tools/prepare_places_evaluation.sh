# 0. folder preparation
 #用于存储高分辨率图像#用于存储不同类型的掩码。
rm -r -f datasets/Chinese/evaluation/hires/

mkdir -p datasets/Chinese/evaluation/hires/
mkdir -p datasets/Chinese/evaluation/random_thick_256/
mkdir -p datasets/Chinese/evaluation/random_thin_256/
mkdir -p datasets/Chinese/evaluation/random_medium_256/
mkdir -p datasets/Chinese/evaluation/free_form_256/

# 1. sample 30000 new images,运行 eval_sampler.py 脚本来采样 20000 张新图像，并将输出结果存储在变量 OUT 中，并打印出来。
OUT=$(python3 tools/eval_sampler.py)
echo ${OUT}

echo "Preparing images..."
FILELIST=$(cat datasets/Chinese/eval_random_files.txt)
for i in $FILELIST
do
    $(cp ${i} datasets/Chinese/evaluation/hires/)
done

# 2. generate all kinds of masks
#指定图像数据文件夹。  #指定掩码配置文件。#指定掩码保存位置。
python3 tools/gen_masks.py \

    --img_data=datasets/Chinese/evaluation/hires/ \
    --lama_cfg=training/data/configs/thin_256.yaml \
    --msk_type=datasets/Chinese/evaluation/random_thin_256

python3 tools/gen_masks.py \
    --img_data=datasets/Chinese/evaluation/hires/ \
    --lama_cfg=training/data/configs/thick_256.yaml \
    --msk_type=datasets/Chinese/evaluation/random_thick_256

python3 tools/gen_masks.py \
    --img_data=datasets/Chinese/evaluation/hires/ \
    --lama_cfg=training/data/configs/medium_256.yaml \
    --msk_type=datasets/Chinese/evaluation/random_medium_256

python3 tools/gen_masks.py \
    --img_data=datasets/Chinese/evaluation/hires/ \
    --msk_ratio=0.0 \
    --msk_ratio=0.7 \
    --msk_type=datasets/Chinese/evaluation/free_form_256
#指定了掩码比例参数 --msk_ratio，分别设置为 0.0 和 0.7。