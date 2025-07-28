# 0. folder preparation
#删除 val_hires 文件夹及其内容。创建 val_hires 文件夹。创建 random_val 文件夹。
rm -r -f datasets/Chinese/evaluation/val_hires/
mkdir -p datasets/Chinese/evaluation/val_hires/
mkdir -p datasets/Chinese/evaluation/random_val/

# 1. sample 10000 new images运行 val_sampler.py 脚本以采样 10000 张新图像，并将输出结果存储在变量 OUT 中，并打印出来。
OUT=$(python3 tools/val_sampler.py)
echo ${OUT}

echo "Preparing images..."
FILELIST=$(cat datasets/Chinese/val_random_files.txt)
for i in $FILELIST
do
    $(cp ${i} datasets/Chinese/evaluation/val_hires/)
done
#读取 val_random_files.txt 文件，获取采样图像的文件列表。遍历文件列表中的每个文件路径。将每个文件复制到 val_hires 文件夹中。确保将采样的 10000 张图像复制到 val_hires 文件夹中，以便后续生成掩码使用。

#2.generate all kinds of masks指定图像数据文件夹。指定掩码覆盖比例。两个比例表明生成两种不同的掩码，一种覆盖比例为0.0，另一种为0.7。指定生成掩码的保存位置,为采样图像生成不同类型和覆盖比例的掩码，并保存到 random_val 文件夹中。

python3 tools/gen_masks.py \
    --img_data=datasets/Chinese/evaluation/val_hires/ \
    --msk_ratio=0.0 \
    --msk_ratio=0.7 \
    --msk_type=datasets/Chinese/evaluation/random_val
