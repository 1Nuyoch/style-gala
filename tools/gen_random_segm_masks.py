#!/usr/bin/env python3
import sys
import glob
import os
import shutil
import traceback

import PIL.Image as Image
import numpy as np
from joblib import Parallel, delayed

# 获取脚本所在目录的父目录
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))

# 将根目录添加到 sys.path
sys.path.append(root_dir)

from metrics.evaluation.masks.mask import SegmentationMask, propose_random_square_crop
from metrics.evaluation.utils import load_yaml, SmallMode
from tqdm import tqdm


class MakeManyMasksWrapper:#impl 是生成掩码的实现，variants_n 是要生成的掩码数量。
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):#get_masks 方法接收一个图像并生成 variants_n 个掩码。
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]


def process_images(src_images, indir, outdir, config):#处理输入图像生成掩码。读取图像后，根据配置文件调整图像大小，生成掩码，并过滤不合适的掩码。最后保存处理后的图像和掩码。
    mask_generator = SegmentationMask(**config.mask_generator_kwargs)

    max_tamper_area = config.get('max_tamper_area', 1)

    for infile in tqdm(src_images, desc='generating segm pairs...'):
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            # scale input image to output resolution and filter smaller images
            if min(image.size) < config.cropping.out_min_size:
                handle_small_mode = SmallMode(config.cropping.handle_small_mode)
                if handle_small_mode == SmallMode.DROP:
                    continue
                elif handle_small_mode == SmallMode.UPSCALE:
                    factor = config.cropping.out_min_size / min(image.size)
                    out_size = (np.array(image.size) * factor).round().astype('uint32')
                    image = image.resize(out_size, resample=Image.BICUBIC)
            else:
                factor = config.cropping.out_min_size / min(image.size)
                out_size = (np.array(image.size) * factor).round().astype('uint32')
                image = image.resize(out_size, resample=Image.BICUBIC)

            # generate and select masks
            src_masks = mask_generator.get_masks(image)

            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if config.cropping.out_square_crop:
                    (crop_left,
                     crop_top,
                     crop_right,
                     crop_bottom) = propose_random_square_crop(cur_mask,
                                                               min_overlap=config.cropping.crop_min_overlap)
                    cur_mask = cur_mask[crop_top:crop_bottom, crop_left:crop_right]
                    cur_image = image.copy().crop((crop_left, crop_top, crop_right, crop_bottom))
                else:
                    cur_image = image

                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue

                filtered_image_mask_pairs.append((cur_image, cur_mask))

            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                            replace=False)

            # crop masks; save masks together with input image
            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename + f'_crop{i:03d}'
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + f'_mask{i:03d}.png')
                cur_image.save(cur_basename + '.png')
        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')


def main(args):#加载配置文件并调用 process_images 处理图像。
    if not args.indir.endswith('/'):
        args.indir += '/'

    os.makedirs(args.outdir, exist_ok=True)

    config = load_yaml(args.config)

    in_files = list(glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True))
    if args.n_jobs == 0:
        process_images(in_files, args.indir, args.outdir, config)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.n_jobs + (1 if in_files_n % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_images)(in_files[start:start+chunk_size], args.indir, args.outdir, config)
            for start in range(0, len(in_files), chunk_size)
        )


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to config for dataset generation')
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')
    aparser.add_argument('--n-jobs', type=int, default=0, help='How many processes to use')
    aparser.add_argument('--ext', type=str, default='jpg', help='Input image extension')

    main(aparser.parse_args())
