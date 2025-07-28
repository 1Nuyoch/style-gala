"""Generate images using pretrained network pickle."""
import sys
import os
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch
# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.data.gen_loader import get_loader

import warnings
warnings.filterwarnings("ignore")
from colorama import init
from colorama import Fore, Style
from icecream import ic
init(autoreset=True)
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import yaml

#----------------------------------------------------------------------------
    
def denormalize(tensor):#用于对张量进行去标准化操作。张量 tensor 被还原到原始图像的像素值范围（0-255）
    pixel_mean = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    pixel_std = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    denormalizer = lambda x: torch.clamp((x * pixel_std) + pixel_mean, 0, 255.)

    return denormalizer(tensor)

def visualize_gen(i, img, inv_mask, msk_type):#负责将生成的图像和掩码保存为 PNG 文件。使用 matplotlib 将图像保存，使用 PIL 将掩码保存
    lo, hi = [-1, 1]
    
    comp_img = np.asarray(img[0], dtype=np.float32).transpose(1, 2, 0)
    comp_img = (comp_img - lo) * (255 / (hi - lo))
    comp_img = np.rint(comp_img).clip(0, 255).astype(np.uint8)
    plt.imsave(f'{msk_type}/' + i + '.png', comp_img / 255)
    
    inv_mask = torch.stack([inv_mask[0] * torch.tensor(255.)]*3, dim=0).squeeze(1)
    inv_mask = np.asarray(inv_mask, dtype=np.float32).transpose(1, 2, 0)
    inv_mask = np.rint(inv_mask).clip(0, 255).astype(np.uint8)

    mask = PIL.Image.fromarray(inv_mask)
    mask.save(f'{msk_type}/' + i + '_mask000.png')
    plt.close()


def create_folders(msk_type):#如果指定的文件夹不存在，则创建该文件夹。
    if not os.path.exists(f'{msk_type}'):
        os.makedirs(f'{msk_type}')

#----------------------------------------------------------------------------
#使用随机种子确保生成结果的可重复性。根据提供的分辨率调整图像大小，若未提供则默认为256。加载配置文件 lama_cfg 以生成掩码，若未提供则使用默认掩码比例 msk_ratio。
#调用 get_loader 函数获取图像加载器（假定 get_loader 已定义），读取图像并生成掩码。为每张图像生成掩码并保存，调用 visualize_gen 函数将生成的图像和掩码保存到文件夹中。
@click.command()
@click.pass_context
@click.option('--img_data', help='Training images (directory)', metavar='PATH', required=True)
@click.option('--msk_type', help='mask description', required=True)
@click.option('--lama_cfg', help='lama mask config')
@click.option('--msk_ratio', help='comodgan mask ratio', multiple=True)
@click.option('--resolution', help='Res of train [default: 256]', type=int, metavar='INT')
@click.option('--num', help='Number of train [default: 10]', type=int, metavar='INT')
def generate_images(#使用 click 库定义的命令行接口。生成用于训练的图像和掩码
    ctx: click.Context,
    img_data: str,
    msk_type: str,
    msk_ratio: list,
    lama_cfg: str,
    resolution: int,
    num: int,
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if resolution is None:
        resolution = 256
    
    if lama_cfg is not None:
        lama_cfg = yaml.safe_load(open(lama_cfg))
        msk_ratio = None
    
    if msk_ratio is not None:
        msk_ratio = [float(x) for x in msk_ratio]
    
    if lama_cfg is None and msk_ratio is None:
        msk_ratio = [0.7, 0.9]

    
    dataloader = get_loader(img_path=img_data, resolution=resolution, msk_ratio=msk_ratio, lama_cfg=lama_cfg)

    create_folders(msk_type)
    
    for _, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc='Generating Evaluation data...'):

        images, _, invisible_masks, fnames = data

        mask  = invisible_masks
        fname = fnames[0]

        visualize_gen(fname, images, mask, msk_type)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter调用 generate_images 函数启动整个图像生成和处理流程。

#----------------------------------------------------------------------------
