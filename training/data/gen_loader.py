import numpy as np
import cv2
import os
import PIL
import torch
from .dataset import Dataset
from . import mask_generator
from . import lama_mask_generator_test as lama_mask_generator
import os.path as osp


# 用于加载和处理图像数据集，
class ImageDataset(Dataset):

    def __init__(self,
                 img_path,  # Path to images.
                 resolution=256,  # Ensure specific resolution, None = highest available.
                 msk_ratio=None,  # Masked ratio for freeform masks
                 lama_cfg=None,  # Lama masks config file
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self.sz = resolution
        self.img_path = img_path
        self._type = 'dir'
        self.files = []
        self.idx = 0
        self.is_comod = msk_ratio is not None
        self.mask_ratio = msk_ratio
        # 如果不使用自由形状掩码模式 (is_comod 为 False)，则初始化 LAMA 掩码生成器。
        if not self.is_comod:
            self.lama_mask_generator = lama_mask_generator.get_mask_generator(kind=lama_cfg['kind'],
                                                                              cfg=lama_cfg['mask_gen_kwargs'])
            self.iter = 0

        self._all_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in
                            os.walk(self.img_path) for fname in files]
        PIL.Image.init()
        self._image_fnames = sorted(os.path.join(self.img_path, fname) for fname in self._all_fnames if
                                    self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.files = []

        for f in self._image_fnames:
            if not '_mask' in f:
                self.files.append(f)

        self.files = sorted(self.files)

    # 返回数据集中图像文件的数量。
    def __len__(self):
        return len(self.files)

    @staticmethod  # 静态方法，用于获取文件的扩展名，并将其转换为小写。
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    # 接受文件名 fn，使用 PIL 库打开图像文件，并转换为 RGB 模式后返回。
    def _load_image(self, fn):
        return PIL.Image.open(fn).convert('RGB')

    # 根据索引 idx 获取数据集中的图像和相应的掩码。获取图像文件名和扩展名，并加载图像并调整大小为指定的 resolution。
    # 如果处于自由形状掩码模式 (is_comod 为 True)，则调用 mask_generator.generate_random_mask 生成随机掩码。
    # 否则，调用预先初始化的 LAMA 掩码生成器生成掩码，并更新迭代器 iter。返回处理后的图像数据 rgb、文件名和掩码 mask。

    def _get_image(self, idx):
        fname = self.files[idx]
        ext = self._file_ext(fname)

        rgb = np.array(self._load_image(fname))  # uint8
        rgb = cv2.resize(rgb,
                         (self.sz, self.sz), interpolation=cv2.INTER_AREA)

        if self.is_comod:
            mask = mask_generator.generate_random_mask(s=self.sz, hole_range=self.mask_ratio)
        else:
            mask = self.lama_mask_generator(shape=(self.sz, self.sz), iter_i=self.iter)
            self.iter += 1

        return rgb, fname.split('/')[-1].replace(ext, ''), mask

    # 根据索引 idx 获取数据集中的图像和相应的擦除后的图像、掩码和文件名。调用 _get_image 方法获取原始图像 rgb 和掩码 mask。
    # 将 mask 转换为 PyTorch 张量 mask_tensor，并对 rgb 进行数据类型转换和归一化处理。复制 rgb 到 rgb_erased，并根据 mask_tensor 擦除图像的部分像素值。
    # 返回处理后的图像数据 rgb、擦除后的图像数据 rgb_erased、掩码 mask_tensor 和文件名 fname。

    def __getitem__(self, idx):
        rgb, fname, mask = self._get_image(idx)  # modal, uint8 {0, 1}
        rgb = rgb.transpose(2, 0, 1)  # 将通道放到第一个维度

        mask_tensor = torch.from_numpy(mask).to(torch.float32)  # 转换为 PyTorch 张量
        rgb = torch.from_numpy(rgb.astype(np.float32))
        rgb = (rgb.to(torch.float32) / 127.5 - 1)  # 归一化到 [-1, 1] 的范围
        rgb_erased = rgb.clone()
        rgb_erased = rgb_erased * (1 - mask_tensor)  # erase rgb
        rgb_erased = rgb_erased.to(torch.float32)

        return rgb, rgb_erased, mask_tensor, fname


def collate_fn(data):
    """Creates mini-batch tensors from the list of images.
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list
            - image: torch tensor of shape (3, 256, 256).
            
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        
    """

    rgbs, rgbs_erased, mask_tensors, fnames = zip(*data)

    rgbs = list(rgbs)
    rgbs_erased = list(rgbs_erased)
    mask_tensors = list(mask_tensors)
    fnames = list(fnames)

    return torch.stack(rgbs, dim=0), torch.stack(rgbs_erased, dim=0), torch.stack(mask_tensors, dim=0), fnames


# 用于创建用于训练或测试的 PyTorch 数据加载器。接受图像路径 img_path、分辨率 resolution、掩码比例 msk_ratio 和 LAMA 配置文件 lama_cfg 作为输入。
# 创建一个 ImageDataset 实例 ds，并传入相关参数。
# 使用 torch.utils.data.DataLoader 创建数据加载器 data_loader，设置批次大小为 batch_size=1，不打乱数据，使用单个进程加载数据，并使用自定义的 collate_fn 处理数据。
# 返回创建的数据加载器 data_loader
def get_loader(img_path, resolution, msk_ratio, lama_cfg):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    ds = ImageDataset(img_path=img_path, resolution=resolution, msk_ratio=msk_ratio, lama_cfg=lama_cfg)

    data_loader = torch.utils.data.DataLoader(dataset=ds,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader
