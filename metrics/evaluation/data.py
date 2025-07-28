import glob
import os

import cv2
import PIL.Image as Image
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F


def load_image(fname, mode='RGB', return_orig=False):#用于加载图像，并将其转换为指定的颜色模式和数据格式。图像文件的路径。mode：指定图像的颜色模式（默认为 'RGB'）。return_orig：决定是否返回原始图像数据（未归一化）
    img = np.array(Image.open(fname).convert(mode))#加载图像并转换为指定颜色模式，然后转为 numpy 数组。
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))#如果图像是三维的，将其维度从 (H, W, C) 转换为 (C, H, W)。c
    out_img = img.astype('float32') / 255#将图像像素值归一化到 [0, 1] 范围。
    if return_orig:#返回处理后的图像，或同时返回处理后的图像和原始图像。
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):#该函数用于计算向上取整到指定模数的值。x：输入值。mod：模数。x % mod == 0：如果 x 已经是 mod 的倍数，则直接返回 x。(x // mod + 1) * mod：否则，返回比 x 大的最小的 mod 的倍数。
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):#用于将图像填充到指定模数。
    channels, height, width = img.shape#channels, height, width = img.shape：获取图像的通道数、高度和宽度。
    out_height = ceil_modulo(height, mod)#out_height 和 out_width：计算填充后的高度和宽度。
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')#np.pad：对图像进行对称填充。


def pad_tensor_to_modulo(img, mod):#用于将 tensor 填充到指定模数。
    batch_size, channels, height, width = img.shape#batch_size, channels, height, width = img.shape：获取 tensor 的维度。
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')#F.pad：对 tensor 进行反射填充。


def scale_image(img, factor, interpolation=cv2.INTER_AREA):#用于缩放图像。
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

##img：输入图像。factor：缩放因子。interpolation：插值方法（默认为 cv2.INTER_AREA）。cv2.resize：使用 OpenCV 进行图像缩放。
    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)
    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class InpaintingDataset(Dataset):#用于图像修补的数据集。
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):#初始化函数，设置数据目录、图像后缀、填充模数和缩放因子
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))#掩码文件名列表。
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]#对应的图像文件名列表。
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):#返回数据集的长度。
        return len(self.mask_filenames)

    def __getitem__(self, i):#__getitem__：获取指定索引的数据项。
        image = load_image(self.img_filenames[i], mode='RGB')#加载图像和掩码。
        mask = load_image(self.mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)#scale_image：缩放图像和掩码。
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:#pad_img_to_modulo：填充图像和掩码。
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class OurInpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, 'mask', '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [os.path.join(self.datadir, 'img', os.path.basename(fname.rsplit('-', 1)[0].rsplit('_', 1)[0]) + '.png') for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        result = dict(image=load_image(self.img_filenames[i], mode='RGB'),
                      mask=load_image(self.mask_filenames[i], mode='L')[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class PrecomputedInpaintingResultsDataset(InpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix='_inpainted.jpg', **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = load_image(self.pred_filenames[i].replace('_mask', ''))
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result

class OurPrecomputedInpaintingResultsDataset(OurInpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix="png", **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.basename(os.path.splitext(fname)[0]) + f'_inpainted.{inpainted_suffix}')
                               for fname in self.mask_filenames]
        # self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
        #                        for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = self.file_loader(self.pred_filenames[i])

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result

class InpaintingEvalOnlineDataset(Dataset):
    def __init__(self, indir, mask_generator, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None,  **kwargs):
        self.indir = indir
        self.mask_generator = mask_generator
        self.img_filenames = sorted(list(glob.glob(os.path.join(self.indir, '**', f'*{img_suffix}' ), recursive=True)))
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, i):
        img, raw_image = load_image(self.img_filenames[i], mode='RGB', return_orig=True)
        mask = self.mask_generator(img, raw_image=raw_image)
        result = dict(image=img, mask=mask)

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
        return result