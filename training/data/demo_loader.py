from tabnanny import filename_only
import numpy as np
import cv2
import os
import PIL
import torch
from .dataset import Dataset

class ImageDataset(Dataset):
    
    def __init__(self,
        img_path,                   # Path to images.
        resolution      = None,     # Ensure specific resolution, None = highest available.
        **super_kwargs,             # Additional arguments for the Dataset base class.
    ):
        self.sz = resolution
        self.img_path = img_path
        self._type = 'dir'
        self.files = []
        self.idx = 0

        self._all_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in os.walk(self.img_path) for fname in files]
        PIL.Image.init()
        self._image_fnames = sorted(os.path.join(self.img_path,fname) for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self.files = []
        
        for f in self._image_fnames:
            # print(f)
            if not '_mask' in f:
                self.files.append(f)
        
        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)
    
    @staticmethod #获取文件名的扩展名的静态方法
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_image(self, fn):#从文件加载图像并将其转换为 RGB 格式。
        return PIL.Image.open(fn).convert('RGB')
    
    def _get_image(self, idx):#给定索引，加载相应的图像及其掩码，调整大小后返回调整后的 RGB 图像、文件名和掩码。
        # imgfn, seg_map, img_id = self.data_reader.get_image(idx)
        
        fname = self.files[idx]
        ext = self._file_ext(fname)

        mask = np.array(self._load_image(fname.replace(ext, f'_mask{ext}')).convert('L')) / 255
        # print(mask)
        mask = cv2.resize(mask,
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        rgb = np.array(self._load_image(fname)) # uint8
        rgb = cv2.resize(rgb,
            (self.sz, self.sz), interpolation=cv2.INTER_AREA)

        return rgb, fname.split('/')[-1].replace(ext, ''), mask
        
    def __getitem__(self, idx):
        rgb, fname, mask = self._get_image(idx) # modal, uint8 {0, 1}
        rgb = rgb.transpose(2,0,1)

        mask_tensor = torch.from_numpy(mask).to(torch.float32)
        mask_tensor = mask_tensor.unsqueeze(0)
        rgb = torch.from_numpy(rgb.astype(np.float32))
        rgb = (rgb.to(torch.float32) / 127.5 - 1)
        rgb_erased = rgb.clone()
        rgb_erased = rgb_erased * (1 - mask_tensor) # erase rgb
        rgb_erased = rgb_erased.to(torch.float32)
        
        return rgb, rgb_erased, mask_tensor, fname
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of images.一个自定义的合并函数，用于 DataLoader
    接受一批数据（一个由 __getitem__ 返回的元组列表），并：
    将批次解压为 RGB 图像、擦除的 RGB 图像、掩码张量和文件名的单独列表。
    沿新批次维度堆叠这些张量列表，创建每个组件的批处理张量。
    返回这些批处理张量和文件名列表。
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
    
def get_loader(img_path, resolution):
    """Returns torch.utils.data.DataLoader for custom coco dataset.创建并返回一个自定义数据集的 DataLoader。
    使用提供的图像路径和分辨率实例化 ImageDataset。创建一个 DataLoader，指定批次大小为 1，不进行随机打乱，并使用自定义的 collate_fn。返回 DataLoader"""

    ds = ImageDataset(img_path=img_path, resolution=resolution)
    
    data_loader = torch.utils.data.DataLoader(dataset=ds, 
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader