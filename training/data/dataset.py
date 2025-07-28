import os
import numpy as np
import PIL.Image
import json
import torch
import dnnlib
import dnnlib
import cv2
from icecream import ic
from . import mask_generator
import os.path as osp
import matplotlib.pyplot as plt
from icecream import ic
import matplotlib.cm as cm
import copy
import albumentations as A
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).其中 N 是图像数量，C 是通道数，H 是高度，W 是宽度。
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):#获取原始标签，如果标签为空且启用标签，则加载原始标签，否则创建一个零标签数组。确保标签的类型和形状正确
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass 供子类重载，用于关闭数据集。
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass 供子类重载，用于加载原始图像
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass 供子类重载，用于加载原始标签。
        raise NotImplementedError

    def __getstate__(self):#用于对象序列化时的状态获取。
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):#对象销毁时的清理工作。
        try:
            self.close()
        except:
            pass

    def __len__(self):#返回数据集中图像的数量。
        return self._raw_idx.size

    def __getitem__(self, idx):#给定索引，检索图像及其标签。处理包括：加载图像并检查其类型和形状。如果启用 xflip，则对图像进行水平翻转。返回图像和标签。
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):#获取指定索引的标签。如果标签类型为 int64，则将其转换为 one-hot 编码。
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):#获取指定索引的数据详情，包括原始索引、翻转标志和原始标签。
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):#返回图像形状，格式为 (C, H, W)。
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):#返回标签形状。如果标签为 int64 类型，则返回标签的最大值加一作为 one-hot 编码的维度。
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):#返回标签的维度。
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):#检查数据集中是否有标签。
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):#检查数据集的标签是否为 one-hot 编码
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageDataset(Dataset):
    
    def __init__(self,
        img_path,                   # Path to images.
        resolution      = None,     # Ensure specific resolution, None = highest available.
        **super_kwargs,             # Additional arguments for the Dataset base class.传递给父类 Dataset 的额外参数。
    ):
        self.sz = resolution
        self.img_path = img_path
        self._type = 'dir'
        self.files = []

        self._all_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in os.walk(self.img_path) for fname in files]
        PIL.Image.init()
        self._image_fnames = sorted(os.path.join(self.img_path,fname) for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self.files = []
        
        for f in self._image_fnames:
            if not '_mask' in f:
                self.files.append(f)
        
        self.files = sorted(self.files)
#使用 albumentations 库定义了一系列图像增强变换：
        self.transform = A.Compose([
        A.PadIfNeeded(min_height=self.sz, min_width=self.sz),#填充图像到指定的最小高度和宽度。
        A.OpticalDistortion(),#应用光学畸变。
        A.RandomCrop(height=self.sz, width=self.sz),#随机裁剪到指定的高度和宽度。
        A.HorizontalFlip(),#水平翻转图像。
        A.CLAHE(),#应用自适应直方图均衡化。
        A.ToFloat()#将图像转换为浮点数格式。
    ])

        name = os.path.splitext(os.path.basename(self.img_path))[0]
        raw_shape = [len(self.files)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __len__(self):
        return len(self.files)

    def _load_image(self, fn):#从文件中加载图像并转换为 RGB 格式。
        return PIL.Image.open(fn).convert('RGB')
    
    @staticmethod
    def _file_ext(fname):#获取文件名的扩展名的静态方法。
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, raw_idx):#给定原始索引，加载图像并应用定义的增强变换。转换后的图像从 HWC 格式转换为 CHW 格式。
        fname = self.files[raw_idx]
        image = np.array(PIL.Image.open(fname).convert('RGB'))
        image = self.transform(image=image)['image']
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image
    
    def _load_raw_labels(self):#加载原始标签数据。如果标签文件 dataset.json 不存在，则返回 None。否则，加载并解析标签数据。
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _get_image(self, idx):#给定索引，加载图像并生成随机掩码，应用变换后返回图像和掩码。图像使用 self._load_image 方法加载，并应用定义的变换。掩码使用 mask_generator.generate_random_mask 方法生成。
        fname = self.files[idx]
        mask = mask_generator.generate_random_mask(s=self.sz, hole_range=[0.1,0.7])

        rgb = np.array(self._load_image(fname)) # uint8
        rgb = self.transform(image=rgb)['image']
        rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)
        
        return rgb, mask
        
    def __getitem__(self, idx):#给定索引，检索图像及其掩码，并返回图像、掩码和标签。将图像转换为 CHW 格式。返回图像、掩码和标签。
        rgb, mask = self._get_image(idx) # modal, uint8 {0, 1}
        rgb = rgb.transpose(2,0,1)

        return rgb, mask, super().get_label(idx)