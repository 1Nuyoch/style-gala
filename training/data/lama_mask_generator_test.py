import math
import random
import hashlib
import logging
from enum import Enum

import cv2
import numpy as np

from utils.data_utils import LinearRamp
from metrics.evaluation.masks.mask import SegmentationMask

LOGGER = logging.getLogger(__name__)


# 枚举定义了三种绘制形状的方法：直线 (LINE)、圆形 (CIRCLE) 和正方形 (SQUARE)，分别用字符串 'line'、'circle' 和 'square' 表示
class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


# 用于生成随机形状的掩码
def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


# 构造函数初始化随机形状掩码生成器的参数
class RandomIrregularMaskGenerator:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    # 实现了类的可调用接口，用于生成随机形状的掩码,raw_image 是原始图像，可选参数，目前未被使用,根据 ramp 控制当前的 max_len、max_width 和 max_times 参数。,
    # iter_i 是迭代次数，用于控制形状参数的变化。，调用make_random_irregular_mask函数生成随机形状掩码，并返回生成的掩码。
    def __call__(self, shape, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(shape, max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)


# 用于生成随机矩形形状的掩码。
# margin 控制矩形离边界的最小间距。bbox_min_size 和 bbox_max_size 控制矩形的宽度和高度的最小和最大尺寸。min_times
# 和max_times 控制生成矩形的次数范围。在随机位置生成 times 个随机大小的矩形，并将它们绘制到 mask 上返回生成的掩码 mask，类型为 np.float32，维度为 (1, height, width)
def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


# 用于生成随机矩形形状的掩码。ramp 参数是一个可选的线性斜坡对象，用于控制参数随时间变化。
class RandomRectangleMaskGenerator:
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    # 使对象实例可调用，接受 shape 参数表示掩码的形状，返回生成的随机矩形掩码，调用了 make_random_rectangle_mask 函数。
    def __call__(self, shape, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(shape, margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=self.min_times,
                                          max_times=cur_max_times)


# 用于生成超分辨率掩码，这种掩码模拟了图像的分辨率增强效果。使用随机生成的步长和宽度，在随机偏移的位置生成水平和垂直方向的直线，形成网格状的掩码
# 返回生成的超分辨率掩码，类型为 np.float32，维度为 (1, height, width)。
def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)

    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)

    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]


# 用于生成超分辨率掩码的生成器。字典 kwargs，用于配置 make_random_superres_mask 函数的参数。
class RandomSuperresMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    # 使对象实例可调用，接受 shape 参数表示掩码的形状，返回生成的超分辨率掩码，调用了 make_random_superres_mask 函数
    def __call__(self, shape, iter_i=None):
        return make_random_superres_mask(shape, **self.kwargs)


# 用于混合生成不同类型的随机掩码。
class MixedMaskGenerator:
    def __init__(self, irregular_proba=1 / 3, hole_range=[0, 0, 0.7], irregular_kwargs=None,
                 box_proba=1 / 3, box_kwargs=None,
                 segm_proba=1 / 3, segm_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 outpainting_proba=0, outpainting_kwargs=None,
                 invert_proba=0):
        self.probas = []
        self.gens = []
        self.hole_range = hole_range

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskGenerator(**squares_kwargs))

        if superres_proba > 0:
            self.probas.append(superres_proba)
            if superres_kwargs is None:
                superres_kwargs = {}
            self.gens.append(RandomSuperresMaskGenerator(**superres_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    # 使对象实例可调用，接受 shape 参数表示掩码的形状，根据设定的生成概率随机选择一种掩码生成器生成掩码，并在一定概率下对生成的掩码进行反转。
    def __call__(self, shape, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(shape, iter_i=iter_i, raw_image=raw_image)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        if np.mean(result) <= self.hole_range[0] or np.mean(result) >= self.hole_range[1]:
            return self.__call__(shape, iter_i=iter_i, raw_image=raw_image)
        else:
            return result


# 用于生成随机的图像分割掩码。
class RandomSegmentationMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.impl = SegmentationMask(**self.kwargs)

    # 使对象实例可调用，接受输入图像 img 和可选的参数 iter_i、raw_image，以及掩码的空洞范围 hole_range。
    # 在输入图像上调用 SegmentationMask 实例的 get_masks 方法，获取多个可能的分割掩码。筛选，从筛选后的掩码列表中随机选择一个掩码作为结果返回。
    def __call__(self, img, iter_i=None, raw_image=None, hole_range=[0.0, 0.3]):

        masks = self.impl.get_masks(img)
        fil_masks = []
        for cur_mask in masks:
            if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > hole_range[1]:
                continue
            fil_masks.append(cur_mask)

        mask_index = np.random.choice(len(fil_masks),
                                      size=1,
                                      replace=False)
        mask = fil_masks[mask_index]

        return mask


# 用于生成图像分割掩码的生成器。初始化方法接受参数 hole_range，用于控制生成掩码的空洞范围。
# 创建 RandomSegmentationMaskGenerator 实例 gen，并传入 segm_kwargs 作为参数。
class SegMaskGenerator:
    def __init__(self, hole_range=[0.1, 0.2], segm_kwargs=None):
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gen = RandomSegmentationMaskGenerator(**segm_kwargs)
        self.hole_range = hole_range

    # 使对象实例可调用，接受输入图像 img 和可选的参数 iter_i、raw_image。调用 gen 实例生成图像分割掩码，返回生成的掩码。
    def __call__(self, img, iter_i=None, raw_image=None):
        result = self.gen(img=img, iter_i=iter_i, raw_image=raw_image, hole_range=self.hole_range)
        return result


# 用于生成前景分割掩码，
class FGSegmentationMaskGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.impl = SegmentationMask(**self.kwargs)

    # 使对象实例可调用，接受输入图像 img 和可选的参数 iter_i、raw_image，以及掩码的空洞范围 hole_range
    # 在输入图像上调用 SegmentationMask 实例的 get_masks 方法，获取多个可能的分割掩码。将所有有效的掩码叠加在一起，生成前景分割掩码。
    def __call__(self, img, iter_i=None, raw_image=None, hole_range=[0.0, 0.3]):

        masks = self.impl.get_masks(img)
        mask = masks[0]
        for cur_mask in masks:
            if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > hole_range[1]:
                continue
            mask += cur_mask

        mask = mask > 0
        return mask


# 用于生成背景分割掩码。创建 FGSegmentationMaskGenerator 实例 gen，并传入 segm_kwargs 作为参数。创建 MixedMaskGenerator 实例
# bg_mask_gen，使用指定的配置参数 cfg。
class SegBGMaskGenerator:
    def __init__(self, hole_range=[0.1, 0.2], segm_kwargs=None):
        if segm_kwargs is None:
            segm_kwargs = {}
        self.gen = FGSegmentationMaskGenerator(**segm_kwargs)
        self.hole_range = hole_range
        self.cfg = {
            'irregular_proba': 1,
            'hole_range': [0.0, 1.0],
            'irregular_kwargs': {
                'max_angle': 4,
                'max_len': 250,
                'max_width': 150,
                'max_times': 3,
                'min_times': 1,
            },
            'box_proba': 0,
            'box_kwargs': {
                'margin': 10,
                'bbox_min_size': 30,
                'bbox_max_size': 150,
                'max_times': 4,
                'min_times': 1,
            }
        }
        self.bg_mask_gen = MixedMaskGenerator(**self.cfg)

    # 使对象实例可调用，接受输入图像 img 和可选的参数 iter_i、raw_image。调用 gen 实例生成前景分割掩码 mask_fg。
    # 计算背景比例 bg_ratio，并调用 bg_mask_gen 生成背景掩码，将生成的背景掩码减去前景掩码，确保掩码均值在指定的空洞范围内，若不在范围内则递归调用call方法重新生成
    def __call__(self, img, iter_i=None, raw_image=None):
        shape = img.shape[:2]
        mask_fg = self.gen(img=img, iter_i=iter_i, raw_image=raw_image, hole_range=self.hole_range)
        bg_ratio = 1 - np.mean(mask_fg)
        result = self.bg_mask_gen(shape, iter_i=iter_i, raw_image=raw_image)
        result = result - mask_fg
        if np.mean(result) <= self.hole_range[0] * bg_ratio or np.mean(result) >= self.hole_range[1] * bg_ratio:
            return self.__call__(shape, iter_i=iter_i, raw_image=raw_image)
        return result

#get_mask_generator 函数根据指定的 kind 返回相应的掩码生成器类实例。
def get_mask_generator(kind, cfg=None):
    if kind is None:
        kind = "mixed"

    if cfg is None:
        cfg = {
            'irregular_proba': 1,
            'hole_range': [0.0, 0.7],
            'irregular_kwargs': {
                'max_angle': 4,
                'max_len': 200,
                'max_width': 100,
                'max_times': 5,
                'min_times': 1,
            },
            'box_proba': 1,
            'box_kwargs': {
                'margin': 10,
                'bbox_min_size': 30,
                'bbox_max_size': 150,
                'max_times': 4,
                'min_times': 1,
            },
            'segm_proba': 0, }

    if kind == "mixed":
        cl = MixedMaskGenerator
    elif kind == "segmentation":
        cl = SegBGMaskGenerator
    else:
        raise NotImplementedError(f"No such generator kind = {kind}")
    return cl(**cfg)
