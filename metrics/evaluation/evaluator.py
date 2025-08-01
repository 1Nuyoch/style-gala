import logging
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from metrics.evaluation.utils import move_to_device

LOGGER = logging.getLogger(__name__)

#定义 InpaintingEvaluator 类的构造函数，初始化评估器的各种参数。dataset：包含图像和掩码的 torch.utils.data.Dataset 对象。scores：一个字典，包含评估指标的名称和对应的评估对象。
#area_grouping：是否根据掩码的面积分组进行评估。bins：分组的数量，分组由 np.linspace(0., 1., bins + 1) 生成。batch_size：数据加载器的批量大小。device：使用的设备（例如 'cuda'）。
#integral_func：综合评估函数。integral_title：综合评估函数的标题。clamp_image_range：图像的值范围。
class InpaintingEvaluator():
    def __init__(self, dataset, scores, area_grouping=True, bins=10, batch_size=32, device='cuda',
                 integral_func=None, integral_title=None, clamp_image_range=None):
        """
        :param dataset: torch.utils.data.Dataset which contains images and masks
        :param scores: dict {score_name: EvaluatorScore object}
        :param area_grouping: in addition to the overall scores, allows to compute score for the groups of samples
            which are defined by share of area occluded by mask
        :param bins: number of groups, partition is generated by np.linspace(0., 1., bins + 1)
        :param batch_size: batch_size for the dataloader
        :param device: device to use
        """
        self.scores = scores
        self.dataset = dataset

        self.area_grouping = area_grouping
        self.bins = bins
        self.batch_size = batch_size

        self.device = torch.device(device)

#创建数据加载器 self.dataloader，不打乱数据，批量大小为 batch_size。
        self.dataloader = DataLoader(self.dataset, shuffle=False, batch_size=batch_size)

        self.integral_func = integral_func
        self.integral_title = integral_title
        self.clamp_image_range = clamp_image_range

    def _get_bin_edges(self):#用于计算分组的边界。
        bin_edges = np.linspace(0, 1, self.bins + 1)#使用 np.linspace 生成 bins + 1 个均匀分布的值，作为分组边界。

#计算每个分组的名称，以百分比表示num_digits 确定百分比的小数位数。使用 round 和格式化字符串生成分组名称。
        num_digits = max(0, math.ceil(math.log10(self.bins)) - 1)
        interval_names = []
        for idx_bin in range(self.bins):
            start_percent, end_percent = round(100 * bin_edges[idx_bin], num_digits), \
                                         round(100 * bin_edges[idx_bin + 1], num_digits)
            start_percent = '{:.{n}f}'.format(start_percent, n=num_digits)
            end_percent = '{:.{n}f}'.format(end_percent, n=num_digits)
            interval_names.append("{0}-{1}%".format(start_percent, end_percent))

#遍历数据加载器，计算每个批次掩码的面积，并确定其所属的分组。使用 np.searchsorted 找到每个面积所属的分组索引。将所有批次的分组索引合并到一个数组中。
        groups = []
        for batch in tqdm.auto.tqdm(self.dataloader, desc='edges'):
            mask = batch['mask']
            batch_size = mask.shape[0]
            area = mask.to(self.device).reshape(batch_size, -1).mean(dim=-1)
            bin_indices = np.searchsorted(bin_edges, area.detach().cpu().numpy(), side='right') - 1
            # corner case: when area is equal to 1, bin_indices should return bins - 1, not bins for that element
            bin_indices[bin_indices == self.bins] = self.bins - 1
            groups.append(bin_indices)
        groups = np.hstack(groups)

        return groups, interval_names#返回分组索引和分组名称。

    def evaluate(self, model=None):
        """
        :param model: callable with signature (image_batch, mask_batch); should return inpainted_batch
        :return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
        """
        #初始化一个空字典 results 存储评估结果。如果启用了区域分组，调用 _get_bin_edges 函数获取分组索引和分组名称
        results = dict()
        if self.area_grouping:
            groups, interval_names = self._get_bin_edges()
        else:
            groups = None

#遍历所有评分指标，使用 torch.no_grad() 上下文管理器，禁用梯度计算。遍历数据加载器，将批次数据移动到设备上。如果指定了图像值范围，使用 torch.clamp 限制图像值。
#如果没有指定模型，则从批次数据中获取预计算的修复结果，否则使用模型生成修复结果。计算修复结果的评分。获取评分的总体结果和分组结果
        for score_name, score in tqdm.auto.tqdm(self.scores.items(), desc='scores'):
            score.to(self.device)
            with torch.no_grad():
                score.reset()
                for batch in tqdm.auto.tqdm(self.dataloader, desc=score_name, leave=False):
                    batch = move_to_device(batch, self.device)
                    image_batch, mask_batch = batch['image'], batch['mask']
                    if self.clamp_image_range is not None:
                        image_batch = torch.clamp(image_batch,
                                                  min=self.clamp_image_range[0],
                                                  max=self.clamp_image_range[1])
                    if model is None:
                        assert 'inpainted' in batch, \
                            'Model is None, so we expected precomputed inpainting results at key "inpainted"'
                        inpainted_batch = batch['inpainted']
                    else:
                        inpainted_batch = model(image_batch, mask_batch)
                    score(inpainted_batch, image_batch, mask_batch)
                total_results, group_results = score.get_value(groups=groups)

            results[(score_name, 'total')] = total_results#将总体结果和分组结果存储到 results 字典中。
            if groups is not None:#果指定了综合评估函数，计算综合评分，并存储到 results 字典中。
                for group_index, group_values in group_results.items():
                    group_name = interval_names[group_index]
                    results[(score_name, group_name)] = group_values

        if self.integral_func is not None:
            results[(self.integral_title, 'total')] = dict(mean=self.integral_func(results))

        return results
    
    def dist_evaluate(self, device, num_gpus, rank, model=None):
        """
        分布式评估函数 dist_evaluate，用于在多个 GPU 上进行评估。
        :param model: callable with signature (image_batch, mask_batch); should return inpainted_batch
        :return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
        """
        #计算数据集的长度，并根据 GPU 数量和当前 GPU 的索引（rank）划分数据集。创建数据加载器，使用指定的参数。
        num_items = len(self.dataset)
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        # Main loop.
        item_subset = [(i * num_gpus + rank) % num_items for i in range((num_items - 1) // num_gpus + 1)]
        dataloader = torch.utils.data.DataLoader(dataset=self.dataset, sampler=item_subset, batch_size=self.batch_size, **data_loader_kwargs)

        #如果启用了区域分组，调用_get_bin_edges函数获取分组索引和分组名称。
        results = dict()
        if self.area_grouping:
            groups, interval_names = self._get_bin_edges()
        else:
            groups = None

#遍历所有评分指标，使用 torch.no_grad() 上下文管理器，禁用梯度计算。遍历数据加载器，将批次数据移动到设备上。如果指定了图像值范围，使用 torch.clamp 限制图像值。
#如果没有指定模型，则从批次数据中获取预计算的修复结果，否则使用模型生成修复结果。计算修复结果的评分。获取评分的总体结果和分组结果。
        for score_name, score in self.scores.items():
            score.to(device)
            with torch.no_grad():
                score.reset()
                for _, batch in tqdm.auto.tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=score_name + f' on GPU: {device}'):
                    batch = move_to_device(batch, device)
                    image_batch, mask_batch = batch['image'], batch['mask']
                    if self.clamp_image_range is not None:
                        image_batch = torch.clamp(image_batch,
                                                  min=self.clamp_image_range[0],
                                                  max=self.clamp_image_range[1])
                    if model is None:
                        assert 'inpainted' in batch, \
                            'Model is None, so we expected precomputed inpainting results at key "inpainted"'
                        inpainted_batch = batch['inpainted']
                    else:
                        inpainted_batch = model(image_batch, mask_batch)
                    score(inpainted_batch, image_batch, mask_batch)
                total_results, group_results = score.get_value(groups=groups)

            results[(score_name, 'total')] = total_results
            if groups is not None:
                for group_index, group_values in group_results.items():
                    group_name = interval_names[group_index]
                    results[(score_name, group_name)] = group_values

#如果指定了综合评估函数，计算综合评分，并存储到 results 字典中。
        if self.integral_func is not None:
            results[(self.integral_title, 'total')] = dict(mean=self.integral_func(results))

        return results


#定义 ssim_fid100_f1 函数，计算 SSIM 和 FID 综合评分。从 metrics 中获取 SSIM 和 FID 的平均值。计算相对 FID（fid_rel），确保值在 [0, 1] 范围内。计算 F1 综合评分，返回结果
def ssim_fid100_f1(metrics, fid_scale=100):
    ssim = metrics[('ssim', 'total')]['mean']
    fid = metrics[('fid', 'total')]['mean']
    fid_rel = max(0, fid_scale - fid) / fid_scale
    f1 = 2 * ssim * fid_rel / (ssim + fid_rel + 1e-3)
    return f1

#定义 lpips_fid100_f1 函数，计算 LPIPS 和 FID 综合评分。从 metrics 中获取 LPIPS 的平均值，并取反，使其值越大越好。计算相对 FID（fid_rel），确保值在 [0, 1] 范围内。计算 F1 综合评分，返回结果。
def lpips_fid100_f1(metrics, fid_scale=100):
    neg_lpips = 1 - metrics[('lpips', 'total')]['mean']  # invert, so bigger is better
    fid = metrics[('fid', 'total')]['mean']
    fid_rel = max(0, fid_scale - fid) / fid_scale
    f1 = 2 * neg_lpips * fid_rel / (neg_lpips + fid_rel + 1e-3)
    return f1



class InpaintingEvaluatorOnline(nn.Module):
    #定义 InpaintingEvaluatorOnline 类的构造函数，初始化评估器的各种参数。scores：一个字典，包含评估指标的名称和对应的评估对象。bins：分组的数量，分组由 np.linspace(0., 1., bins + 1) 生成。
    #image_key：原始图像的键名。inpainted_key：修复后图像的键名。integral_func：综合评估函数。integral_title：综合评估函数的标题。clamp_image_range：图像的值范围
    def __init__(self, scores, bins=10, image_key='image', inpainted_key='inpainted',
                 integral_func=None, integral_title=None, clamp_image_range=None):
        """
        :param scores: dict {score_name: EvaluatorScore object}
        :param bins: number of groups, partition is generated by np.linspace(0., 1., bins + 1)
        :param device: device to use
        """
        super().__init__()#调用父类的构造函数。记录日志，表明初始化开始。
        LOGGER.info(f'{type(self)} init called')
        self.scores = nn.ModuleDict(scores)
        self.image_key = image_key
        self.inpainted_key = inpainted_key
        self.bins_num = bins
        self.bin_edges = np.linspace(0, 1, self.bins_num + 1)

#计算每个分组的名称，以百分比表示。num_digits 确定百分比的小数位数。使用 round 和格式化字符串生成分组名称
        num_digits = max(0, math.ceil(math.log10(self.bins_num)) - 1)
        self.interval_names = []
        for idx_bin in range(self.bins_num):
            start_percent, end_percent = round(100 * self.bin_edges[idx_bin], num_digits), \
                                         round(100 * self.bin_edges[idx_bin + 1], num_digits)
            start_percent = '{:.{n}f}'.format(start_percent, n=num_digits)
            end_percent = '{:.{n}f}'.format(end_percent, n=num_digits)
            self.interval_names.append("{0}-{1}%".format(start_percent, end_percent))

#初始化 groups 列表，用于存储分组索引。初始化 integral_func、integral_title 和 clamp_image_range。记录日志，表明初始化完成
        self.groups = []
        self.integral_func = integral_func
        self.integral_title = integral_title
        self.clamp_image_range = clamp_image_range

        LOGGER.info(f'{type(self)} init done')

#定义一个私有函数 _get_bins，用于计算每个掩码批次的分组索引。batch_size：批次大小。area：计算每个掩码的面积。bin_indices：找到每个面积所属的分组索引，并确保索引在 [0, self.bins_num - 1] 范围内。
    def _get_bins(self, mask_batch):
        batch_size = mask_batch.shape[0]
        area = mask_batch.view(batch_size, -1).mean(dim=-1).detach().cpu().numpy()
        bin_indices = np.clip(np.searchsorted(self.bin_edges, area) - 1, 0, self.bins_num - 1)
        return bin_indices

#定义前向传播函数 forward，计算并累积批次的评估指标。batch：包含必须字段 mask、image 和 inpainted 的批次字典。使用 torch.no_grad() 禁用梯度计算。
#获取原始图像、掩码和修复后图像的批次数据。如果指定了图像值范围，使用 torch.clamp 限制图像值。计算掩码的分组索引，并将其添加到 groups 列表中
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Calculate and accumulate metrics for batch. To finalize evaluation and obtain final metrics, call evaluation_end
        :param batch: batch dict with mandatory fields mask, image, inpainted (can be overriden by self.inpainted_key)
        """
        result = {}
        with torch.no_grad():
            image_batch, mask_batch, inpainted_batch = batch[self.image_key], batch['mask'], batch[self.inpainted_key]
            if self.clamp_image_range is not None:
                image_batch = torch.clamp(image_batch,
                                          min=self.clamp_image_range[0],
                                          max=self.clamp_image_range[1])
            self.groups.extend(self._get_bins(mask_batch))

            for score_name, score in self.scores.items():#遍历所有评分指标，计算每个评分，并将结果存储到 result 字典中。返回评估结果。
                result[score_name] = score(inpainted_batch, image_batch, mask_batch)
        return result

    def process_batch(self, batch: Dict[str, torch.Tensor]):#定义批处理处理函数 process_batch，调用 forward 函数处理批次数据。
        return self(batch)

    def evaluation_end(self, states=None):
        """:return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
            定义评估结束函数 evaluation_end，在评估结束时调用，返回最终的评估结果。记录日志，表明评估结束函数被调用。将 groups 列表转换为 numpy 数组
        """
        LOGGER.info(f'{type(self)}: evaluation_end called')

        self.groups = np.array(self.groups)

#初始化一个空字典 results 存储评估结果。遍历所有评分指标，计算每个评分的总体结果和分组结果。如果 states 参数不为空，从中获取当前评分的状态。将总体结果和分组结果存储到 results 字典中
        results = {}
        for score_name, score in self.scores.items():
            LOGGER.info(f'Getting value of {score_name}')
            cur_states = [s[score_name] for s in states] if states is not None else None
            total_results, group_results = score.get_value(groups=self.groups, states=cur_states)
            LOGGER.info(f'Getting value of {score_name} done')
            results[(score_name, 'total')] = total_results

            for group_index, group_values in group_results.items():
                group_name = self.interval_names[group_index]
                results[(score_name, group_name)] = group_values

        if self.integral_func is not None:#如果指定了综合评估函数，计算综合评分，并存储到 results 字典中。
            results[(self.integral_title, 'total')] = dict(mean=self.integral_func(results))

#记录日志，表明重置评分开始，重置 groups 列表。重置每个评分指标的状态。记录日志，表明重置评分完成。记录日志，表明评估结束函数完成。返回评估结果。
        LOGGER.info(f'{type(self)}: reset scores')
        self.groups = []
        for sc in self.scores.values():
            sc.reset()
        LOGGER.info(f'{type(self)}: reset scores done')

        LOGGER.info(f'{type(self)}: evaluation_end done')
        return results
