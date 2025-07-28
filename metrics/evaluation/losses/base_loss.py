import logging
from abc import abstractmethod, ABC

import numpy as np
import sklearn
import sklearn.svm
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy import linalg

from training.losses.ade20k import SegmentationModule, NUM_CLASS, segm_options
from .fid.inception import InceptionV3
from .lpips import PerceptualLoss
from .ssim import SSIM

LOGGER = logging.getLogger(__name__)


def get_groupings(groups):
    """
    groups 参数是一个包含各个元素所属分组编号的数组，使用 np.unique 函数计算唯一的分组编号 label_groups 和每个分组的计数 count_groups。
    :param groups: group numbers for respective elements
    :return: dict of kind {group_idx: indices of the corresponding group elements}
    """
    label_groups, count_groups = np.unique(groups, return_counts=True)

    indices = np.argsort(groups)#np.argsort 返回 groups 数组排序后元素的索引，indices 是按照分组编号排序后的索引数组。

#初始化一个空字典 grouping。遍历每个分组编号及其计数，确定当前分组的索引范围。根据索引范围将元素分组，并存储到 grouping 字典中，字典的键是分组编号，值是该分组中元素的索引。
    grouping = dict()
    cur_start = 0
    for label, count in zip(label_groups, count_groups):
        cur_end = cur_start + count
        cur_indices = indices[cur_start:cur_end]
        grouping[label] = cur_indices
        cur_start = cur_end
    return grouping


class EvaluatorScore(nn.Module):#EvaluatorScore 类继承自 torch.nn.Module，是一个抽象类，定义了评估指标的基本接口。
    @abstractmethod
    def forward(self, pred_batch, target_batch, mask):#抽象方法，子类需要实现，用于计算评估指标。
        pass

    @abstractmethod
    def get_value(self, groups=None, states=None):#抽象方法，子类需要实现，用于获取评估结果。
        pass

    @abstractmethod
    def reset(self):#抽象方法，子类需要实现，用于重置评估状态。
        pass


class PairwiseScore(EvaluatorScore, ABC):#PairwiseScore 类继承自 EvaluatorScore 和 ABC，是一个具体的评估指标类，计算成对的评分。
    def __init__(self):#初始化 individual_values 属性，用于存储单个评估值。
        super().__init__()
        self.individual_values = None

    def get_value(self, groups=None, states=None):
        """
        用于计算和返回评估结果。states 不为空时，将其拼接成一个长向量并转换为 numpy 数组，否则使用 self.individual_values。
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        individual_values = torch.cat(states, dim=-1).reshape(-1).cpu().numpy() if states is not None \
            else self.individual_values

#计算 individual_values 的均值和标准差，存储在 total_results 字典中。
        total_results = {
            'mean': individual_values.mean(),
            'std': individual_values.std()
        }

        if groups is None:#如果没有分组信息，返回 total_results 和 None。
            return total_results, None


#如果有分组信息，初始化 group_results 字典。使用 get_groupings 函数获取分组信息。遍历每个分组，计算每个分组的均值和标准差，存储在 group_results 字典中。返回 total_results 和 group_results。
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_scores = individual_values[index]
            group_results[label] = {
                'mean': group_scores.mean(),
                'std': group_scores.std()
            }
        return total_results, group_results

    def reset(self):#重置 individual_values 属性为空列表。
        self.individual_values = []


#SSIMScore 类继承自 PairwiseScore，用于计算 SSIM（结构相似性指数）。初始化时，创建一个 SSIM 对象，用于计算 SSIM 分数，并调用 reset 方法初始化状态。
class SSIMScore(PairwiseScore):
    def __init__(self, window_size=11):
        super().__init__()
        self.score = SSIM(window_size=window_size, size_average=False).eval()
        self.reset()

#forward 方法计算预测图像和目标图像的 SSIM 分数，并将结果添加到 individual_values 中。返回当前批次的 SSIM 分数。
    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch)
        self.individual_values = np.hstack([
            self.individual_values, batch_values.detach().cpu().numpy()
        ])
        return batch_values

#LPIPSScore 类继承自 PairwiseScore，用于计算 LPIPS（感知相似性）。初始化时，创建一个 PerceptualLoss 对象，用于计算 LPIPS 分数，并调用 reset 方法初始化状态。
class LPIPSScore(PairwiseScore):
    def __init__(self, model='net-lin', net='vgg', model_path=None, use_gpu=True):
        super().__init__()
        self.score = PerceptualLoss(model=model, net=net, model_path=model_path,
                                    use_gpu=use_gpu, spatial=False).eval()
        self.reset()

#forward 方法计算预测图像和目标图像的 LPIPS 分数，并将结果添加到 individual_values 中。返回当前批次的 LPIPS 分数。
    def forward(self, pred_batch, target_batch, mask=None):
        batch_values = self.score(pred_batch, target_batch).flatten()
        self.individual_values = np.hstack([
            self.individual_values, batch_values.detach().cpu().numpy()
        ])
        return batch_values


def fid_calculate_activation_statistics(act):#计算给定激活值的均值 mu 和协方差矩阵 sigma。
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(activations_pred, activations_target, eps=1e-6):#计算两个激活值的均值和协方差矩阵。
    mu1, sigma1 = fid_calculate_activation_statistics(activations_pred)
    mu2, sigma2 = fid_calculate_activation_statistics(activations_target)

    diff = mu1 - mu2
#算两个协方差矩阵的乘积的平方根，处理可能的奇异情况。
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        LOGGER.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component处理可能的复数情况，确保结果为实数。
    # 计算 Frechet 距离，并返回结果。
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)



#FIDScore 类继承自 EvaluatorScore，用于计算 FID（Frechet Inception Distance）。初始化时，创建一个 InceptionV3 模型，并调用 reset 方法初始化状态。
class FIDScore(EvaluatorScore):
    def __init__(self, dims=2048, eps=1e-6):
        LOGGER.info("FIDscore init called")
        super().__init__()
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.reset()
        LOGGER.info("FIDscore init done")

#forward 方法计算预测图像和目标图像的激活值，并将结果添加到 activations_pred 和 activations_target 中。返回当前批次的激活值
    def forward(self, pred_batch, target_batch, mask=None):
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)

        self.activations_pred.append(activations_pred.detach().cpu())
        self.activations_target.append(activations_target.detach().cpu())

        return activations_pred, activations_target


#get_value 方法计算并返回 FID 分数，包括总分数和分组分数（如果有分组信息）。调用 calculate_frechet_distance 函数计算 Frechet 距离。重置状态。
    def get_value(self, groups=None, states=None):
        LOGGER.info("FIDscore get_value called")
        activations_pred, activations_target = zip(*states) if states is not None \
            else (self.activations_pred, self.activations_target)

        print("activations_pred length:", len(activations_pred))
        if not activations_pred:
            raise ValueError("No activations were generated. Check your data and model.")

        activations_pred = torch.cat(activations_pred).cpu().numpy()
        activations_target = torch.cat(activations_target).cpu().numpy()

        total_distance = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)
        total_results = dict(mean=total_distance)

        if groups is None:
            group_results = None
        else:
            group_results = dict()
            grouping = get_groupings(groups)
            for label, index in grouping.items():
                if len(index) > 1:
                    group_distance = calculate_frechet_distance(activations_pred[index], activations_target[index],
                                                                eps=self.eps)
                    group_results[label] = dict(mean=group_distance)

                else:
                    group_results[label] = dict(mean=float('nan'))

        self.reset()

        LOGGER.info("FIDscore get_value done")

        return total_results, group_results

    def reset(self):# ：重置 activations_pred 和 activations_target 属性为空列表。
        self.activations_pred = []
        self.activations_target = []

    def _get_activations(self, batch):#_get_activations 方法计算给定批次图像的激活值。确保激活值的维度正确。
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            assert False, \
                'We should not have got here, because Inception always scales inputs to 299x299'
            # activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.squeeze(-1).squeeze(-1)
        return activations


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

def iqa_metric(result_tensor, gt_tensor):
    mse = torch.mean((result_tensor - gt_tensor) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


class PSNRCalculator(nn.Module):
    def __init__(self, device="cuda", eps=1e-6):
        super().__init__()
        self.device = device
        self.eps = eps
        self.psnr_values = []
        self.reset()

    def reset(self):
        self.psnr_values = []

    def forward(self, pred_batch, target_batch, mask=None):
        batch_psnr = []
        for pred_img, target_img in zip(pred_batch, target_batch):
            if pred_img.shape != target_img.shape:
                raise ValueError("Input images must have the same dimensions.")

            pred_tensor = torch.tensor(pred_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            target_tensor = torch.tensor(target_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            psnr_value = iqa_metric(pred_tensor, target_tensor).item()
            self.psnr_values.append(psnr_value)
            batch_psnr.append(psnr_value)

        return batch_psnr

    def get_value(self, groups=None, states=None):
        if states is not None:
            psnr_values = [val for batch in states for val in batch]
        else:
            psnr_values = self.psnr_values

        if not psnr_values:
            raise ValueError("No PSNR values were computed. Check your data and model.")

        total_mean_psnr = np.mean(psnr_values)
        total_results = dict(mean=total_mean_psnr)

        if groups is None:
            group_results = None
        else:
            group_results = dict()
            grouping = get_groupings(groups)
            for label, index in grouping.items():
                if len(index) > 1:
                    group_psnr = np.mean([psnr_values[i] for i in index])
                    group_results[label] = dict(mean=group_psnr)
                else:
                    group_results[label] = dict(mean=float('nan'))

        self.reset()
        return total_results, group_results

    def compute_psnr(self, gt_path, results_path):
        gt_images = sorted(os.listdir(gt_path))
        result_images = sorted(os.listdir(results_path))

        pred_batch, target_batch = [], []
        for gt_image, result_image in zip(gt_images, result_images):
            gt_image_path = os.path.join(gt_path, gt_image)
            result_image_path = os.path.join(results_path, result_image)

            gt_img = cv2.imread(gt_image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            result_img = cv2.imread(result_image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

            pred_batch.append(result_img)
            target_batch.append(gt_img)

        return self.forward(pred_batch, target_batch)


class PSNR(nn.Module):
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val
        self.individual_values = []  # 用于存储每个批次的 PSNR 值

    def forward(self, img1, img2, mask=None):
        assert img1.shape == img2.shape, "输入图像形状必须一致"
        assert img1.dim() == 4, "输入图像必须是 4 维张量 (B, C, H, W)"

        if mask is not None:
            assert mask.shape[0] == img1.shape[0] and mask.shape[2:] == img1.shape[2:], "掩码形状不匹配"
            assert mask.shape[1] == 1, "掩码通道数必须为 1"
            mse = F.mse_loss(img1 * mask, img2 * mask, reduction='none')
            mse = mse.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)  # 避免除 0
        else:
            mse = F.mse_loss(img1, img2, reduction='none')
            mse = mse.mean(dim=(1, 2, 3))  # 计算每个样本的 MSE

        mse = torch.clamp(mse, min=1e-6)  # 避免 log10(0) 错误
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))  # 计算 PSNR

        print("MSE values:", mse)  # 调试输出
        print("PSNR values:", psnr)  # 调试输出

        self.individual_values.append(psnr.cpu().numpy())  # 存储 PSNR 值
        return psnr.mean()  # 返回平均 PSNR 值

    def reset(self):
        self.individual_values = []

    def get_value(self, groups=None):
        total_results = np.concatenate(self.individual_values, axis=0) if self.individual_values else np.array([])
        group_results = None  # 这里你需要自己定义 group_results 的逻辑

        return total_results, group_results  # 确保返回两个值


#该类继承自 EvaluatorScore，用于计算分割感知得分。初始化时，加载分割网络权重并设置模型为评估模式。初始化三个属性，用于存储目标和预测的类别频率。
class SegmentationAwareScore(EvaluatorScore):
    def __init__(self, weights_path):
        super().__init__()
        self.segm_network = SegmentationModule(weights_path=weights_path, use_default_normalization=True).eval()
        self.target_class_freq_by_image_total = []
        self.target_class_freq_by_image_mask = []
        self.pred_class_freq_by_image_mask = []

#forward 方法：预测分割结果并计算目标和预测的类别频率。将目标和预测的类别频率添加到相应的属性中。返回当前批次的类别频率
    def forward(self, pred_batch, target_batch, mask):
        pred_segm_flat = self.segm_network.predict(pred_batch)[0].view(pred_batch.shape[0], -1).long().detach().cpu().numpy()
        target_segm_flat = self.segm_network.predict(target_batch)[0].view(pred_batch.shape[0], -1).long().detach().cpu().numpy()
        mask_flat = (mask.view(mask.shape[0], -1) > 0.5).detach().cpu().numpy()

        batch_target_class_freq_total = []
        batch_target_class_freq_mask = []
        batch_pred_class_freq_mask = []

        for cur_pred_segm, cur_target_segm, cur_mask in zip(pred_segm_flat, target_segm_flat, mask_flat):
            cur_target_class_freq_total = np.bincount(cur_target_segm, minlength=NUM_CLASS)[None, ...]
            cur_target_class_freq_mask = np.bincount(cur_target_segm[cur_mask], minlength=NUM_CLASS)[None, ...]
            cur_pred_class_freq_mask = np.bincount(cur_pred_segm[cur_mask], minlength=NUM_CLASS)[None, ...]

            self.target_class_freq_by_image_total.append(cur_target_class_freq_total)
            self.target_class_freq_by_image_mask.append(cur_target_class_freq_mask)
            self.pred_class_freq_by_image_mask.append(cur_pred_class_freq_mask)

            batch_target_class_freq_total.append(cur_target_class_freq_total)
            batch_target_class_freq_mask.append(cur_target_class_freq_mask)
            batch_pred_class_freq_mask.append(cur_pred_class_freq_mask)

        batch_target_class_freq_total = np.concatenate(batch_target_class_freq_total, axis=0)
        batch_target_class_freq_mask = np.concatenate(batch_target_class_freq_mask, axis=0)
        batch_pred_class_freq_mask = np.concatenate(batch_pred_class_freq_mask, axis=0)
        return batch_target_class_freq_total, batch_target_class_freq_mask, batch_pred_class_freq_mask

    def reset(self):#重置类别频率属性为空列表。
        super().reset()
        self.target_class_freq_by_image_total = []
        self.target_class_freq_by_image_mask = []
        self.pred_class_freq_by_image_mask = []


def distribute_values_to_classes(target_class_freq_by_image_mask, values, idx2name):#计算各类的分布值。返回一个字典，其中包含各类的名称及其对应的分布值
    assert target_class_freq_by_image_mask.ndim == 2 and target_class_freq_by_image_mask.shape[0] == values.shape[0]
    total_class_freq = target_class_freq_by_image_mask.sum(0)
    distr_values = (target_class_freq_by_image_mask * values[..., None]).sum(0)
    result = distr_values / (total_class_freq + 1e-3)
    return {idx2name[i]: val for i, val in enumerate(result) if total_class_freq[i] > 0}


#返回一个字典，将类别索引映射到类别名称。
def get_segmentation_idx2name():
    return {i - 1: name for i, name in segm_options['classes'].set_index('Idx', drop=True)['Name'].to_dict().items()}


#类继承自 SegmentationAwareScore，用于计算分割感知的成对得分。初始化时，调用父类的初始化方法，并初始化 individual_values 和 segm_idx2name 属性。
class SegmentationAwarePairwiseScore(SegmentationAwareScore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.individual_values = []
        self.segm_idx2name = get_segmentation_idx2name()

#forward 方法：调用父类的 forward 方法，并计算分数值，将其添加到 individual_values 属性中。返回当前批次的类别频率和分数值。
    def forward(self, pred_batch, target_batch, mask):
        cur_class_stats = super().forward(pred_batch, target_batch, mask)
        score_values = self.calc_score(pred_batch, target_batch, mask)
        self.individual_values.append(score_values)
        return cur_class_stats + (score_values,)

    @abstractmethod
    def calc_score(self, pred_batch, target_batch, mask):#抽象方法，必须在子类中实现。
        raise NotImplementedError()

    def get_value(self, groups=None, states=None):
        """
        Get_value 方法：计算并返回总的和分组的分数结果。如果有状态传入，使用传入的状态计算结果，否则使用当前对象的属性。计算总结果和分组结果，并返回。
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            (target_class_freq_by_image_total,
             target_class_freq_by_image_mask,
             pred_class_freq_by_image_mask,
             individual_values) = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            individual_values = self.individual_values

        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        individual_values = np.concatenate(individual_values, axis=0)

        total_results = {
            'mean': individual_values.mean(),
            'std': individual_values.std(),
            **distribute_values_to_classes(target_class_freq_by_image_mask, individual_values, self.segm_idx2name)
        }

        if groups is None:
            return total_results, None

        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_class_freq = target_class_freq_by_image_mask[index]
            group_scores = individual_values[index]
            group_results[label] = {
                'mean': group_scores.mean(),
                'std': group_scores.std(),
                ** distribute_values_to_classes(group_class_freq, group_scores, self.segm_idx2name)
            }
        return total_results, group_results

    def reset(self):#调用父类的 reset 方法，并重置 individual_values 属性为空列表。
        super().reset()
        self.individual_values = []


#SegmentationClassStats 类继承自 SegmentationAwarePairwiseScore，其 calc_score 方法固定返回 0，表示该类仅用于统计分割的类别频率而不计算额外的分数。
class SegmentationClassStats(SegmentationAwarePairwiseScore):
    def calc_score(self, pred_batch, target_batch, mask):
        return 0

    def get_value(self, groups=None, states=None):
        """
    计算和返回总的和分组的分割类别频率统计。计算全局和分组的类别频率总和和差异，并将结果更新到 total_results 和 group_results 中。
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            (target_class_freq_by_image_total,
             target_class_freq_by_image_mask,
             pred_class_freq_by_image_mask,
             _) = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask

        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)

        target_class_freq_by_image_total_marginal = target_class_freq_by_image_total.sum(0).astype('float32')
        target_class_freq_by_image_total_marginal /= target_class_freq_by_image_total_marginal.sum()

        target_class_freq_by_image_mask_marginal = target_class_freq_by_image_mask.sum(0).astype('float32')
        target_class_freq_by_image_mask_marginal /= target_class_freq_by_image_mask_marginal.sum()

        pred_class_freq_diff = (pred_class_freq_by_image_mask - target_class_freq_by_image_mask).sum(0) / (target_class_freq_by_image_mask.sum(0) + 1e-3)

        total_results = dict()
        total_results.update({f'total_freq/{self.segm_idx2name[i]}': v
                              for i, v in enumerate(target_class_freq_by_image_total_marginal)
                              if v > 0})
        total_results.update({f'mask_freq/{self.segm_idx2name[i]}': v
                              for i, v in enumerate(target_class_freq_by_image_mask_marginal)
                              if v > 0})
        total_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v
                              for i, v in enumerate(pred_class_freq_diff)
                              if target_class_freq_by_image_total_marginal[i] > 0})

        if groups is None:
            return total_results, None

        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_target_class_freq_by_image_total = target_class_freq_by_image_total[index]
            group_target_class_freq_by_image_mask = target_class_freq_by_image_mask[index]
            group_pred_class_freq_by_image_mask = pred_class_freq_by_image_mask[index]

            group_target_class_freq_by_image_total_marginal = group_target_class_freq_by_image_total.sum(0).astype('float32')
            group_target_class_freq_by_image_total_marginal /= group_target_class_freq_by_image_total_marginal.sum()

            group_target_class_freq_by_image_mask_marginal = group_target_class_freq_by_image_mask.sum(0).astype('float32')
            group_target_class_freq_by_image_mask_marginal /= group_target_class_freq_by_image_mask_marginal.sum()

            group_pred_class_freq_diff = (group_pred_class_freq_by_image_mask - group_target_class_freq_by_image_mask).sum(0) / (
                    group_target_class_freq_by_image_mask.sum(0) + 1e-3)

            cur_group_results = dict()
            cur_group_results.update({f'total_freq/{self.segm_idx2name[i]}': v
                                      for i, v in enumerate(group_target_class_freq_by_image_total_marginal)
                                      if v > 0})
            cur_group_results.update({f'mask_freq/{self.segm_idx2name[i]}': v
                                      for i, v in enumerate(group_target_class_freq_by_image_mask_marginal)
                                      if v > 0})
            cur_group_results.update({f'mask_freq_diff/{self.segm_idx2name[i]}': v
                                      for i, v in enumerate(group_pred_class_freq_diff)
                                      if group_target_class_freq_by_image_total_marginal[i] > 0})

            group_results[label] = cur_group_results
        return total_results, group_results


class SegmentationAwareSSIM(SegmentationAwarePairwiseScore):#SegmentationAwareSSIM 类继承自 SegmentationAwarePairwiseScore，用于计算分割感知的 SSIM（结构相似性）得分。初始化时，设置 SSIM 计算器
    def __init__(self, *args, window_size=11, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_impl = SSIM(window_size=window_size, size_average=False).eval()

    def calc_score(self, pred_batch, target_batch, mask):#计算预测和目标批次的 SSIM 得分，并返回。
        return self.score_impl(pred_batch, target_batch).detach().cpu().numpy()


class SegmentationAwareLPIPS(SegmentationAwarePairwiseScore):#继承自 SegmentationAwarePairwiseScore，用于计算分割感知的 LPIPS（感知相似性）得分。初始化时，设置 LPIPS 计算器
    def __init__(self, *args, model='net-lin', net='vgg', model_path=None, use_gpu=True,  **kwargs):
        super().__init__(*args, **kwargs)
        self.score_impl = PerceptualLoss(model=model, net=net, model_path=model_path,
                                         use_gpu=use_gpu, spatial=False).eval()

    def calc_score(self, pred_batch, target_batch, mask):#计算预测和目标批次的 LPIPS 得分，并返回。
        return self.score_impl(pred_batch, target_batch).flatten().detach().cpu().numpy()


def calculade_fid_no_img(img_i, activations_pred, activations_target, eps=1e-6):
    activations_pred = activations_pred.copy()
    activations_pred[img_i] = activations_target[img_i]
    return calculate_frechet_distance(activations_pred, activations_target, eps=eps)


class SegmentationAwareFID(SegmentationAwarePairwiseScore):#继承自 SegmentationAwarePairwiseScore，用于计算分割感知的 FID（弗里奇特距离）得分。初始化时，设置 FID 计算器，加载 InceptionV3 模型
    def __init__(self, *args, dims=2048, eps=1e-6, n_jobs=-1, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(FIDScore, '_MODEL', None) is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            FIDScore._MODEL = InceptionV3([block_idx]).eval()
        self.model = FIDScore._MODEL
        self.eps = eps
        self.n_jobs = n_jobs

    def calc_score(self, pred_batch, target_batch, mask):#获取预测和目标批次的激活值，并返回。
        activations_pred = self._get_activations(pred_batch)
        activations_target = self._get_activations(target_batch)
        return activations_pred, activations_target

    def get_value(self, groups=None, states=None):
        """
        计算和返回总的和分组的 FID 得分。调用 distribute_fid_to_classes 方法将 FID 得分分配到各个类别。
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            (target_class_freq_by_image_total,
             target_class_freq_by_image_mask,
             pred_class_freq_by_image_mask,
             activation_pairs) = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            activation_pairs = self.individual_values

        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        activations_pred, activations_target = zip(*activation_pairs)
        activations_pred = np.concatenate(activations_pred, axis=0)
        activations_target = np.concatenate(activations_target, axis=0)

        total_results = {
            'mean': calculate_frechet_distance(activations_pred, activations_target, eps=self.eps),
            'std': 0,
            **self.distribute_fid_to_classes(target_class_freq_by_image_mask, activations_pred, activations_target)
        }

        if groups is None:
            return total_results, None

        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            if len(index) > 1:
                group_activations_pred = activations_pred[index]
                group_activations_target = activations_target[index]
                group_class_freq = target_class_freq_by_image_mask[index]
                group_results[label] = {
                    'mean': calculate_frechet_distance(group_activations_pred, group_activations_target, eps=self.eps),
                    'std': 0,
                    **self.distribute_fid_to_classes(group_class_freq,
                                                     group_activations_pred,
                                                     group_activations_target)
                }
            else:
                group_results[label] = dict(mean=float('nan'), std=0)
        return total_results, group_results

#计算每个类别的 FID 得分。通过并行计算每张图像去除后的 FID 得分，计算误差并分配到各个类别。
    def distribute_fid_to_classes(self, class_freq, activations_pred, activations_target):
        real_fid = calculate_frechet_distance(activations_pred, activations_target, eps=self.eps)

        fid_no_images = Parallel(n_jobs=self.n_jobs)(
            delayed(calculade_fid_no_img)(img_i, activations_pred, activations_target, eps=self.eps)
            for img_i in range(activations_pred.shape[0])
        )
        errors = real_fid - fid_no_images
        return distribute_values_to_classes(class_freq, errors, self.segm_idx2name)

    def _get_activations(self, batch):#通过 InceptionV3 模型获取批次图像的激活值，并返回。
        activations = self.model(batch)[0]
        if activations.shape[2] != 1 or activations.shape[3] != 1:
            activations = F.adaptive_avg_pool2d(activations, output_size=(1, 1))
        activations = activations.squeeze(-1).squeeze(-1).detach().cpu().numpy()
        return activations
