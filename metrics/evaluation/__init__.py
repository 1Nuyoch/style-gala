import logging

import torch

from metrics.evaluation.evaluator import InpaintingEvaluatorOnline, ssim_fid100_f1, lpips_fid100_f1
from metrics.evaluation.losses.base_loss import SSIMScore, LPIPSScore, FIDScore

'''
根据参数创建并返回一个图像评估器对象。
kind: 指定评估器的类型，默认值为 'default'。
ssim: 是否计算 SSIM（结构相似性指数），默认为 True。
lpips: 是否计算 LPIPS（感知相似性），默认为 True。
fid: 是否计算 FID（生成对抗网络的判别性能），默认为 True。
integral_kind: 指定整合函数的类型，可选值为 'ssim_fid100_f1' 或 'lpips_fid100_f1'，默认为 None。
kwargs: 其他传递给 InpaintingEvaluatorOnline 的参数。
逻辑流程
'''
def make_evaluator(kind='default', ssim=True, lpips=True, fid=True, integral_kind=None, **kwargs):
    logging.info(f'Make evaluator {kind}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {}#根据参数 ssim、lpips 和 fid，选择性地将对应的评分类实例添加到 metrics 字典
    if ssim:
        metrics['ssim'] = SSIMScore()
    if lpips:
        metrics['lpips'] = LPIPSScore()
    if fid:
        metrics['fid'] = FIDScore().to(device)
        
    if integral_kind is None:#根据 integral_kind 参数的值，选择对应的整合函数，或设置为 None。
        integral_func = None
    elif integral_kind == 'ssim_fid100_f1':#ssim_fid100_f1 和 lpips_fid100_f1: 用于整合不同评分的函数。

        integral_func = ssim_fid100_f1
    elif integral_kind == 'lpips_fid100_f1':
        integral_func = lpips_fid100_f1
    else:
        raise ValueError(f'Unexpected integral_kind={integral_kind}')

    if kind == 'default':#如果 kind 为 'default'，则创建并返回 InpaintingEvaluatorOnline 对象，传入 metrics、integral_func、integral_title 和其他参数。
        return InpaintingEvaluatorOnline(scores=metrics,#InpaintingEvaluatorOnline 类.该类用于在线评估图像修补任务中的结果，使用指定的评分指标和整合函数。
                                         integral_func=integral_func,
                                         integral_title=integral_kind,
                                         **kwargs)
