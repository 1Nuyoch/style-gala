#!/usr/bin/env python3

import os

import pandas as pd

from metrics.evaluation.data import PrecomputedInpaintingResultsDataset
from metrics.evaluation.evaluator import InpaintingEvaluator, lpips_fid100_f1
from metrics.evaluation.losses.base_loss import SegmentationAwareSSIM, \
SegmentationClassStats, SSIMScore, LPIPSScore, FIDScore, SegmentationAwareLPIPS, SegmentationAwareFID
from metrics.evaluation.utils import load_yaml


def main(args):
    config = load_yaml(args.config) #加载配置文件，通过load_yaml函数加载配置文件，args.config是配置文件的路径，包含各种评估参数。
#创建一个PrecomputedInpaintingResultsDataset对象，用于加载预计算的修复结果。args.datadir是包含图像和掩码的数据目录，args.predictdir是包含预测结果的目录，config.dataset_kwargs是其他数据集相关的参数。
    dataset = PrecomputedInpaintingResultsDataset(args.datadir, args.predictdir, **config.dataset_kwargs)
#定义基本的评估指标，包括SSIM、LPIPS和FID
    metrics = {
        'ssim': SSIMScore(),
        'lpips': LPIPSScore(),
        'fid': FIDScore()
    }
    #如果配置文件中启用了分割评估，则添加分割相关的评估指标，并加载分割模型的权重
    enable_segm = config.get('segmentation', dict(enable=False)).get('enable', False)
    if enable_segm:
        weights_path = os.path.expandvars(config.segmentation.weights_path)
        metrics.update(dict(
            segm_stats=SegmentationClassStats(weights_path=weights_path),
            segm_ssim=SegmentationAwareSSIM(weights_path=weights_path),
            segm_lpips=SegmentationAwareLPIPS(weights_path=weights_path),
            segm_fid=SegmentationAwareFID(weights_path=weights_path)
        ))
    #创建一个InpaintingEvaluator对象，传入数据集和评估指标。integral_title和integral_func用于设置综合评估函数，config.evaluator_kwargs是其他评估器相关的参数
    evaluator = InpaintingEvaluator(dataset, scores=metrics,
                                    integral_title='lpips_fid100_f1', integral_func=lpips_fid100_f1,
                                    **config.evaluator_kwargs)

    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)#创建输出文件的目录，如果目录不存在则创建。

    results = evaluator.evaluate()#调用评估器的evaluate方法进行评估，得到评估结果。

    #将结果转换为DataFrame格式，去除所有值为NaN的列，并保存为制表符分隔的文本文件。
    results = pd.DataFrame(results).stack(1).unstack(0)
    results.dropna(axis=1, how='all', inplace=True)
    results.to_csv(args.outpath, sep='\t', float_format='%.4f')

    #如果启用了分割评估，则保存简短的评估结果（不包括分割指标），并打印这些结果。分割评估的结果被处理并保存为另一个文件。
    if enable_segm:
        only_short_results = results[[c for c in results.columns if not c[0].startswith('segm_')]].dropna(axis=1, how='all')
        only_short_results.to_csv(args.outpath + '_short', sep='\t', float_format='%.4f')

        print(only_short_results)

        segm_metrics_results = results[['segm_ssim', 'segm_lpips', 'segm_fid']].dropna(axis=1, how='all').transpose().unstack(0).reorder_levels([1, 0], axis=1)
        segm_metrics_results.drop(['mean', 'std'], axis=0, inplace=True)

        segm_stats_results = results['segm_stats'].dropna(axis=1, how='all').transpose()
        segm_stats_results.index = pd.MultiIndex.from_tuples(n.split('/') for n in segm_stats_results.index)
        segm_stats_results = segm_stats_results.unstack(0).reorder_levels([1, 0], axis=1)
        segm_stats_results.sort_index(axis=1, inplace=True)
        segm_stats_results.dropna(axis=0, how='all', inplace=True)

        segm_results = pd.concat([segm_metrics_results, segm_stats_results], axis=1, sort=True)
        segm_results.sort_values(('mask_freq', 'total'), ascending=False, inplace=True)

        segm_results.to_csv(args.outpath + '_segm', sep='\t', float_format='%.4f')
    else:
        print(results)

#定义命令行接口，解析命令行参数并调用主函数main。参数包括配置文件路径、数据目录、预测目录和输出路径。
if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to evaluation config')
    aparser.add_argument('datadir', type=str,
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('predictdir', type=str,
                         help='Path to folder with predicts (e.g. predict_hifill_baseline.py)')
    aparser.add_argument('outpath', type=str, help='Where to put results')

    main(aparser.parse_args())
