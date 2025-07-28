import bisect
import warnings

from torch._utils import _accumulate
from torch import randperm


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (iterable): List of datasets to be concatenated
    """

    @staticmethod  # 用于计算每个子数据集在整个串联数据集中的累积样本数量。
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    # 初始化方法接收一个可迭代的数据集列表 datasets，并进行断言确保 datasets 长度大于0。然后，将输入的数据集列表转换为列表形式，并计算每个数据集累积的大小。
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):  # 返回串联数据集中所有样本的总数，即最后一个数据集的累积大小
        return self.cumulative_sizes[-1]

    # 根据给定的索引 idx，确定应该从哪个子数据集中获取样本。使用 bisect.bisect_right 函数找到对应的子数据集索引，然后计算在该子数据集中的样本索引并返回。
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class Subset(Dataset):
    def __init__(self, dataset, indices):#初始化方法接收一个数据集对象 dataset 和一个索引列表 indices
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):#根据给定的索引 idx，从原始数据集 dataset 中获取对应 indices[idx] 的样本。
        return self.dataset[self.indices[idx]]

    def __len__(self):# 返回子集的长度，即索引列表 indices 的长度
        return len(self.indices)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths
    ds

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (iterable): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
