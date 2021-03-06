import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa PyPep8Naming

from torch.utils.data import Dataset, DataLoader, Sampler

CPU_DEVICE = torch.device("cpu")


def normalize_embedding_tensor(embedding):
    return F.normalize(embedding, p=2, dim=1)


def embedding_norm_loss(raw_embedding):
    norms = raw_embedding.norm(dim=1)
    return F.mse_loss(norms, torch.ones_like(norms), reduction="none")


def get_val(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


def compute_pred_entropy(logits):
    # logits are pre softmax
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    return -(p * log_p).sum(dim=-1).mean()


def compute_pred_entropy_clean(logits):
    return float(compute_pred_entropy(logits).item())


def copy_state_dict(state_dict, target_device=None):
    copied_state_dict = copy.deepcopy(state_dict)
    if target_device is None:
        return copied_state_dict
    else:
        return {k: v.to(target_device) for k, v in copied_state_dict.items()}


def get_parent_child_module_list(model):
    ls = []
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            ls.append((parent_name, parent_module, child_name, child_module))
    return ls


class IdentityModule(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        else:
            return inputs


def set_requires_grad(named_parameters, requires_grad):
    for name, param in named_parameters:
        param.requires_grad = requires_grad


def get_only_requires_grad(parameters, requires_grad=True):
    if isinstance(parameters, list):
        if not parameters:
            return []
        elif isinstance(parameters[0], tuple):
            return [(n, p) for n, p in parameters if p.requires_grad == requires_grad]
        else:
            return [p for p in parameters if p.requires_grad == requires_grad]
    elif isinstance(parameters, dict):
        return {n: p for n, p in parameters if p.requires_grad == requires_grad}
    else:
        # TODO: Support generators  (Issue #56)
        raise RuntimeError("generators not yet supported")


class ListDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class DataLoaderWithLength(DataLoader):
    def __len__(self):
        # TODO: Revert after https://github.com/pytorch/pytorch/issues/36176 addressed  (Issue #55)
        # try:
        #     return super().__len__()
        # except TypeError as e:
        #     try:
        #         return self.get_num_batches()
        #     except TypeError:
        #         pass
        #     raise e
        return self.get_num_batches()

    def get_num_batches(self):
        return math.ceil(len(self.dataset) / self.batch_size)

class MatchedRandomBatchSampler(Sampler):
    """Random batch sampler that wraps individual sampler. Samples with clustered data with replacement to generate
       batches of clusters.

        Returns potentially different sized batches at least min_batch_size and at most max_batch_size.
        Clusters are precomputed and passed through a list of lists.

        Attributes:
            min_batch_size : Min batch size of individual example indices
            min_batch_size : Max batch size of individual example indices
            drop_last      : Boolean, whether to drop last batch unfinished batch
            match_list     : List of clusters. Clusters represented as list of individual example indices.
            n_clusters     : Total number of clusters.
            total_batches  : Total batches required for run.
    """
    def __init__(self,
                 min_batch_size,
                 max_batch_size,
                 drop_last,
                 match_list,
                 total_batches):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last
        self.match_list = match_list
        self.n_clusters = len(match_list)
        self.total_batches = total_batches

    def __iter__(self):
        batch = []
        yielded = 0
        while yielded < self.total_batches:
            sampled_cluster = torch.randint(len(self.match_list), (1,)).item()
            if len(batch) + len(self.match_list[sampled_cluster]) > self.max_batch_size:
                if len(batch) < self.min_batch_size:
                    continue
                else:
                    yielded += 1
                    yield batch
                    batch = self.match_list[sampled_cluster]
            else:
                batch += self.match_list[sampled_cluster]
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return self.total_batches

def is_data_parallel(torch_module):
    return isinstance(torch_module, nn.DataParallel)


def safe_save(obj, path, temp_path=None):
    if temp_path is None:
        temp_path = path + "._temp"
    torch.save(obj, temp_path)
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)


def get_model_for_saving(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    else:
        return model
