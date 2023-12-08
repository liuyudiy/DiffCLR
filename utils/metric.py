import torch
import numpy as np
import logging

from .basic import deter_scatter_add_


class PredictionMetrics:
    def __init__(self):
        self._ranks = []
        self._weights = []

    def digest(self, pred: torch.Tensor, truth: torch.Tensor, weight: torch.Tensor = None):
        num_nodes = pred.shape[0]
        assert truth.shape[0] == num_nodes
        truth_prob = pred.gather(dim=1, index=truth.unsqueeze(1))
        rank = pred.gt(truth_prob).sum(dim=1) + 1
        self._ranks.append(rank)
        if weight is None:
            self._weights.append(torch.ones_like(rank))
        else:
            self._weights.append(weight)

    def get_ranks(self):
        if len(self._ranks) > 1:
            self._ranks = [torch.cat(self._ranks)]
        return self._ranks[0]

    def get_weight(self):
        w = torch.cat(self._weights)
        self._weights = [w]
        return w

    def _weighted_mean(self, arr):
        w = self.get_weight()
        return ((arr.float() * w).sum() / w.sum()).item()

    def MRR(self):
        return self._weighted_mean(self.get_ranks().float().reciprocal())

    def hits_at(self, k):
        return self._weighted_mean(self.get_ranks().le(k))


def loss_cross_entropy_multi_ans(score, query, ans, posi_x, posi_ans, query_w=None):
    assert len(posi_x) >= len(query)
    device = score.device
    num_nodes = len(score)
    score = score.exp()
    ent_posi_sum = torch.zeros(num_nodes, dtype=torch.double, device=device)

    deter_scatter_add_(posi_x, score[posi_x, posi_ans], ent_posi_sum)
    ans_score = score[query, ans]
    ans_score[ans_score < 0] = 1e-10

    total_score = score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score
    if any(total_score < 0):
        logging.info("Note that total score less than zero occurs")
        ent_posi_sum = torch.zeros(num_nodes, dtype=torch.double, device=device)
        deter_scatter_add_(query, score[query, ans], ent_posi_sum)
        total_score = score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score
    assert all(total_score > 0)

    loss_arr = ans_score / total_score
    loss_arr = -loss_arr.log()

    if query_w is None:
        query_w = torch.ones_like(loss_arr)
    # loss = torch.sum(loss_arr * query_w)
    # weight_sum = query_w.sum()
    # return loss.float(), weight_sum
    from torch_scatter import segment_csr
    loss_arr = loss_arr * query_w
    _, counts = torch.unique_consecutive(query, return_counts=True)
    counts = torch.cat((torch.tensor([0], device=device), counts))
    counts = torch.cumsum(counts, dim=0)
    loss = segment_csr(loss_arr, counts, reduce="sum")
    weight_sum = segment_csr(query_w, counts, reduce="sum")
    
    return loss.float(), weight_sum


def loss_label_smoothing_multi_ans(score, query, ans, posi_x, posi_ans, smoothing, query_w=None):
    assert len(posi_x) >= len(query)
    device = score.device
    num_nodes = len(score)
    score = score.exp()
    ent_posi_sum = torch.zeros(num_nodes, dtype=torch.double, device=device)
    deter_scatter_add_(posi_x, score[posi_x, posi_ans], ent_posi_sum)
    ans_score = score[query, ans]
    ans_score[ans_score < 0] = 1e-10

    total_score = score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score
    if any(total_score < 0):
        logging.info("Note that total score less than zero occurs")
        ent_posi_sum = torch.zeros(num_nodes, dtype=torch.double, device=device) # (BE,)
        deter_scatter_add_(query, score[query, ans], ent_posi_sum)
        total_score = score.sum(dim=1)[query] - ent_posi_sum[query] + ans_score
    if any(total_score <= 0):
        mask = total_score <= 0
        print('less than zeros', total_score[mask])
    assert all(total_score > 0)

    loss_arr = ans_score / total_score
    whole_loss = score[query] / total_score.unsqueeze(1)    

    whole_loss = -whole_loss.log()
    loss_arr = -loss_arr.log()
    if query_w is None:
        query_w = torch.ones_like(loss_arr)
    # loss = torch.sum(loss_arr * query_w)
    # rand_loss = torch.sum(whole_loss.mean(-1) * query_w)
    # weight_sum = query_w.sum()
    # LSloss = smoothing * rand_loss + (1 - smoothing) * loss

    from torch_scatter import segment_csr
    loss_arr = loss_arr * query_w
    _, counts = torch.unique_consecutive(query, return_counts=True)
    counts = torch.cat((torch.tensor([0], device=device), counts))
    counts = torch.cumsum(counts, dim=0)
    loss = segment_csr(loss_arr, counts, reduce="sum")
    rand_loss = whole_loss.mean(-1) * query_w
    rand_loss = segment_csr(whole_loss.mean(-1), counts, reduce="sum")
    weight_sum = segment_csr(query_w, counts, reduce="sum")
    LSloss = smoothing * rand_loss + (1 - smoothing) * loss
    return LSloss, weight_sum