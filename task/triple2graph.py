import torch
from torch.utils.data import Sampler, DataLoader
from functools import partial
from typing import Iterable

from .betae import BetaEDataset
from utils.graph import Graph, MatGraph, BatchMatGraph
from utils.graph import IndexedGraph, get_directed_lap_matrix_np
from utils.graph import GraphWithAnswer, EdgeIndexer
from utils.sampler import mini_sampler



def _batch_mini_sampler(batch, igraph, relation_cnt, lap_matrix, args):
    stream = iter(batch)

    stream = map(
        lambda x: mini_sampler(igraph, x, lap_matrix, relation_cnt, args),
        stream)
    stream = filter(lambda x: x is not None, stream)
    stream = map(lambda x: MatGraph.make_line_graph(x, relation_cnt), stream)
    g = BatchMatGraph.from_mat_list(list(stream))
    return g


def get_dataloader_train(full_train_graph, num_nodes, relation_cnt, args, shuffle):
    igraph = IndexedGraph.from_graph(full_train_graph)
    if args.pretrain_dataset_source == 'relation':
        dataset = igraph.edge_index.T
    elif args.pretrain_dataset_source == 'entity':
        dataset = torch.arange(num_nodes).unsqueeze(-1)
    else:
        assert False        
    # noinspection PyTypeChecker
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=partial(
            _batch_mini_sampler,
            igraph=igraph,
            relation_cnt=relation_cnt,
            lap_matrix=get_directed_lap_matrix_np(igraph.edge_index, igraph.num_nodes),
            args=args,
        ),
        shuffle=shuffle,
    )
    return dataloader


def get_dataloader_test(edge_list, exclude_edges, num_nodes, relation_cnt, args, shuffle):
    existing_ans = EdgeIndexer(num_nodes, relation_cnt)
    if exclude_edges is not None:
        for a, p, b in exclude_edges:
            existing_ans.add_edge(a, p, b)

    def collate(batch):
        num_edges = len(batch)
        batch = torch.stack(batch, dim=1)
        g_list = []
        ei_single = torch.tensor([[0], [1]])
        one_arr = torch.tensor([1])
        for i in range(num_edges):
            a, p, b = torch.flatten(batch[:, i]).tolist()
            x_node = torch.tensor([a, -1])
            edge_attr=torch.tensor([p])
            g = GraphWithAnswer(
                x=x_node,
                x_d=torch.zeros_like(x_node, dtype=torch.long),
                r_d=torch.zeros_like(edge_attr, dtype=torch.long),
                edge_index=ei_single,
                edge_attr=edge_attr,
                x_query=one_arr,
                x_ans=torch.tensor([b]),
            )
            if exclude_edges is not None:
                mask_arr = existing_ans.get_targets(a, p)
                if b not in mask_arr:
                    mask_arr.append(b)
                g.x_pred_mask = torch.tensor([
                    [1] * len(mask_arr),
                    mask_arr,
                ])
            g_list.append(MatGraph.make_line_graph(g, relation_cnt))
        # TODO: another half
        g = BatchMatGraph.from_mat_list(g_list)
        return g

    dataloader = DataLoader(
        torch.tensor(edge_list),
        batch_size=4096,
        collate_fn=collate,
        num_workers=10,
        shuffle=shuffle
    )
    return dataloader


class Triple2Graph:
    def __init__(self, args):
        self.args = args
        self.betae = BetaEDataset(args.data_dir)
        
        self.id2ent = self.betae.get_file("id2ent.pkl")
        self.id2rel = self.betae.get_file("id2rel.pkl")
        self.num_nodes = len(self.id2ent)
        self.relation_cnt = len(self.id2rel) // 2

        self.train_edge = self.betae.get_file("train.txt")
        self.valid_edge = self.betae.get_file("valid.txt")
        self.test_edge = self.betae.get_file("test.txt")

    def get_full_train_graph(self):
        arr = torch.tensor(self.train_edge).T  # (3, 272115)
        return Graph(
            x=torch.arange(self.num_nodes),
            edge_index=arr[[0, 2]],
            edge_attr=arr[1],
        )

    def dataloader_train(self):
        return get_dataloader_train(
            self.get_full_train_graph(),
            self.num_nodes,
            self.relation_cnt,
            self.args,
            shuffle=False
        )

    def dataloader_test(self):
        return get_dataloader_test(
            self.test_edge,
            self.train_edge + self.valid_edge + self.test_edge,
            self.num_nodes,
            self.relation_cnt,
            self.args,
            shuffle=False
        )
