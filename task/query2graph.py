import torch
from torch.utils.data import Sampler, DataLoader
from functools import partial

from .betae import BetaEDataset
from utils.graph import GraphWithAnswer, MatGraph, BatchMatGraph


def query_to_graph(mode, q, hop, q_neis, hard_ans_list, mask_list=None, gen_pred_mask=True):
    if not gen_pred_mask:
        assert mask_list is None or len(hard_ans_list) == len(mask_list)
    x= []
    edge_ans = []

    edge_index = [[], []]
    edge_attr = [] 

    x_d, r_d = [], []
    def add_raw_edge(a, rel, b):
        if rel % 2 == 1:
            a, b = b, a
            rel = (rel - 1) / 2
        else:
            rel = rel / 2
        edge_index[0].append(a)
        edge_index[1].append(b)
        edge_attr.append(rel)
        r_d.append(0)
    
    def sanitize_edge(r):
        if r % 2 == 1:
            r = int((r-1) / 2)
        else:
            r = int(r / 2)
        return r

    def add_neighbors(entity_list, relation_list):
        neis_dict, rels_dict, tris_rel_dict = q_neis
        neis = set(list(neis_dict.keys())) - set(entity_list)
        if neis:
            x.extend(list(neis))
            for nei in neis:
                nei_hop = neis_dict[nei]-1
                degree = max(min(nei_hop, hop-1), 0)
                x_d.append(degree)
            for rel in tris_rel_dict.keys():
                if rel not in relation_list:
                    h, t = tris_rel_dict[rel]
                    h_index, t_index = x.index(h), x.index(t)
                    edge_index[0].append(h_index)
                    edge_index[1].append(t_index)
                    edge_attr.append(rel)
                    rel_hop = rels_dict[rel]-1
                    degree = max(min(rel_hop, hop-1), 0)
                    r_d.append(degree)

    q_cnt = 0
    x_query = []
    x_ans = []
    x_pred_weight = []
    x_pred_mask_x = []
    x_pred_mask_y = []
    joint_nodes = []
    union_query = []

    def push_anslist(node_id, hard_anslist, mask_list):
        nonlocal x_query, x_ans, x_pred_weight
        anslen = len(hard_anslist)
        if anslen == 0:
            return
        x_query += [node_id] * anslen
        x_ans += hard_anslist
        x_pred_weight += [1 / anslen] * anslen
        if not gen_pred_mask:
            return
        nonlocal x_pred_mask_x, x_pred_mask_y, q_cnt
        # assert that mask_list already contains hard_anslist
        x_pred_mask_x += [node_id] * len(mask_list)
        x_pred_mask_y += mask_list

    def push_anslist_and_masklist(node_id, mask_id, hard_anslist, mask_list):
        nonlocal x_query, x_ans, x_pred_weight
        anslen = len(hard_anslist)
        if anslen == 0:
            return
        x_query += [node_id] * anslen
        x_ans += hard_anslist
        x_pred_weight += [1 / anslen] * anslen
        if not gen_pred_mask:
            return
        nonlocal x_pred_mask_x, x_pred_mask_y, q_cnt
        for i in mask_id:
            x_pred_mask_x += [i] * len(mask_list)
            x_pred_mask_y += mask_list

    mask_list = mask_list or []
    if mode == '1p':
        x = [q[0], -1] 
        x_d = [0] * len(x)
        add_raw_edge(0, q[1][0], 1)
        push_anslist(1, hard_ans_list, mask_list)
        if q_neis:
            entity_list = [q[0]]
            relation_list = list(map(sanitize_edge, [q[1][0]]))  
            add_neighbors(entity_list, relation_list)

    elif mode == '2p':
        r"""
        0 - 1(-1) - 2(-1)
        """
        x = [q[0], -1, -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[1][0], 1)
        add_raw_edge(1, q[1][1], 2)
        push_anslist(2, hard_ans_list, mask_list)
        if q_neis:
            entity_list = [q[0]]
            relation_list = list(map(sanitize_edge, [q[1][0], q[1][1]]))  
            add_neighbors(entity_list, relation_list)

    elif mode == '3p':
        r"""
        0 - 1(-1) - 2(-1) - 3(-1)
        """
        x = [q[0], -1, -1, -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[1][0], 1)
        add_raw_edge(1, q[1][1], 2)
        add_raw_edge(2, q[1][2], 3)
        push_anslist(3, hard_ans_list, mask_list)
        if q_neis:
            entity_list = [q[0]]
            relation_list = list(map(sanitize_edge, [q[1][0], q[1][1], q[1][2]]))  
            add_neighbors(entity_list, relation_list)

    elif mode == '2i':
        r"""
        0 - 
            2(-1)
        1 - 
        """
        x = [q[0][0], q[1][0], -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[0][1][0], 2)
        add_raw_edge(1, q[1][1][0], 2)
        push_anslist(2, hard_ans_list, mask_list)
        if q_neis:
            entity_list = [q[0][0], q[1][0]]
            relation_list = list(map(sanitize_edge, [q[0][1][0], q[1][1][0]]))  
            add_neighbors(entity_list, relation_list)

    elif mode == '3i':
        r"""
        0 - 
        1 -  3(-1)
        2 - 
        """
        x = [q[0][0], q[1][0], q[2][0], -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[0][1][0], 3)
        add_raw_edge(1, q[1][1][0], 3)
        add_raw_edge(2, q[2][1][0], 3)
        push_anslist(3, hard_ans_list, mask_list)
        if q_neis:
            entity_list =[q[0][0], q[1][0], q[2][0]]
            relation_list = list(map(sanitize_edge, [q[0][1][0], q[1][1][0], q[2][1][0]]))  
            add_neighbors(entity_list, relation_list)

    elif mode == 'pi':
        r"""
        0 - 1(-1) -
                    3(-1)
                2 -  
        """
        x = [q[0][0], -1, q[1][0], -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[0][1][0], 1)
        add_raw_edge(1, q[0][1][1], 3)
        add_raw_edge(2, q[1][1][0], 3)
        push_anslist(3, hard_ans_list, mask_list)
        if q_neis:
            entity_list = [q[0][0], q[1][0]]
            relation_list = list(map(sanitize_edge, [q[0][1][0], q[0][1][1], q[1][1][0]]))  
            add_neighbors(entity_list, relation_list)

    elif mode == 'ip':
        r"""
        0 - 
            2(-1) - 3(-1)
        1 - 
        """
        x = [q[0][0][0], q[0][1][0], -1, -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[0][0][1][0], 2)
        add_raw_edge(1, q[0][1][1][0], 2)
        add_raw_edge(2, q[1][0], 3)
        push_anslist(3, hard_ans_list, mask_list)
        if q_neis:
            entity_list= [q[0][0][0], q[0][1][0]]
            relation_list = list(map(sanitize_edge, [q[0][0][1][0], q[0][1][1][0], q[1][0]])) 
            add_neighbors(entity_list, relation_list)

    elif mode == '2u':
        r"""
                0 - 2(-1)
                |
                4(-1)
                |
                1 - 3(-1)
        """
        x = [q[0][0], q[1][0], -1, -1, -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[0][1][0], 2)
        add_raw_edge(0, q[0][1][0], 4)
        add_raw_edge(1, q[1][1][0], 3)
        add_raw_edge(1, q[1][1][0], 4)
        push_anslist_and_masklist(4, [2, 3], hard_ans_list, mask_list)
        if q_neis:
            entity_list = [q[0][0], q[1][0]]
            relation_list = list(map(sanitize_edge, [q[0][1][0], q[1][1][0]])) 
            add_neighbors(entity_list, relation_list)
        anslen = len(hard_ans_list)
        joint_nodes = [2, 3] * anslen
        union_query = [4] * anslen
    
    elif mode == 'up':
        r"""
                0 - 2(-1) - 3(-1)
                |
                6(-1) - 7(-1)
                |
                1 - 4(-1) - 5(-1)
        """
        x = [q[0][0][0], q[0][1][0], -1, -1, -1, -1, -1, -1]
        x_d = [0] * len(x)
        add_raw_edge(0, q[0][0][1][0], 2)
        add_raw_edge(0, q[0][0][1][0], 6)
        add_raw_edge(1, q[0][1][1][0], 4)
        add_raw_edge(1, q[0][1][1][0], 6)
        add_raw_edge(2, q[1][0], 3)
        add_raw_edge(6, q[1][0], 7)
        add_raw_edge(4, q[1][0], 5)
        push_anslist_and_masklist(7, [3, 5], hard_ans_list, mask_list)
        if q_neis:
            entity_list = [q[0][0][0], q[0][1][0]]
            relation_list = list(map(sanitize_edge, [q[0][0][1][0], q[0][1][1][0], q[1][0]])) 
            add_neighbors(entity_list, relation_list)
        anslen = len(hard_ans_list)
        joint_nodes = [3, 5] * anslen
        union_query = [7] * anslen
    else:
        assert False

    g = GraphWithAnswer(
        x=torch.tensor(x, dtype=torch.long),
        x_d=torch.tensor(x_d, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.long),
        r_d=torch.tensor(r_d, dtype=torch.long),
        x_query=torch.tensor(x_query, dtype=torch.long),
        x_ans=torch.tensor(x_ans, dtype=torch.long),
        edge_ans=torch.tensor(edge_ans, dtype=torch.long),
        x_pred_weight=torch.tensor(x_pred_weight, dtype=torch.float),
        joint_nodes=torch.tensor(joint_nodes, dtype=torch.long),
        union_query=torch.tensor(union_query, dtype=torch.long)
    )
    if gen_pred_mask:
        g.x_pred_mask = torch.tensor([x_pred_mask_x, x_pred_mask_y], dtype=torch.long)
    return g


class SubsetSumSampler(Sampler):
    def __init__(self, values, lim=512, shuffle=True):
        super(SubsetSumSampler, self).__init__(values)
        n = len(values)
        self._order = list(range(n))
        if shuffle:
            import random
            random.shuffle(self._order)
        self._index_list = []
        # cnt = 0
        # for i in range(n):
        #     cur = values[self._order[i]]
        #     if cnt + cur > lim and cnt > 0:
        #         cnt = cur
        #         self._index_list.append(i)
        #     else:
        #         cnt += cur
        for i in range(0, n, lim):
            self._index_list.append(i)
        self._index_list.append(n)

    def __iter__(self):
        l = 0
        for r in self._index_list[1:]:
            yield self._order[l:r]
            l = r

    def __getitem__(self, item):
        idx = self._index_list
        l = idx[item - 1] if item != 0 else 0
        r = idx[item]
        return self._order[l:r]

    def __len__(self):
        return len(self._index_list)


class Query2Graph:
    def __init__(self, args):
        self.args = args
        self.betae = BetaEDataset(args.data_dir)
        self.train_mode = args.train_modes
        self.test_mode = args.test_modes

        self.id2ent = self.betae.get_file("id2ent.pkl")
        self.id2rel = self.betae.get_file("id2rel.pkl")
        self.num_nodes = len(self.id2ent)
        self.relation_cnt = len(self.id2rel) // 2

        self.train_query = self.betae.get_file("train-queries.pkl")
        # self.valid_query = self.betae.get_file("valid-queries.pkl")
        self.test_query = self.betae.get_file("test-queries.pkl")

        self.train_answer = self.betae.get_file("train-answers.pkl")
        # self.valid_hard_answer = self.betae.get_file("valid-hard-answers.pkl")
        # self.valid_easy_answer = self.betae.get_file("valid-easy-answers.pkl")
        # self.valid_answer = self.betae.get_file("valid-answers.pkl")
        self.test_hard_answer = self.betae.get_file("test-hard-answers.pkl")
        self.test_easy_answer = self.betae.get_file("test-easy-answers.pkl")
        # self.test_answer = self.betae.get_file("test-answers.pkl")
        if args.use_neis:     
            self.train_query_neis = self.betae.get_file("train-queries-neis.pkl")
            self.valid_query_neis = self.betae.get_file("valid-queries-neis.pkl")
            self.test_query_neis = self.betae.get_file("test-queries-neis.pkl")
        else:
            self.train_query_neis = None
            self.valid_query_neis = None
            self.test_query_neis = None

    @staticmethod
    def _batch_q2g(batch, relation_cnt):
        arr = [query_to_graph(*x) for x in batch]
        arr = [MatGraph.make_line_graph(g, relation_cnt) for g in arr]
        return BatchMatGraph.from_mat_list(arr)

    def _get_dataloader(self, query, query_neis, pred_answer, mask_answer, modelist, gen_pred_mask,
                        batch_size, num_workers, shuffle):
        data = []
        for m in modelist:
            for q in query[m]:
                mask_set = pred_answer[q]
                if mask_answer is not None:
                    mask_set = mask_set | mask_answer[q]
                if query_neis:
                    q_neis = query_neis[m][q]
                else:
                    q_neis = None
                hop = self.args.hop
                data.append((m, q, hop, q_neis, list(pred_answer[q]), list(mask_set), gen_pred_mask))

        data = DataLoader(
            data,
            batch_sampler=SubsetSumSampler(
                [len(x[4]) for x in data],
                lim=batch_size,
                shuffle=shuffle
            ),
            collate_fn=partial(
                Query2Graph._batch_q2g,
                relation_cnt=self.relation_cnt
            ),
            num_workers=num_workers
        )

        return data

    def dataloader_train(self):
        mode = self.train_mode
        ans = self.train_answer
        query = self.train_query
        query_neis = self.train_query_neis
        return self._get_dataloader(query, query_neis, pred_answer=ans, mask_answer=None, modelist=mode, 
                                    gen_pred_mask=False, batch_size=self.args.batch_size,
                                    num_workers=self.args.num_workers, shuffle=True)

    def dataloader_test(self):
        mode = self.test_mode
        hardans = self.test_hard_answer
        easyans = self.test_easy_answer
        query = self.test_query
        query_neis = self.test_query_neis
        return self._get_dataloader(query, query_neis, pred_answer=hardans, mask_answer=easyans, modelist=mode,
                                         gen_pred_mask=True, batch_size=self.args.batch_size,
                                         num_workers=self.args.num_workers, shuffle=False)

    def dataloader_valid(self):
        mode = self.test_mode
        hardans = self.valid_hard_answer
        easyans = self.valid_easy_answer
        query = self.valid_query
        query_neis = self.valid_query_neis
        return self._get_dataloader(query, query_neis, pred_answer=hardans, mask_answer=easyans, modelist=mode,
                                         gen_pred_mask=True, batch_size=self.args.batch_size,
                                         num_workers=self.args.num_workers, shuffle=False)
