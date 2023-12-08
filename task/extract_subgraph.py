import os
import json
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from itertools import *
from scipy.sparse import csc_matrix
import scipy.sparse as ssp
from betae import BetaEDataset


def process_files(args):
    betae = BetaEDataset(args.data_dir)

    id2ent = betae.get_file("id2ent.pkl")
    id2rel = betae.get_file("id2rel.pkl")
    num_nodes = len(id2ent)
    relation_cnt = len(id2rel) // 2

    train_triple = betae.get_file("train.txt")
    valid_triple = betae.get_file("valid.txt")
    test_triple = betae.get_file("test.txt")
    triplets = {"train": np.array(train_triple), "valid": np.array(valid_triple), "test": np.array(test_triple)} 

    adj_list = []
    for i in range(relation_cnt):
        idx = np.argwhere(triplets["train"][:, 1] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), 
                                    (triplets["train"][:, 0][idx].squeeze(1), 
                                     triplets["train"][:, 2][idx].squeeze(1))), 
                                     shape=(num_nodes, num_nodes)))

    betae = BetaEDataset(args.data_dir)
    train_query = betae.get_file("train-queries.pkl")
    valid_query = betae.get_file("valid-queries.pkl")
    test_query = betae.get_file("test-queries.pkl")
    queries = {"train": train_query, "valid": valid_query, "test": test_query}

    return adj_list, triplets, queries


def get_query_entity(mode, q):
    entity_list = []
    relation_list = []

    def sanitize_edge(r):
        if r % 2 == 1:
            r = int((r-1) / 2)
        else:
            r = int(r / 2)
        return r

    if mode == '1p':
        entity_list.extend([q[0]])
        relation_list = list(map(sanitize_edge, [q[1][0]])) 
    elif mode == '2p':
        entity_list.extend([q[0]])
        relation_list = list(map(sanitize_edge, [q[1][0], q[1][1]])) 
    elif mode == '3p':
        entity_list.extend([q[0]])
        relation_list = list(map(sanitize_edge, [q[1][0], q[1][1], q[1][2]]))
    elif mode == '2i':
        entity_list.extend([q[0][0], q[1][0]])
        relation_list = list(map(sanitize_edge, [q[0][1][0], q[1][1][0]]))
    elif mode == '3i':
        entity_list.extend([q[0][0], q[1][0], q[2][0]])
        relation_list = list(map(sanitize_edge, [q[0][1][0], q[1][1][0], q[2][1][0]]))
    elif mode == 'pi':
        entity_list.extend([q[0][0], q[1][0]])
        relation_list = list(map(sanitize_edge, [q[0][1][0], q[0][1][1], q[1][1][0]]))
    elif mode == 'ip':
        entity_list.extend([q[0][0][0], q[0][1][0]])
        relation_list = list(map(sanitize_edge, [q[0][0][1][0], q[0][1][1][0], q[1][0]]))
    elif mode == '2u':
        entity_list.extend([q[0][0], q[1][0]])
        relation_list = list(map(sanitize_edge, [q[0][1][0], q[1][1][0]]))
    elif mode == 'up':
        entity_list.extend([q[0][0][0], q[0][1][0]])
        relation_list = list(map(sanitize_edge, [q[0][0][1][0], q[0][1][1][0], q[1][0]]))
    else:
        assert False
    return entity_list, relation_list


def incidence_matrix(adj_list):
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)

    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _get_neighbors(adj, nodes):
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:
        for v in current_lvl:
            visited.add(v)
        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))
        yield next_lvl

        current_lvl = set.union(next_lvl)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return lvls


def subgraph_extraction(entity_list, relation_list, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T    

    nei_int_set, nei_hop_dict = set(), {}
    for ent in entity_list:
        neis_hop = get_neighbor_nodes(set([ent]), A_incidence, h, max_nodes_per_hop)
        neis_all = set().union(*neis_hop)
        if nei_int_set:
            nei_int_set = nei_int_set.intersection(neis_all)
        else:
            nei_int_set = neis_all
        for hop, neis in enumerate(neis_hop):
            for nei in neis:
                if nei not in nei_hop_dict:
                    nei_hop_dict[nei] = hop+1
                else:
                    nei_hop_dict[nei] = max(hop+1, nei_hop_dict[nei])
        
    if len(nei_int_set) > args.max_nodes_hop:
        nei_int_set = set(random.sample(list(nei_int_set), args.max_nodes_hop))

    neighbors_dict, relations_dict, triples_rel_dict = {}, {}, {}
    for ent in entity_list:
        neighbors_dict[ent] = h + 1
    for nei in nei_int_set:
        if nei not in entity_list:
            neighbors_dict[nei] = h + 1 - nei_hop_dict[nei]
    nodes = entity_list + list(nei_int_set)
    nodes = list(set(nodes))
    subgraph = [adj[nodes, :][:, nodes] for adj in A_list]
    subgraph = [adj.todense().tolist() for adj in subgraph]
    for rel in relation_list:
        relations_dict[rel] = h + 1
    for element in combinations(nodes,2):
        h, t = element
        h_index, t_index = nodes.index(h), nodes.index(t)
        for rel, adj in enumerate(subgraph):
            if adj[h_index][t_index]:
                if rel not in relations_dict:
                    relations_dict[rel] = min(neighbors_dict[h], neighbors_dict[t])
                    triples_rel_dict[rel] = (h, t)
            if adj[t_index][h_index]:
                if rel not in relations_dict:
                    relations_dict[rel] = min(neighbors_dict[h], neighbors_dict[t])
                    triples_rel_dict[rel] = (t, h)   
 
    return neighbors_dict, relations_dict, triples_rel_dict


def generate_subgraph_datasets(args):
    adj_list, triplets, queries = process_files(args)

    for split in args.split:
        query_list = queries[split]
        query_neis_dict = {}
        for mode in tqdm(args.mode_dict[split]):
            query_neis_dict[mode] = {}
            for query in tqdm(query_list[mode]):
                entity_list, relation_list = get_query_entity(mode, query)
                neis_dict, rels_dict, tris_rel_dict = subgraph_extraction(
                    entity_list, relation_list, adj_list, 
                    args.hop, args.enclosing_sub_graph, args.max_nodes_per_hop)
                query_neis_dict[mode][query] = [neis_dict, rels_dict, tris_rel_dict]

        name = split +'-queries-neis.pkl'
        save_path = os.path.join(args.data_dir, name)
        with open(save_path, 'wb') as f:
            pickle.dump(query_neis_dict, f)


def process_files_answers(args):
    betae = BetaEDataset(args.data_dir)
    train_answers = betae.get_file("train-answers.pkl")
    test_easy_answers = betae.get_file("test-easy-answers.pkl")
    test_hard_answers = betae.get_file("test-hard-answers.pkl")
    answers = {"train": train_answers, "test": test_hard_answers}  
    return answers


def remove_nodes(A_incidence, nodes):
    if nodes:
        idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    else:
        idxs_wo_nodes = list(set(range(A_incidence.shape[1])))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def get_parser():
    parser = argparse.ArgumentParser(description='Subgraph Extraction')
    parser.add_argument("--data_dir", type=str, default="./kg_data/NELL-betae")
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--enclosing_sub_graph", type=bool, default=True)
    parser.add_argument("--max_nodes_hop", type=int, default=100)
    parser.add_argument("--max_nodes_per_hop", type=int, default=1000)
    parser.add_argument("--split", type=list, default=[])
    parser.add_argument("--mode_dict", type=dict, default={})
    args = parser.parse_args()

    mode_dict = {
        "train": ["1p", "2p", "3p", "2i", "3i"],
        "valid": ["1p", "2p", "3p", "2i", "3i"],
        "test": ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"]
        }
    args.mode_dict = mode_dict
    args.split = list(mode_dict.keys())

    return args


if __name__ == "__main__":
    args = get_parser()
    print(args)
    generate_subgraph_datasets(args)
    
