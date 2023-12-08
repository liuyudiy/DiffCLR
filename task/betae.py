import os
import pickle as pkl


def read_txt_triples(path):
    triples = []
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = [int(token) for token in line.strip().split("\t")]
            assert len(tokens) == 3
            triples.append(tokens)
    return triples[:20]


def read_pkl(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }


class BetaEDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_file(self, name):
        abs_path = os.path.join(self.data_path, name)
        if name.endswith('.txt'):
            data = read_txt_triples(abs_path)
            data = list(self.triple_sanitize(data))
        elif name.endswith('-answers.pkl'):
            data = read_pkl(abs_path)
        elif name.endswith('-queries.pkl'):
            data = read_pkl(abs_path)
            data = {query_name_dict[k]: v for k, v in data.items()}
        elif name.endswith('-neis.pkl'):
            data = read_pkl(abs_path)
            data = {query_name_dict[k]: v for k, v in data.items()}
        elif name.endswith('.pkl') and '2' in name and 'id' in name:
            data = read_pkl(abs_path)
        else:
            raise f'Unrecognized file: {name}'
        return data

    def calldata(self):
        all_files = ['train.txt', 'valid.txt', 'test.txt']
        return tuple(self.get_file(f) for f in all_files)


    def triple_sanitize(self, triples):
        for t in triples:
            a, r, b = t
            if r % 2 == 1:
                a, b = b, a
                r = int((r - 1) / 2)
                continue
            else:
                r = int(r / 2)
            t = (a, r, b)
            if t is not None:
                yield t