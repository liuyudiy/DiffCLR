# DiffCLR

This is the original implementation for WSDM 2024 paper
[Generative Models for Complex Logical Reasoning over Knowledge Graphs]

## Prerequisites

### Enviroment

* `pytorch == 1.12.1`
* `pytorch-geometric == 2.2.0`
* `scipy == 1.9.3`

### Dataset
We use FB15k-237 and NELL995 dataset for knowledge graph reasoning. 
Dataset can be downloaded from [http://snap.stanford.edu/betae/KG_data.zip](http://snap.stanford.edu/betae/KG_data.zip). And put in the data directory.


## Reproduction

The parameters in the paper is preloaded in [`configs/`](configs/). And modify the corresponding parameters here according to the requirements.


### Train & Test

Commands for reproducing the results for `FB15k-237`:

```shell
# pretrain model
python main.py --config configs/fb15k237.json --task pretrain1

# finetine model
python main.py --config configs/fb15k237.json --task reasoning
```

Commands for reproducing the results for `NELL995`:

```shell
# pretrain model
python main.py --config configs/nell995.json --task pretrain1

# finetine model
python main.py --config configs/nell995.json --task reasoning
```

