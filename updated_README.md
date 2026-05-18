# Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning

This repository is the official implementation of *Self-Exploring
Language Models for Explainable Link Forecasting on Temporal Graphs via
Reinforcement Learning*.

## Dataset Preparation

The project uses the **ReaL‑TG QA dataset** adapted from the Temporal
Graph Benchmark (TGB). Data is provided in two forms:

  ------------------------------------------------------------------------
  Dataset Format                            Location Description
  --------------------- ---------------------------- ---------------------
  Raw JSONL               `tgrl/dataset_raw/realtg/` Human-readable
                                                     QA-style temporal
                                                     graph samples

  Processed Parquet                  `tgrl/dataset/` Optimized for
                                                     training, generation,
                                                     and evaluation

  Data construction                  `data_creator/` Scripts used to
  scripts                                            generate datasets and
                                                     context graphs
  ------------------------------------------------------------------------

The Temporal Context Graph Selection (T‑CGS) implementation is available
in:

`data_creator/neighbor_tracker_analysis.py` → `get_neighbor_topk_tppr()`

### Download Steps

Clone the repository:

``` bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

Install TGB from source:

``` bash
git clone https://github.com/shenyangHuang/TGB.git
cd TGB
pip install -e .
```

After installation, TGB datasets can be downloaded and processed through
the dataset creation scripts.

### Directory Structure

``` text
project_root/
├── tgrl/
│   ├── dataset/
│   │   └── *.parquet
│   ├── dataset_raw/
│   │   └── realtg/
│   │       └── *.jsonl
│   ├── utils.py
│   └── logs/
├── data_creator/
│   ├── neighbor_tracker_analysis.py
│   └── ...
└── README.md
```

## Knowledge Graph / Entity Mapping Files

The paper relies on anonymized graph entities and temporal interactions.
If dataset mapping files are generated during preprocessing, include
them under:

``` text
dataset/
└── mappings/
    ├── node_mapping.csv
    ├── edge_mapping.csv
    └── timestamp_mapping.csv
```

Suggested descriptions:

  File                      Purpose
  ------------------------- -----------------------------------------------
  `node_mapping.csv`        Maps original node IDs to anonymized IDs
  `edge_mapping.csv`        Stores interaction relations between entities
  `timestamp_mapping.csv`   Maps temporal values used in TG processing

Document all additional mapping files here whenever datasets are
extended.

## Requirements and Environment Setup

``` bash
conda create -n graphrl python==3.10
conda activate graphrl
```

Install verl:

``` bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

Install vllm:

``` bash
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

Install dependencies:

``` bash
pip install tqdm==4.67.1
pip install wandb==1.52.0
```

## Training

``` bash
cd PATH_TO_VERL/recipe/tgrl/
./train_qwen3-4b_realtg.sh
```

## Generation

``` bash
./run_qwen3-4b_realtg.sh
```

## Evaluation

``` bash
python eval_mrr_realtg.py
```
