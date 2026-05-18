# Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning

This repository is the official implementation of *Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning*

## Dataset
We provide the ReaL-TG QA dataset adapted from TGB datasets in `tgrl/dataset_raw/realtg`, which is in JSONL format for easy inspection and debugging.

We also provide the processed dataset in parquet format in `tgrl/dataset`, which can be directly used for LLM training, response generation, and evaluation.

In addition, we provide dataset construction scripts in `data_creator`, including the T-CGS implementation in the function `get_neighbor_topk_tppr` in line 331 of `data_creator/neighbor_tracker_analysis.py`

### Dataset Download and Preparation

To prepare datasets:

```bash
git clone https://github.com/shenyangHuang/TGB.git
cd TGB
pip install -e .
```

The project uses datasets from the Temporal Graph Benchmark (TGB). Dataset creation and preprocessing scripts are included in the repository and generate the required files automatically.

### Directory Structure

```text
tgrl/
├── dataset/
│   └── *.parquet
├── dataset_raw/
│   └── realtg/
│       └── *.jsonl
├── utils.py
└── logs/

data_creator/
├── neighbor_tracker_analysis.py
└── ...
```

### Dataset File Description

| Location | Format | Description |
|---|---|---|
| `tgrl/dataset_raw/realtg/` | JSONL | Raw QA-formatted temporal graph dataset |
| `tgrl/dataset/` | Parquet | Processed dataset for training and evaluation |
| `data_creator/` | Python scripts | Dataset generation and preprocessing scripts |

### Entity / Mapping Files

The paper uses anonymized temporal graph entities rather than semantic node labels. If mapping files are generated during preprocessing, store and document them under:

```text
dataset/mappings/
├── node_mapping.csv
├── edge_mapping.csv
└── timestamp_mapping.csv
```

| File | Purpose |
|---|---|
| `node_mapping.csv` | Maps original node identifiers to anonymized IDs |
| `edge_mapping.csv` | Stores interaction relationships |
| `timestamp_mapping.csv` | Maps timestamps used in temporal graph processing |


## Requirements and Environment Setup
Create virtual environment
```
conda create -n graphrl python==3.10
conda activate graphrl
```

Install verl==0.3.1.dev
```
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

Install vllm==0.8.5.post1
```
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

Install TGB from source to download datasets and run evaluation:
```
git clone git@github.com:shenyangHuang/TGB.git
pip install -e .   
```

Install tqdm and wandb
```
pip install tqdm==4.67.1
pip install wandb==1.52.0
```

## Model Training, Generation and Evaluation
### Copy the Source Code
Copy the directory of `tgrl` and paste it under the `PATH_TO_VERL/recipe/` directory
### Model Training
You can freely change training parameters according to verl's documentation
```
cd PATH_TO_VERL/recipe/tgrl/
./train_qwen3-4b_realtg.sh
```
### Response Generation
Before running the generation script, make sure the model checkpoint has been merged if model was trained on multiple GPUs
```
cd PATH_TO_VERL/scripts
python model_merger.py merge --backend fsdp --local_dir PATH_TO_CHECKPOINT_DIR --target_dir PATH_TO_VERL/recipe/tgrl/merged_checkpoints/verl_grpo_tgrl/realtg_4b
```
Now you can generate the responses using the trained model. You can freely change generation parameters according to verl's documentation
```
cd PATH_TO_VERL/recipe/tgrl/
./run_qwen3-4b_realtg.sh
```
### Evaluation
To evaluate model performance, run the following scripts that reads model responses and calculates the metrics
```
cd PATH_TO_VERL/recipe/tgrl/logs/test_generation/
python read_parquet.py
cd PATH_TO_VERL/recipe/tgrl/
python eval_mrr_realtg.py
```
The source code of penalized MRR is in the function `predict_link_complete_discount` in `tgrl/utils.py`

## LLM-as-a-Judge
To evaluate reasoning quality with the LLM judge, run the following script
```
cd PATH_TO_VERL/recipe/tgrl/
python llm_judge.py
```