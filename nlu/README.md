# Hydra: Multi-head Low-rank Adaptation for Parameter Efficient Fine-tuning

This directory includes Natural Language Understanding (NLU) experiments with Hydra in RoBERTa (base).

## Results

|Method|#Params (M)|Avg.|MNLI|SST-2|MRPC|CoLA|QNLI|QQP|RTE|STS-B|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Full tuning|125|86.4|**87.6**|94.8|90.2|63.6|92.8|**91.9**|78.7|91.2|
|BitFit|0.1|85.2|84.7|93.7|**92.7**|62.0|91.8|84.0|81.5|90.8|
|AdapterDrop|0.3|84.4|87.1|94.2|88.5|60.8|93.1|90.2|71.5|89.7|
|AdapterDrop|0.9|85.4|87.3|94.7|88.4|62.6|93.0|90.6|75.9|90.3|
|LoRA|0.3|87.2|87.5|**95.1**|89.7|63.4|93.3|90.8|86.6|91.5|
|Hydra|0.3|**87.9**|87.5|95.0|92.2|**65.4**|92.8|90.8|**87.4**|**91.7**|

## Dependency Setup
```console
conda env create -f environment.yml
conda activate hydra_nlu
pip install .
```
## Run experiments
Run experiments on each dataset.  
All experiments are run on 4 NVIDIA-A100 GPUs (80GB).  
Each script includes both training and evaluation.

```console
# MNLI
bash hydra_mnli.sh
# SST-2
bash hydra_sst2.sh
# MRPC
bash hydra_mrpc.sh
# CoLA
bash hydra_cola.sh
# QNLI
bash hydra_qnli.sh
# QQP
bash hydra_qqp.sh
# RTE
bash hydra_rte.sh
# STS-B
bash hydra_stsb.sh
```