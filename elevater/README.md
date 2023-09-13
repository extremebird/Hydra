## Hydra on ELEVATER
This folder is for adapting Hydra on elevater benchmark.

## Result
|Method|#Params(M)|Avg Acc.|PE|Caltech101|CIFAR10|CIFAR100|Country211|DTD|EuroSat|FER2013|FGVCAircraft|Food101|GTSRB|HatefulMemes|KittiDistance|MNIST|Flowers102|OxfordPets|PatchCamelyon|SST2|RESISC45|StanfordCars|VOC2007|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Full tuning|87.9|65.49|0.498|87.64|**91.11**|71.52|15.75|54.36|85.24|**52.72**|26.22|83.28|74.05|55.64|39.15|65.55|80.55|87.31|64.92|59.09|75.61|57.21|82.95|
|Linear-probing|0.03|66.32|0.663|90.96|90.35|67.31|17.36|62.04|72.95|51.91|29.52|83.82|56.47|55.83|40.37|77.50|92.29|88.03|59.00|59.36|78.10|68.30|**84.99**|
|Adapter|1.24|65.08|0.647|90.18|90.14|73.57|16.83|57.13|67.97|41.76|30.52|83.58|58.50|48.91|37.18|80.34|90.78|86.52|59.92|58.70|79.22|67.68|82.22|
|LoRA|0.18|61.48|0.614|87.64|90.52|69.69|17.12|50.16|74.03|51.04|20.01|83.76|42.96|55.88|48.05|61.36|74.28|85.49|63.20|57.04|62.09|54.89|80.33|
|Compacter|0.08|62.79|0.628|89.02|79.96|44.33|**28.22**|52.93|50.48|35.46|**41.13**|78.28|66.90|47.60|**57.72**|85.82|88.29|79.23|61.83|**64.22**|63.76|64.79|75.84|
|KAdaptation|0.08|68.92|0.689|88.96|90.03|73.92|17.53|63.97|76.25|47.45|30.04|**84.38**|80.71|55.86|42.29|85.20|**93.19**|89.05|63.39|59.18|79.96|70.21|84.49|
|Hydra|0.20|**70.95**|**0.709**|**91.23**|90.89|**74.20**|17.75|**64.47**|**87.00**|51.10|33.05|84.27|**87.11**|**55.91**|42.05|**90.76**|93.18|**89.38**|**70.83**|59.58|**82.41**|**71.19**|82.66|

## Dataset
You can follow dataset in [Parameter-efficient Model Adaptation for Vision Transformers](https://github.com/eric-ai-lab/PEViT).
The configs of datasets and pre-trained models are in `vision_benchmark/resources`. The root of dataset can be changed in `script/hydra.sh`.
You can refer to elevator benchmark paper([link][https://arxiv.org/abs/2204.08790]) for manually downloaded data.

## Dependency Setup

```Shell
conda env create -f requirements.yaml
conda activate hydra
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## Run experiments
All experiments are run on a single NVIDIA-A100 GPU.
Script includes both training and evaluation.

```Shell
cd script
bash Hydra_clip.sh
```
