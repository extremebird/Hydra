## Hydra on VTAB-1K
This folder is for adapting Hydra on VTAB-1K benchmark.

## Result
|Method|#Params (M)|Cifar100|Caltech101|DTD|Flower102|Pets|SVHN|Sun397|Camelyon|EuroSAT|Resisc45|Retinopathy|Clevr-Count|Clevr-Dist|DMLab|KITTI-Dist|dSPR-Loc|dSPR-Ori|sNORB-Azim|sNORB-Ele|Avg.|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Full tuning|85.8|68.9|87.7|64.3|97.2|86.9|87.4|38.8|79.7|95.7|84.2|73.9|56.3|58.6|41.7|65.5|57.5|46.7|25.7|29.1|68.9|
|Linear probe|0.04|64.4|85.0|63.2|97.0|86.3|36.6|51.0|78.5|87.5|68.5|74.0|34.3|30.6|33.2|55.4|12.5|20.0|9.6|19.2|57.6|
|Adapter|0.16|69.2|90.1|68.0|98.8|89.9|82.8|54.3|84.0|94.9|81.9|75.5|80.9|65.3|48.6|78.3|74.8|48.5|29.9|41.6|73.9|
|AdaptFormer|0.16|70.8|91.2|70.5|99.1|90.9|86.6|54.8|83.0|95.8|84.4|**76.3**|81.9|64.3|49.3|80.3|76.3|45.7|31.7|41.1|74.7|
|LoRA|0.29|67.1|91.4|69.4|98.8|90.4|85.3|54.0|84.9|95.3|84.4|73.6|82.9|**69.2**|49.8|78.5|75.7|47.1|31.0|44.0|74.5|
|VPT|0.53|**78.8**|90.8|65.8|98.0|88.3|78.1|49.6|81.8|96.1|83.4|68.4|68.5|60.0|46.5|72.8|73.6|47.9|32.9|37.8|72.0|
|NOAH|0.36|69.6|**92.7**|70.2|99.1|90.4|86.1|53.7|84.4|95.4|83.9|75.8|82.8|68.9|49.9|81.7|**81.8**|48.3|32.8|**44.2**|75.5|
|SSF|0.22|69.0|92.6|**75.1**|**99.4**|**91.8**|90.2|52.9|**87.4**|95.9|**87.4**|75.5|75.9|62.3|**53.3**|80.6|77.3|**54.9**|29.5|37.9|75.7|
|FacT-TK|0.07|70.6|90.6|70.8|99.1|90.7|88.6|54.1|84.8|**96.2**|84.5|75.7|82.6|68.2|49.8|80.7|80.8|47.4|33.2|43.0|75.6|
|RepAdapter|0.22|72.4|91.6|71.0|99.2|91.4|**90.7**|55.1|85.3|95.9|84.6|75.9|82.3|68.0|50.4|79.9|80.4|49.2|**38.6**|41.0|76.1|
|Hydra|0.28|72.7|91.3|72.0|99.2|91.4|**90.7**|**55.5**|85.8|96.0|86.1|75.9|**83.2**|68.2|50.9|**82.3**|80.3|50.8|34.5|43.1|**76.5**|

## Dataset
We provide the prepared datasets, which can be download from  [google drive](https://drive.google.com/file/d/1jRinJ9nDErBtmIOj3GL-xl2bpchSycdz/view?usp=drive_link).
After unzip at the workspace, you can use the data.
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to root folder of workspace.

## Dependency Setup
After generating environment with python, run

```Shell
pip install -r requirement.yaml
```

## Training
1. Train Hydra
```sh 
bash scripts/train_hydra.sh
``` 

2. Test Hydra
```sh 
bash scripts/test_hydra.sh
```

