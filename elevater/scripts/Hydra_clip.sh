############## Configuration section begins ##################

# Model Config: [vitb32_CLIP, vitb16_CLIP, mae_vitb16, mocov3_vitb16, vit_base_patch16_224, vit_base_patch32_224, deit_base_patch16_224]
model_cfg=vitb32_CLIP

# Use FP32 [default: True]
use_fp32=True

# Model checkpoint
model_ckpt=.

# output directory
output_dir=./output
data_dir=./data1

############ Configurations for hyperparameter tuning begin ############
# set to True to disable the automatic hyperparameter tuning
# and set the learning rate and weight accordingly below
# This option is only effective for linear probe and finetuning.

disable_hyperparameter_tuning=False
# disable_hyperparameter_tuning=True
learning_rate=0.01
l2_weight_decay=1e-6

############ Configurations for hyperparameter tuning end   ############


############ Configurations for linear_probe/finetune begin ############

# Shots: {5, 20, 50} for few shot, and -1 for full-shot
num_shots=5

# Whether to init the linear head with the text encoder
init_head_with_text_encoder=True

# whether to merge the encoder and the linear head
merge_encoder_and_proj=False

############ Configurations for linear_probe/finetune end   ############


############ Configurations for adding knowledge begin ############
# Please change the knowledge source accordingly.

use_wordnet_hierachy=False
use_wordnet_definition=False
use_wiktionary_definition=False
use_gpt3=False
use_gpt3_count=0

############ Configurations for adding knowledge end ############


############## Configuration section ends ##################

cd ../vision_benchmark

for dataset in caltech101 cifar10 cifar100 country211 dtd eurosat-clip fer2013 fgvc-aircraft-2013b food101 gtsrb hateful-memes kitti-distance mnist flower102 oxford-iiit-pets patchcamelyon rendered-sst2 resisc45-clip stanfordcar voc2007classification
do
    CUDA_VISIBLE_DEVICES=2 python commands/Hydra_clip.py --ds resources/datasets/$dataset.yaml --model resources/model/$model_cfg.yaml --no-tuning $disable_hyperparameter_tuning --lr $learning_rate --l2 $l2_weight_decay MODEL.CLIP_FP32 $use_fp32 DATASET.NUM_SAMPLES_PER_CLASS $num_shots DATASET.ROOT $data_dir/datasets OUTPUT_DIR $output_dir/$random_seed/$model_cfg/log DATASET.RANDOM_SEED_SAMPLING 2 TRAIN.INIT_HEAD_WITH_TEXT_ENCODER $init_head_with_text_encoder TRAIN.MERGE_ENCODER_AND_HEAD_PROJ $merge_encoder_and_proj KNOWLEDGE.WORDNET.USE_HIERARCHY $use_wordnet_hierachy KNOWLEDGE.WORDNET.USE_DEFINITION $use_wordnet_definition KNOWLEDGE.WIKITIONARY.USE_DEFINITION $use_wiktionary_definition KNOWLEDGE.GPT3.USE_GPT3 $use_gpt3 KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS $use_gpt3_count TEST.MODEL_FILE $model_ckpt
done