#!/bin/bash
# 'model_name: VGG16, VGG16_TR, VGG16_TR_FT, INCEPTION_TR_FT, DENSE201_TR_FT, RES50_TR_FT, ALEXNET'
# layer: INCEPTION_TR_FT --> conv2d_94, DENSE201_TR_FT --> conv5_block32_2_conv, RES50_TR_FT --> res5c_branch2c
#        VGG16_TR_FT --> block5_conv3, ALEXNET --> conv2d_5

SECONDS=0

## Virtual environment setting ##
source /home/hbcho/anaconda3/etc/profile.d/conda.sh
conda activate DPR_tf114-gpu
#conda activate py36


## GPU selection ##
export CUDA_VISIBLE_DEVICES=0:

## Training and predction option ##
DATA=pro           #ex) cs, pro, total // cs: CS 9300, pro: ProMax, total: CS 9300 + ProMax
MODEL=INCEPTION_TR_FT
SAVE=0427
FOLD=0            #ex) 0,1,2,3,4,5 // 0: 1-5 folds
LAYER=conv2d_94
THRES=0.5 

python step2_Train.py -data $DATA -model_name $MODEL -save_name $SAVE -fold $FOLD
python step3_Prediction.py -data $DATA -model_name $MODEL -save_name $SAVE
python step4_Gradcam.py -data $DATA -model_name $MODEL -save_name $SAVE -activation_layer $LAYER -threshold $THRES -fold $FOLD

DATA=cs
python step2_Train.py -data $DATA -model_name $MODEL -save_name $SAVE -fold $FOLD
python step3_Prediction.py -data $DATA -model_name $MODEL -save_name $SAVE
python step4_Gradcam.py -data $DATA -model_name $MODEL -save_name $SAVE -activation_layer $LAYER -threshold $THRES -fold $FOLD

DATA=total
python step2_Train.py -data $DATA -model_name $MODEL -save_name $SAVE -fold $FOLD
python step3_Prediction.py -data $DATA -model_name $MODEL -save_name $SAVE
python step4_Gradcam.py -data $DATA -model_name $MODEL -save_name $SAVE -activation_layer $LAYER -threshold $THRES -fold $FOLD



duration=$SECONDS
echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
