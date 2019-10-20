N_GPU=2
WDIR='seresnext101v01'
FOLD=5
SIZE='256'

bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=darraghdog/kaggle:apex_build \
            -m dbslp1828 -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/rsna/scripts/$WDIR && nvidia-smi && python3 trainorig.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --epochs 8 --fold $FOLD  --lr 0.00002 --batchsize 128  --workpath scripts/$WDIR  \
            --imgpath data/mount/512X512X6/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin"