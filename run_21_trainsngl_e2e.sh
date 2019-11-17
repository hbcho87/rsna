N_GPU=4
WDIR='resnext101v12'
FOLD=6
SIZE='480'

# Run image classifier for all with cropping
python3 scripts/trainorig.py  \
            --logmsg Rsna-lb-$SIZE-fp16 --start 0 --epochs 5 --fold $FOLD  --lr 0.00002 --batchsize 64  --workpath scripts/resnext101v01  \
            --imgpath data/proc/ --size $SIZE --weightsname weights/model_512_resnext101$FOLD.bin --autocrop T


