python train.py \
        --cuda \
        -v yolo_nano \
        -d voc \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 640 \
        --multi_scale_range 10 19 \
        -ms \
        --ema \
        --max_epoch 150 \
        --lr_epoch 90 120
        