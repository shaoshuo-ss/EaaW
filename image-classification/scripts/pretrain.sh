gpus=$1

python image-classification/main.py \
    --gpus ${gpus} \
    --wm_mode "nowm" \
    --epochs 300 \
    --lr 0.01 \
    --momentum 0.9 \
    --optim "sgd" \
    --bs 128 \
    --wd 5e-4 \
    --eval_rounds 5 \
    --test_bs 512 \
    --model "ResNet18" \
    --dataset "cifar10" \
    --image_size 32 \
    --seed 42 \
    --save_dir "./results/" \
    --save_model \
    --mode "train"
