model_path=$1
wm_length=$2
wm_path=$3
gpus=$4

python image-classification/main.py \
    --gpus ${gpus} \
    --pre_train_path ${model_path} \
    --image_size 32 \
    --save_model \
    --wm_mode "wm" \
    --test_bs 300 \
    --model "ResNet18" \
    --dataset "cifar10" \
    --optim "sgd" \
    --lr 1e-6 \
    --wd 1e-3 \
    --momentum 0.9 \
    --mode "train" \
    --wm_length ${wm_length} \
    --alpha1 1.0 \
    --epsilon 1e-2\
    --bs 128\
    --seed 42\
    --epochs 30\
    --lam 1e-3 \
    --pattern_path "test.png" \
    --mix_type "cover" \
    --base_trigger "image" \
    --save_dir "./results/" \
    --wm_loss "HingeLike" \
    --target_path ${wm_path}