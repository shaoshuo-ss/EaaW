model_path=$1
wm_length=$2
wm_path=$3
trigger_path=$4
gpus=$5


python image-classification/main.py \
    --gpus ${gpus} \
    --pre_train_path ${model_path} \
    --image_size 32 \
    --wm_mode "wm" \
    --test_bs 512 \
    --model "ResNet18" \
    --dataset "cifar10" \
    --optim "sgd" \
    --momentum 0.9 \
    --mode "test" \
    --wm_length ${wm_length} \
    --alpha1 1.0 \
    --epsilon 1e-2\
    --seed 42\
    --lam 1e-3 \
    --test_image_path ${trigger_path} \
    --save_dir "./results/" \
    --target_path ${wm_path}
