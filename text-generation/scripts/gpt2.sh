gpus=$1
wm_length=$2


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=${gpus} accelerate launch \
    --mixed_precision=bf16 --multi_gpu --main_process_port 29500 \
    text-generation/run_clm.py \
    --model_name_or_path models/gpt2 \
    --train_file data/ptb-text-only/ptb_train.txt \
    --validation_file data/ptb-text-only/ptb_valid.txt \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 20 \
    --learning_rate 3e-4 \
    --num_warmup_steps 50 \
    --alpha1 1.0 \
    --alpha2 1.0 \
    --low_cpu_mem_usage \
    --output_dir "./results/" \
    --train_num_samples 1000 \
    --mode "wm" \
    --max_mask_token_size 8 \
    --do_train \
    --wm_length ${wm_length} \
    --trigger_size 1 \
    --manual_dataset_name "ptb-text-only"