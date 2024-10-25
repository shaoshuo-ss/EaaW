# Explanation as a Watermark (EaaW)

## Introduction
This is the official implementation for our paper "[Explanation as a Watermark: Towards Harmless and Multi-bit Model Ownership Verification via Watermarking Feature Attribution](https://arxiv.org/abs/2405.04825)". This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2025. This project is developed on Python3 and Pytorch.

## Getting Start

### Protecting Image Classification Models
First, create a virtual environment using Anaconda.
```
conda create -n eaaw python=3.8
conda activate eaaw
```

Second, you need to install the necessary packages to run EaaW, including pytorch, opencv-python, tqdm, piqa, and scipy.
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install opencv-python
pip install tqdm
pip install piqa
pip install scipy
```

After installing these packages, we need to pre-train a model. For instance, the bash file `image-classification/scripts/pretrain.sh` can be utilized to train a ResNet-18 model on the CIFAR-10 dataset, as follows. It will automatically download the `cifar-10-python.tar.gz` to `data/cifar10/`. You can also download manually and put the `.tar.gz` file to `data/cifar10/`.
```
bash image-classification/scripts/pretrain.sh {gpus}
```

The `{gpus}` signifies the ids of GPUs used for training and fine-tuning. The device ids should be seperated by commas, $e.g.$, `'0,1,2'`.

After pre-training, you will get a `model_best.pth` file in `results/nowm/cifar10/ResNet18/train/{time}/`. You can run the following bash file to embed the watermark into the pre-trained model. This bash file utilizes the *noise* as the example trigger sample.

```
bash image-classification/scripts/embed.sh {model_path} {wm_length} {wm_path} {gpus}
```
The parameters are discribed below.
- `{model_path}`: The path of the pre-trained model.
- `{wm_length}`: The length of the embedded watermark.
- `{wm_path}`: The path of the watermark. The watermark should be an image. To reproduce our results, you can use `target.png`.
- `{gpus}`: The ids of the GPUs.

You can also use the `test.sh` file to extract the watermark from the watermarked model.
```
bash image-classification/scripts/test.sh {model_path} {wm_length} {wm_path} {trigger_path} {gpus}
```
The parameters are discribed below.
- `{model_path}`: The path of the watermarked model, $i.e.$, the model of the last epoch.
- `{wm_length}`: The length of the embedded watermark.
- `{wm_path}`: The path of the watermark. The watermark should be an image. To reproduce our results, you can use `target.png`.
- `{trigger_path}`: The path of the trigger samples. You can get a `perturbed_images/` folder that contains the trigger samples and use the path of the folder as `{trigger_path}`.
- `{gpus}`: The ids of the GPUs.


### Protecting Text Generation Models

The code of applying EaaW to protect text generation models will soon be released.

## Citation Info

If you find this repository useful for your research, it will be greatly appreciated to cite our paper😄.
```
@inproceedings{shao2025explanation,
    title={Explanation as a Watermark: Towards Harmless and Multi-bit Model Ownership Verification via Watermarking Feature Attribution},
    author={Shao, Shuo and Li, Yiming and Yao, Hongwei and He, Yiling and Qin, Zhan and Ren, Kui},
    booktitle={Network and Distributed System Security Symposium},
    year={2025}
}
```
