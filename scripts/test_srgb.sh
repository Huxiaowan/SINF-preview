eval "$(conda shell.bash hook)"
conda activate flornn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

cd test_models
python sRGB_test.py \
    --model SINF \
    --num_resblocks 5 \
    --noise_sigmas 30 \
    --model_file ../logs/pretrained.pth \
    --test_path ../datasets/Videos/Set8
