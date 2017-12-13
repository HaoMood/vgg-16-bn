# VGG-16-BN
Train CIFAR-10 using VGG-16-BN, which gives 93.39% test set accuracy. This is
a PyTorch implementation.

## Usage
CUDA_VISIBLE_DEVICES=0 ./vgg16_bn_tiny.py --base_lr 5e-2 --batch_size 256 \
    --epochs 170 --weight_decay 1e-3
    
## Description
The model used here is a modified version of VGG-16-BN. Besides, we add a BN
layer after each convolution layer. Dropout is used after fc6 and fc7. The 
network accepts a 3*32*32 input, and the pool5 activation has shape 512*1*1 
since we down-sample 5 times.

   conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
-> conv4^3 (512) -> pool4 -> conv5^3 (512) -> pool5 -> fc6 (512)
-> fc7 (512) -> fc8 (10).
