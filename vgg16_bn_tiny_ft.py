#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train CIFAR-10 by fine-tuning the VGG-16-BN ImageNet pre-trained model.
It gives 93.32% test set accuracy.

Usage:
CUDA_VISIBLE_DEVICES=1 ./vgg16_bn_tiny_ft.py --base_lr_fc 1e-2 \
    --base_lr_all 5e-2 --batch_size 256 --epochs_fc 0 --epochs_all 50 \
    --weight_decay 1e-3
"""


import os

import torch
import torchvision


__all__ = ['VGG16BNTiny', 'VGG16BNTinyManager']
__author__ = 'Hao Zhang'
__copyright__ = '2017 LAMDA'
__date__ = '2017-12-12'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-12-12'
__version__ = '1.0'


class VGG16BNTiny(torch.nn.Module):
    """VGG-16-BN-tiny model for CIFAR-10.

    The VGG-16-BN-tiny model is illustrated as follows. Besides, we add a BN
    layer after each convolution layer. Dropout is used after fc6 and fc7.
    The network accepts a 3*32*32 input, and the pool5 activation has shape
    512*1*1 since we down-sample 5 times.
       conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> pool5 -> fc6 (512)
    -> fc7 (512) -> fc8 (10).

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc6, torch.nn.Module: 512.
        fc7, torch.nn.Module: 512.
        fc8, torch.nn.Module: 10.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # self._features is a nn.Sequential torch.nn.Module.
        self.features = torchvision.models.vgg16_bn(pretrained=True).features
        self.fc6 = torch.nn.Linear(512, 512)
        self.fc7 = torch.nn.Linear(512, 512)
        self.fc8 = torch.nn.Linear(512, 10)

        # parameters() returns an iterator over module parameters.
        for param in self.features.parameters():
            param.requires_grad = False  # Freeze all convolution layers.
        # Initialize the fc layers.
        self._initialization()

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of size N*3*32*32.

        Returns:
            Score, torch.autograd.Variable of size N*10.
        """
        N, D, H, W = X.size()
        assert D == 3 and H == 32 and W == 32
        X = self.features(X)
        assert X.size() == (N, 512, 1, 1)
        X = X.view(N, 512)
        X = torch.nn.functional.relu(self.fc6(X))
        X = torch.nn.functional.dropout(X, training=self.training)
        X = torch.nn.functional.relu(self.fc7(X))
        X = torch.nn.functional.dropout(X, training=self.training)
        X = self.fc8(X)
        assert X.size() == (N, 10)
        return X

    def _initialization(self):
        """Initialization of convolution, fc, and bn layers."""
        # modules() returns an iterator over all modules in the network
        # recursively.
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    torch.nn.init.constant(module.bias.data, val=0)


class VGG16BNTinyManager(object):
    """Manager class to train VGG16-BN-Tiny.

    Attributes:
        _options: Hyperparameters.
        _net: VGG-16-BN-Tiny.
        _criterion: Cross-entropy loss.
        _solver: SGD with momentum.
        _scheduler: Reduce learning rate by a fator of 0.1 when plateau.
        _train_loader: Training data.
        _test_loader: Testing data.
    """
    def __init__(self, options):
        """Prepare the network, criterion, solver, and data.

        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        # Network.
        self._options = options
        self._net = VGG16BNTiny()
        self._net.cuda().half()
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss()
        self._criterion.cuda().half()
        # Solver.
        fc_parameters = [*list(self._net.fc6.parameters()),
                         *list(self._net.fc7.parameters()),
                         *list(self._net.fc8.parameters())]
        self._solver = torch.optim.SGD(
            fc_parameters, lr=self._options['base_lr_fc'], momentum=0.9,
            weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=10, verbose=True,
            threshold=1e-4)

        data_root = '/data/zhangh/data/data-pytorch'
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=train_transforms)
        test_data = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1024, shuffle=False, num_workers=4)

    def train(self, finetune_all):
        """Train the network.

        Args:
            finetune_all, bool: Whether to fine-tune the whole network.
        """
        if finetune_all:
            print('Fine-tuning all layers.')
            for param in self._net.features.parameters():
                param.requires_grad = True  # Fine-tune all layers
            self._solver = torch.optim.SGD(
                self._net.parameters(), lr=self._options['base_lr_all'],
                momentum=0.9, weight_decay=self._options['weight_decay'])
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._solver, mode='max', factor=0.1, patience=10, verbose=True,
                threshold=1e-4)
            epochs = self._options['epochs_all']
        else:
            print('Training classification layers.')
            epochs = self._options['epochs_fc']

        best_acc = 0.0
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(epochs):
            epoch_loss = []
            for X, y in self._train_loader:
                # Data.
                X = torch.autograd.Variable(X.cuda().half())
                y = torch.autograd.Variable(y.cuda())

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.data[0])
                # Backward pass.
                loss.backward()
                self._solver.step()
            train_acc = self._accuracy(self._train_loader)
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                print('*', end='')
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.train(False)
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.
            X = torch.autograd.Variable(X.cuda().half())
            y = torch.autograd.Variable(y.cuda())

            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data)
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total

    def getParameter(self):
        """Get the network's learnable parameters."""
        print(self._net.fc7.weight.data)
        print(self._net.fc7.weight.grad)

    def getStat(self):
        """Get the mean and std value for a certain dataset."""
        print('Compute mean and variance for training data.')
        data_root = '/data/zhangh/data/data-pytorch'
        train_data = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=None)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=4)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in train_loader:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        print(mean)
        print(std)

    def saveModel(self, path='./'):
        """Save the model onto disk.

        Args:
            path, str: Path to save the model.
        """
        torch.save(self._net.state_dict(), os.path.join(path, 'model.pkl'))

    def loadModel(self, path='./'):
        """Load the model from disk.

        Args:
            path, str: Path to load the model.
        """
        self._net.load_state_dict(torch.load(os.path.join(path, 'model.pkl')))


def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train CIFAR-10 using VGG-16-BN-Tiny.')
    parser.add_argument('--base_lr_fc', dest='base_lr_fc', type=float,
                        required=True,
                        help='Base learning rate for training the fc layer.')
    parser.add_argument('--base_lr_all', dest='base_lr_all', type=float,
                        required=True,
                        help='Base learning rate for fine-tuning all layers..')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs_fc', dest='epochs_fc', type=int,
                        required=True, help='Epochs for training the fc layer.')
    parser.add_argument('--epochs_all', dest='epochs_all', type=int,
                        required=True, help='Epochs for fine-tuning all layer.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    args = parser.parse_args()
    if args.base_lr_fc <= 0:
        raise AttributeError('--base_lr_fc parameter must >0.')
    if args.base_lr_all <= 0:
        raise AttributeError('--base_lr_all parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs_fc < 0:
        raise AttributeError('--epochs_fc parameter must >=0.')
    if args.epochs_all < 0:
        raise AttributeError('--epochs_all parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr_fc': args.base_lr_fc,
        'base_lr_all': args.base_lr_all,
        'batch_size': args.batch_size,
        'epochs_fc': args.epochs_fc,
        'epochs_all': args.epochs_all,
        'weight_decay': args.weight_decay,
    }
    manager = VGG16BNTinyManager(options)
    manager.train(finetune_all=False)
    manager.train(finetune_all=True)


if __name__ == '__main__':
    main()
