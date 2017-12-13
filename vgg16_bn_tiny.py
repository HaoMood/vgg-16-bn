#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train CIFAR-10 using VGG-16-BN-tiny. It gives 93.39% test set accuracy.

Usage:
CUDA_VISIBLE_DEVICES=0 ./vgg16_bn_tiny.py --base_lr 5e-2 --batch_size 256 \
    --epochs 170 --weight_decay 1e-3
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
        self.features = torchvision.models.vgg16_bn(pretrained=False).features
        self.fc6 = torch.nn.Linear(512, 512)
        self.fc7 = torch.nn.Linear(512, 512)
        self.fc8 = torch.nn.Linear(512, 10)

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
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    torch.nn.init.constant(module.bias.data, val=0)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant(module.weight.data, val=1)
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
        self._solver = torch.optim.SGD(
            self._net.parameters(), lr=self._options['base_lr'], momentum=0.9,
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

    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._options['epochs']):
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
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, required=True,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs <= 0:
        raise AttributeError('--epochs parameter must >0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }
    manager = VGG16BNTinyManager(options)
    manager.train()


if __name__ == '__main__':
    main()
