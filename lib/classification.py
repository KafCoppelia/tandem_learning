import torch
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import encoding


def training(model, trainloader, optimizer, criterion, device):
    """
Utility function for training the model on the CIFAR-10 dataset.
Params
------
- model: model at the begining of the epoch
- trainloader: data loader for train set
    - optimizer: training optimizer
    - criterion: training criterion
    - device: cpu or gpu
Returns
-------
- model: updated model
- acc_train: average training accuracy over the epoch
- epoch_loss: average training loss over the epoch
"""
    model.train()  # Put the model in train mode

    running_loss = 0.0
    total = 0
    correct = 0
    for i_batch, (inputs, labels) in enumerate(trainloader, 1):
        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
            labels.type(torch.LongTensor).to(device)

        # Model computation and weight update
        y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss/i_batch
    acc_train = correct/total

    return model, acc_train, epoch_loss


def testing(model, testLoader, criterion, device):
    """
Utility function for testing the model on the CIFAR-10 dataset.
Params
------
- model: model to be tested
- testLoader: data loader for test set
    - criterion: testing criterion
    - device: cpu or gpu
Returns
-------
- acc_train: average training accuracy over the epoch
- epoch_loss: average training loss over the epoch
"""
    model.eval()  # Put the model in test mode

    running_loss = 0.0
    correct = 0
    total = 0
    for data in testLoader:
        inputs, labels = data

        # Transfer to GPU
        inputs, labels = inputs.type(torch.FloatTensor).to(device), \
            labels.type(torch.LongTensor).to(device)

        # forward pass
        y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    # calculate epoch statistics
    epoch_loss = running_loss/len(testLoader)
    acc = correct/total

    return acc, epoch_loss


def training_sj(model, trainloader, optimizer, device):
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0

    for img, label in trainloader:
        optimizer.zero_grad()
        img = img.float().to(device)
        label = label.to(device)
        label_onehot = F.one_hot(label, 10).float()

        out_fr = model(img)
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()
        optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
        functional.reset_net(model)

        train_loss /= train_samples
        train_acc /= train_samples

    return model, train_acc, train_loss


def testing_sj(model, testLoader, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    
    with torch.no_grad():
        for frame, label in testLoader:
            frame = frame.float().to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = model(frame)
            loss = F.mse_loss(out_fr, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            
            functional.reset_net(model)

	# calculate epoch statistics
    test_loss /= test_samples
    test_acc /= test_samples
    
    return test_acc, test_loss