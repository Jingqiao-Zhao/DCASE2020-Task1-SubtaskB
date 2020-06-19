import config
from  data_generator import data_generate,get_data
import torch.nn as nn
import torch.optim as optim
import torch
import os
from model import DD_CNN
from evaluate import evaluate_data


print('torch.cuda.is_available() is ',torch.cuda.is_available())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_train,y_train,x_test, y_test= data_generate()
batch_size = config.batch_size
train_data,test_data = get_data(x_train,y_train,x_test, y_test,batch_size)


net = DD_CNN()

def accuracy(predictions, labels):
    pred = torch.max(predictions, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

criterion = nn.CrossEntropyLoss()
optimizer=optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)

num_epochs = config.epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
best_test_acc = 0
best_epoch = 0
label = config.labels
for epoch in range(num_epochs):

    train_rights = []

    for batch_idx, (data, target) in enumerate(train_data):
        data = data.to(device)
        target = target.to(device)
        net.train()
        output = net(data)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        right = accuracy(output, target)

        train_rights.append(right)

        if batch_idx % 200 == 0:

            net.eval()
            val_rights = []

            for (data, target) in test_data:
                data = data.to(device)
                target = target.to(device)

                output = net(data)

                right = accuracy(output, target)

                val_rights.append(right)


            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
            train_right = (100. * train_r[0].cpu().numpy() / train_r[1])
            val_right = (100. * val_r[0].cpu().numpy() / val_r[1])


            if val_right > best_test_acc:
                best_test_acc = val_right
                best_epoch = epoch
                evaluate_data(net,test_data,device)
                torch.save(net,os.path.join(config.path,'DD-CNN_best{}epoch:train_acc{:.2f}%_vsl_acc{:.2f}%.pkl'.format(epoch,train_right,best_test_acc)))

            print('Current epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain_Acc: {:.2f}%\tTest_A_cc: {:.2f}%\tbest_test_acc: {:.2f}%\tbest_epoch: {}'.format(
                    epoch, batch_idx * batch_size, len(train_data.dataset),
                           100. * batch_idx / len(train_data),
                    loss.data,
                    train_right,
                    val_right,
                    best_test_acc,
                    best_epoch))

