import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import config
import torch
import numpy as np

def accuracy(predictions, labels):
    pred = torch.max(predictions, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

def evaluate_data(net,data,device):
    net.eval()
    val_rights = []
    outputs = []
    targets = []
    with torch.no_grad():

        for (data, target) in data:
            data = data.to(device)
            target = target.to(device)
            targets.append(target.cpu().numpy())

            output = net(data)
            pred = torch.max(output, 1)[1]

            outputs.append(pred.cpu().numpy())

            right = accuracy(output, target)
            val_rights.append(right)

    val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
    label = config.labels

    list_targets = [y for x in targets for y in x]
    list_outputs = [y for x in outputs for y in x]

    sns.set()
    f, ax = plt.subplots()

    C2 = confusion_matrix(np.array(list_targets), np.array(list_outputs))
    for i in range(len(label)):
        print(label[i], '{:.2f}%'.format(C2[i][i] / np.sum(C2[i]) * 100))
    print('Average accuracy{:.2f}%'.format(100 * val_r[0].cpu().numpy() / val_r[1]))
    print(C2)
    sns.heatmap(C2, annot=True, ax=ax, fmt='.20g')

    ax.set_title('Confusion matrix')  #
    ax.set_xlabel('Predict\nIndoor, Outdoor, Transport')
    ax.set_ylabel('True\nIndoor, Outdoor, Transport')
    plt.savefig('confusion matrix.pdf')