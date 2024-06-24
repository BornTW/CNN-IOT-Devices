import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.optim.lr_scheduler import LinearLR, StepLR, MultiStepLR
from torch.utils.data import TensorDataset, DataLoader

import utils
from models import CnnNet, CnnLstmNet, CnnGruNet

df = pd.read_csv("./hybrid.csv")
# csv中前1000条数据
x = df.iloc[:, 0:25]
y = df['Label'].astype('category')
y = y.cat.codes
y = y.values
y = np.array(y)
x = np.array(x)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=1)

X_train = torch.from_numpy(X_train).float()
X_train = F.normalize(X_train, p=1, dim=1)
X_test = torch.from_numpy(X_test).float()
X_test = F.normalize(X_test, p=1, dim=1)
Y_train = torch.from_numpy(Y_train)
Y_test = torch.from_numpy(Y_test)


def get_lr(optimizer):  # 从优化器中读取学习率
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(name):
    batch_size = 32
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_size = len(dataset)

    model = CnnGruNet().cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    lr = 0.005

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 35, 45, 55, 65, 75, 80, 85, 90, 95], gamma=0.8)
    epochs = 100

    total_train_step = 0
    best_loss = 10000

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            X_batch = X_batch.reshape(-1, 5, 5)

            pred = model(X_batch)

            optimizer.zero_grad()

            loss = criterion(pred, y_batch.long())
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_train_step += batch_size
            if total_train_step % 10000 == 0:
                print("训练次数：{}，Total_Loss：{}，Current_Loss：{}".format(total_train_step, total_loss, loss.item()))

        # 更新学习率
        scheduler.step()
        lr = get_lr(optimizer)

        message = 'epoch: %d, avg loss: %.5f, lr: %.5f' % (epoch, total_loss / dataset_size * batch_size, lr)
        print(message)
        with open(f"{name}.txt", "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), f"{name}.pth")
            print("模型已保存")

    torch.save(model.state_dict(), "last.pth")
    print("模型已保存")


def test(name):
    batch_size = 32
    dataset = TensorDataset(X_test, Y_test)
    # dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_size = len(dataset)

    model = CnnGruNet().cuda()
    model.load_state_dict(torch.load(f"./{name}.pth"))
    model.eval()

    utils.visual_model(model, [1, 5, 5])

    preds_binarize = None
    reals = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()

        X_batch = X_batch.reshape(-1, 5, 5)

        pred = model(X_batch)
        if preds_binarize is None:
            preds_binarize = pred
        else:
            preds_binarize = torch.cat([preds_binarize, pred], 0)

        # pred = pred.argmax(dim=1)
        # preds.extend(pred.cpu().numpy())
        reals.extend(y_batch.cpu().numpy())

    reals_binarize = label_binarize(reals, classes=[i for i in range(0, 27)])
    preds_binarize = preds_binarize.detach().cpu().numpy()

    preds = [np.argmax(item) for item in preds_binarize]

    accuracy = accuracy_score(preds, reals)
    print(f"Accuracy: {accuracy:.6f}")  # 打印准确率的值，保留两位小数

    cm = confusion_matrix(preds, reals)

    print("label | precision | recall | fscore")
    for label in range(27):
        # 计算每个类别的精确率
        # 精确率是指预测为正例的样本中真正为正例的比例
        pr = utils.precision(label, cm)
        # 计算每个类别的召回率
        # 召回率是指真正为正例的样本中被预测为正例的比例
        rec = utils.recall(label, cm)
        # 计算每个类别的F1分数
        # F1分数是精确率和召回率的调和平均值，用于综合评估模型的性能
        f1 = 2 * (pr * rec / (pr + rec))
        print(f"{label:5d} | {pr:9.4f} | {rec:6.4f} | {f1:6.4f}")

    utils.draw_roc(preds_binarize, reals_binarize)
    utils.visual_cm(cm)


# train("cnn_gru")
test("cnn_gru_66.74")
