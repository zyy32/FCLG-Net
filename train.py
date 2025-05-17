import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from src.cnn import CNN
from utils.dataloader import Datases_loader as dataloader
from utils.dataloader1 import Datases_loader as dataloader1
from utils.loss import Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchsz = 2
lr = 0.0001
items = 200

model = CNN().to(device)

criterion = Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=1)
savedir = r'/T2020027/ayyz2'

train_imgpath = r'/T2020027/ayyz2/data/760/train/img'
train_labpath = r'/T2020027/ayyz2/data/760/train/labelcol'

imgsz = 256

# 加载训练集
train_dataset = dataloader(train_imgpath, train_labpath, imgsz, imgsz)
train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)


lossx = 0
tp, tn, fp, fn = 0, 0, 0, 0
accuracy, precision, recall, F1, IoU, ls_loss = [], [], [], [], [], []
validation_results = []  # 保存每一轮验证结果

# 初始化用于存储最佳验证指标的变量
best_F1 = 0
best_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "F1": 0, "IoU": 0}  # 存储最好的验证集指标

# 验证函数（不包含损失计算）
def validate():
    tp, fp, tn, fn = 0, 0, 0, 0
    for samples in test_loader:
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)
        pred ,y5,y4,y3,y2= model(img)

        p = pred.reshape(-1)
        p[p >= 0.] = 1
        p[p < 0.] = 0
        t = lab.reshape(-1)

        tp_, fp_, tn_, fn_ = compute_confusion_matrix(p.detach().cpu().numpy(), t.detach().cpu().numpy())
        tp += tp_
        fp += fp_
        tn += tn_
        fn += fn_

    accuracy_, precision_, recall_, F1_, IoU_ = compute_indexes(tp, fp, tn, fn)

    return accuracy_, precision_, recall_, F1_, IoU_

# 训练函数
def train():
    global best_F1, best_metrics
    print('总共的epoch', items)
    
    for epoch in range(items):
        print(f'Epoch {epoch + 1}/{items}')
        model.train()
        lossx = 0
        tp, tn, fp, fn = 0, 0, 0, 0

        # 训练循环
        for idx, samples in enumerate(train_loader):
            img, lab = samples['image'], samples['mask']
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()
            pred ,y5, y4, y3, y2 = model(img)

            loss = criterion(pred, lab, y5, y4, y3, y2)

            loss.backward()
            optimizer.step()

            lossx += loss.item()

            p = pred.reshape(-1)
            p[p >= 0.] = 1
            p[p < 0.] = 0
            t = lab.reshape(-1)

            tp_, fp_, tn_, fn_ = compute_confusion_matrix(p.detach().cpu().numpy(), t.detach().cpu().numpy())
            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_

        accuracy_, precision_, recall_, F1_, IoU_ = compute_indexes(tp, fp, tn, fn)
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
        F1.append(F1_)
        IoU.append(IoU_)

        scheduler.step()
        avg_loss = lossx / len(train_loader)
        ls_loss.append(avg_loss)

        print(f'Training Loss: {avg_loss:.4f}, Accuracy: {accuracy_:.4f}, Precision: {precision_:.4f}, Recall: {recall_:.4f}, F1: {F1_:.4f}, IoU: {IoU_:.4f}')

        # 进行验证
        val_accuracy, val_precision, val_recall, val_F1, val_IoU = validate()
        print(f'Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_F1:.4f}, IoU: {val_IoU:.4f}')

        # 保存验证结果
        validation_results.append(f'Epoch {epoch + 1}: Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_F1:.4f}, IoU: {val_IoU:.4f}\n')

        # 如果当前 F1 分数大于历史最佳 F1 分数，则保存模型权重并更新最佳指标
        if val_F1 > best_F1:
            best_F1 = val_F1
            best_metrics = {
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "F1": val_F1,
                "IoU": val_IoU
            }
            savedir_with_epoch = f'{savedir}_epoch_{epoch + 1}_F1_{val_F1:.4f}.pth'
            torch.save(model.state_dict(), savedir_with_epoch)  # 保存模型权重
            print(f'Best F1 improved to {best_F1:.4f}, saving model to {savedir_with_epoch}.')

        # 输出当前最好的验证指标
        print(f'Best validation results so far - Accuracy: {best_metrics["accuracy"]:.4f}, '
              f'Precision: {best_metrics["precision"]:.4f}, Recall: {best_metrics["recall"]:.4f}, '
              f'F1: {best_metrics["F1"]:.4f}, IoU: {best_metrics["IoU"]:.4f}')

if __name__ == '__main__':
    train()
    # 保存结果
    result_str = (f'accuracy: {accuracy}\n'
                  f'precision: {precision}\n'
                  f'recall: {recall}\n'
                  f'F1: {F1}\n'
                  f'IoU: {IoU}\n'
                  f'loss: {ls_loss}\n'
                  f'Validation Results:\n' + ''.join(validation_results))  # 添加验证结果

    filename = r'/T2020027/ayyz2/results.txt'

    with open(filename, mode='w', newline='') as f:
        f.writelines(result_str)
