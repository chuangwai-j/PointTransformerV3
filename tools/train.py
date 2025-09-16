# tools/train.py (部分修改)

# 在训练循环中，修改损失函数计算
criterion = torch.nn.CrossEntropyLoss()

# 在验证和测试时，计算二分类指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data_dict in dataloader:
            # 前向传播
            output = model(data_dict)
            pred = output.argmax(dim=1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(data_dict['label'].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }

    return metrics