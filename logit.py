from openpyxl import load_workbook
import json
import statistics
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_curve, roc_auc_score
)

bad_bids = [5, 7, 20, 34, 46, 54, 65, 76, 88, 99]
for i in range(len(bad_bids)):
    bad_bids[i] -= 1  # 调整为0-based索引

if __name__ == "__main__":
    # 加载原始Excel文件
    workbook = load_workbook(filename='.\\数据\\testdata_mod2.xlsx')
    worksheet = workbook.active

    xm_list, zb_list, tb_list = [], [], []
    connecter = ["projguid", "bid_ann_guid"]

    bad_table = []
    for cell in worksheet['B']:
        if cell.value != None:
            bad_table.append(int(cell.value))
    if bad_table != []:
        bad_bids = []
        for i in range(len(bad_table)):
            if bad_table[i] == 1:
                bad_bids.append(i)  # 转换为0-based索引
    print(f"异常招标索引: {bad_bids}, 共{len(bad_bids)}个")

    # 读取A列数据并处理
    for cell in worksheet['A']:
        if cell.value[0] != "{":
            continue
        dict_str = cell.value
        if dict_str:
            try:
                # 解析字典字符串
                data_dict = json.loads(dict_str)
                # 根据connecter分类
                if connecter[0] in data_dict and connecter[1] not in data_dict:
                    xm_list.append(data_dict)
                elif connecter[1] in data_dict and connecter[0] not in data_dict:
                    tb_list.append(data_dict)
                elif connecter[0] in data_dict and connecter[1] in data_dict:
                    zb_list.append(data_dict)
                else:
                    assert(False)
            except json.JSONDecodeError as e:
                print(f"第{cell.row}行解析失败: {e}")
                continue
    
    vectors = []
    for zb in zb_list:
        bid_ann_guid = zb[connecter[1]]
        tb_group = [tb for tb in tb_list if tb['bid_ann_guid'] == bid_ann_guid]
        tb_group.sort(key=lambda x: x['x_price'])


        # 分离出中标者和落败者的出价
        winning_bids = [bid['x_price'] for bid in tb_group if bid['x_isqualified'] == 1]
        losing_bids = [bid['x_price'] for bid in tb_group if bid['x_isqualified'] == 0]

        # 计算两个最低出价之差Δ
        if len(tb_group) >= 2:
            lowest_bid = tb_group[0]['x_price']
            second_lowest_bid = tb_group[1]['x_price']
            delta = second_lowest_bid - lowest_bid
        else:
            delta = 0  # 出价数量不足时无法计算

        # 计算落败竞价的标准差σ
        if len(losing_bids) >= 2:
            sigma = statistics.stdev(losing_bids)
        elif len(losing_bids) == 1:
            sigma = 0  # 只有一个落败出价时标准差为0
        else:
            sigma = 0

        # 计算比值Δ/σ
        if sigma != 0:
            ratio = delta / sigma
        else:
            ratio = float('inf')  # 如果标准差为0，比值设为无穷大

        # 计算所有投标价格的标准差与算术平均数的比值
        all_bids = [bid['x_price'] for bid in tb_group]
        if len(all_bids) >= 2:
            mean_price = statistics.mean(all_bids)
            std_price = statistics.stdev(all_bids)
            if mean_price != 0:
                std_mean_ratio = std_price / mean_price
            else:
                std_mean_ratio = float('inf')
        else:
            std_mean_ratio = 0  # 出价数量不足时无法计算

        vectors.append([ratio, std_mean_ratio])
        #vectors.append([tb['x_price'] for tb in tb_list if tb['bid_ann_guid'] == bid_ann_guid])
        #vectors.append(winning_bids + losing_bids)
    
    np.random.seed(42)
    vectors = np.array(vectors)

    labels = np.zeros(len(vectors))
    for bid_idx in bad_bids:
        labels[bid_idx] = 1  # 标记异常样本为1
    
    # 划分训练集和测试集
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(
        vectors, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"训练集: {train_vectors.shape[0]}个样本, 测试集: {test_vectors.shape[0]}个样本")
    print(f"训练集异常比例: {train_labels.mean():.2%}, 测试集异常比例: {test_labels.mean():.2%}")

    # 创建并训练logistic回归模型
    logistic_model = LogisticRegression(random_state=42, max_iter=100)
    logistic_model.fit(train_vectors, train_labels)

    # 在测试集上进行预测
    test_predictions = logistic_model.predict(test_vectors)
    test_probs = logistic_model.predict_proba(test_vectors)[:, 1]  # 获取异常类的概率
    
    # 评估模型性能
    print("\n--- 模型评估结果 ---")
    print(f"准确率: {accuracy_score(test_labels, test_predictions):.4f}")
    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, test_predictions))
    print("\n分类报告:")
    print(classification_report(test_labels, test_predictions))

    # 对所有向量进行预测
    all_predictions = logistic_model.predict(vectors)
    all_probs = logistic_model.predict_proba(vectors)[:, 1]  # 获取异常类的概率

    # 输出预测结果
    print("\n--- 所有向量预测结果 ---")
    print(f"原始异常向量下标: {bad_bids}")
    print(f"预测为异常的向量下标: {np.where(all_predictions == 1)[0]}")
    
    # 输出每个向量的置信度分数
    print("\n--- 每个向量的异常置信度分数 ---")
    for i, prob in enumerate(all_probs):
        print(f"向量{i}: 异常置信度 = {prob:.6f}, 预测结果 = {'异常' if all_predictions[i] == 1 else '正常'}")

    # 计算模型在所有数据上的表现
    print(f"\n模型在所有数据上的准确率: {accuracy_score(labels, all_predictions):.4f}, F1-score: {f1_score(labels, all_predictions):.4f}")
    
    # 绘制ROC曲线
    print("\n--- 绘制ROC曲线 ---")
    # 计算ROC曲线的fpr, tpr和阈值
    fpr, tpr, thresholds = roc_curve(labels, all_probs)
    # 计算AUC值
    auc = roc_auc_score(labels, all_probs)
    
    # 计算不同阈值下的准确率
    accuracies = []
    for threshold in thresholds:
        y_pred = (all_probs >= threshold).astype(int)
        accuracies.append(accuracy_score(labels, y_pred))
    
    # 找到准确率最高的阈值
    max_accuracy_idx = np.argmax(accuracies)
    max_accuracy = accuracies[max_accuracy_idx]
    best_threshold = thresholds[max_accuracy_idx]
    best_fpr = fpr[max_accuracy_idx]
    best_tpr = tpr[max_accuracy_idx]
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # 随机猜测线
    
    # 标出最优准确率点
    plt.scatter(best_fpr, best_tpr, color='red', s=100, 
                label=f'最优准确率点 (Accuracy = {max_accuracy:.4f}, Threshold = {best_threshold:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('围标串标识别模型的ROC曲线')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.show()
    
    print(f"最优准确率: {max_accuracy:.4f}, 对应的阈值: {best_threshold:.4f}")