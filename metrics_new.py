from typing import Optional, Dict, Any, Union, List
import numpy as np


from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, roc_auc_score, average_precision_score
)
import numpy as np

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, roc_auc_score, average_precision_score, accuracy_score
)
import numpy as np

def get_eval_metrics(targets, preds, probs_all=None):
    """
    计算多标签分类的评估指标。
    
    参数:
        targets (list or np.ndarray): 真实标签 (二进制向量)。
        preds (list or np.ndarray): 预测标签 (二进制向量)。
        probs_all (list or np.ndarray): 预测概率 (可选，用于计算 ROC-AUC 和 Average Precision)。
    
    返回:
        metrics (dict): 包含各种评估指标的字典。
    """
    targets = np.array(targets)
    preds = np.array(preds)
    
    # 基础指标
    precision = precision_score(targets, preds, average='micro')
    recall = recall_score(targets, preds, average='micro')
    f1 = f1_score(targets, preds, average='micro')
    hamming = hamming_loss(targets, preds)
    jaccard = jaccard_score(targets, preds, average='micro')
    acc = accuracy_score(targets, preds)  # 计算准确率
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'hamming_loss': hamming,
        'jaccard_similarity': jaccard,
        'accuracy': acc,  # 添加准确率
    }
    
    # 如果提供了预测概率，计算 ROC-AUC 和 Average Precision
    if probs_all is not None:
        probs_all = np.array(probs_all)
        try:
            roc_auc = roc_auc_score(targets, probs_all, average='micro')
            avg_precision = average_precision_score(targets, probs_all, average='micro')
            metrics['roc_auc'] = roc_auc
            metrics['average_precision'] = avg_precision
        except ValueError as e:
            print(f"Error calculating ROC-AUC or Average Precision: {e}")
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
    
    # 每个类别的指标
    class_report = {}
    for i in range(targets.shape[1]):  # 遍历每个类别
        class_metrics = {
            'precision': precision_score(targets[:, i], preds[:, i], zero_division=0),
            'recall': recall_score(targets[:, i], preds[:, i], zero_division=0),
            'f1_score': f1_score(targets[:, i], preds[:, i], zero_division=0),
            'accuracy': accuracy_score(targets[:, i], preds[:, i]),  # 每个类别的准确率
        }
        if probs_all is not None:
            try:
                class_metrics['roc_auc'] = roc_auc_score(targets[:, i], probs_all[:, i])
                class_metrics['average_precision'] = average_precision_score(targets[:, i], probs_all[:, i])
            except ValueError as e:
                print(f"Error calculating ROC-AUC or Average Precision for class {i}: {e}")
                class_metrics['roc_auc'] = None
                class_metrics['average_precision'] = None
        class_report[f'class_{i}'] = class_metrics
    
    metrics['report'] = class_report
    return metrics

def print_metrics(eval_metrics: Dict[str, Any]) -> None:
    """
    Print evaluation metrics in a formatted way.

    Args:
        eval_metrics (dict): Dictionary of evaluation metrics to print.
    """
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        print(f"Test {k}: {v:.3f}")
    
    # 打印每个类别的指标
    if "report" in eval_metrics:
        print("\nPer-class metrics:")
        for class_label, metrics in eval_metrics["report"].items():
            print(f"{class_label}: "
                  f"Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, "
                  f"F1-score: {metrics['f1_score']:.3f}, "
                  f"Accuracy: {metrics['accuracy']:.3f}, "
                  f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}, "
                  f"Average Precision: {metrics.get('average_precision', 'N/A')}")