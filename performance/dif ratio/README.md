# `dif ratio/` -- 不同正负样本比例实验（历史实验，保留）

本目录是**历史实验**的保留区，对应稿件中"不同正负样本比例对模型性能的影响"一节（原表 5）。脚本按原样保留，**不再维护**，仅供追溯原始结果。

## 内容

| 文件 | 作用 |
|---|---|
| `operation.py` | 在 HDF5 特征上按 1:1 / 1:2 / 1:3 / 1:4 / 1:5 五种正负比例做 5 折交叉验证（XGBoost，早停），输出各比例的均值与标准差 |
| `ratio fea.py` | 为每种比例构建对应的 HDF5 数据集（`ratio_1_{ratio}.h5`） |
| `cv_results/cv_summary.csv` | 五种比例的汇总：样本量、耗时、AUC / AUPR / Accuracy / F1 / MCC / Precision / Sensitivity / Specificity 的均值与标准差 |
| `cv_results/ratio_1_{1..5}_fold_results.csv` | 每种比例逐折的明细 |

## 保留的结果摘要

| 正负比例 | 总样本 | AUC | AUPR | Accuracy | F1 | MCC |
|---|---|---|---|---|---|---|
| 1:1 | 1,000,000 | 0.9941 | 0.9929 | 0.9661 | 0.9670 | 0.9335 |
| 1:2 | 1,500,000 | 0.9942 | 0.9866 | 0.9632 | 0.9467 | 0.9201 |
| 1:3 | 2,000,000 | 0.9941 | 0.9804 | 0.9656 | 0.9336 | 0.9115 |
| 1:4 | 2,500,000 | 0.9941 | 0.9745 | 0.9679 | 0.9221 | 0.9025 |
| 1:5 | 3,000,000 | 0.9941 | 0.9688 | 0.9695 | 0.9103 | 0.8921 |

随负样本比例升高，AUPRC 与 MCC 单调下降而 AUC 基本不变——这正是稿件用来说明"PR-AUC 的基线随正负比例变化，因此需要按比例分别解读"的那组数据。

## 已知状态

`ratio fea.py` 中的输入路径

```
../dif_classifier/feature_batches/dataset_metadata.json
../dif_classifier/feature_batches/dataset.h5
```

在本仓库中已失效：`dif_classifier` 目录已更名为 `diffierent_classifical`，且 `dataset_metadata.json` 已随分类器对照实验的整理而移除。**因此本目录的脚本无法按原样重新运行**，此处仅保留脚本与已产出的结果；若需重跑，应改用 `../diffierent_classifical/` 下的新特征流水线重新实现。

新协议下的负样本策略比较（uniform / 平均欧氏距离筛选 / DTI 频次匹配）见 `../negative_sampling/`。
