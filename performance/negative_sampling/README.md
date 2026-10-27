# 欧几里得距离 / RMS 负样本采样消融实验

本目录用于进行单冷启动场景下的负样本采样消融实验，与原有的 `negative_sample` 实验相互独立。实验只读取并复用原冷启动数据，不会覆盖已有的 S1、S2、S3 数据或历史训练结果。

## 两种负样本距离

对于一个药物对，分别计算其各个独立 Target 组合的欧几里得距离，再使用以下两种方式聚合：

- `mean_euclidean`：所有 Target 组合欧几里得距离的加权平均值；
- `rms`：所有 Target 组合欧几里得距离的加权均方根。

这里不会对 Target 特征进行平均池化。两种方法也不是简单地在同一个固定维度向量上计算欧几里得距离和 RMS，避免二者只相差一个常数而失去对比意义。

## 实验规模

- 距离阈值：`0.9`、`1.0`、`1.1` 倍 AED；
- 随机种子：`42`、`52`、`62`、`72`、`82`；
- 每组训练数据保持正负样本 `1:1`；
- 共计：2 种方法 x 3 个阈值 x 5 个种子 = 30 组训练。

## GPU 精简模式

为降低显存、内存和磁盘缓存压力，当前训练使用以下配置：

- 使用 GPU：`device=cuda`；
- 每个药物对最多保留 16 个独立 Target 组合；
- 超过 16 个组合时，使用固定且可复现的哈希规则抽样；
- 不进行 Target 平均池化；
- 抽样后重新归一化权重，使每个药物对的组合权重之和为 1；
- `training_batch_size=128`；
- `max_quantile_batches=4`；
- `max_bin=64`；
- `nthread=2`。

GPU 精简结果保存到 `output/results_gpu_lite/`，不会与旧版完整输入结果混合。

## 使用方法

在 PowerShell 中进入本目录，并设置 Python 路径：

```powershell
Set-Location "D:\700_Project\PythonProject\euclidean_rms_ablation"
$py="D:\600_Dev_soft\670_pyCharm\conda\envs\cold-start\python.exe"
```

查看状态和执行数据审计：

```powershell
& $py ".\aed_rms_experiment.py" status
& $py ".\aed_rms_experiment.py" audit
```

审计会计算约 228 万个训练候选药物对的距离，只使用 CPU 和内存，不会训练模型。确认审计结果中 `audit_passed: true` 后，准备实验数据：

```powershell
& $py ".\aed_rms_experiment.py" prepare
```

正式训练示例：

```powershell
& $py ".\aed_rms_experiment.py" train --method mean_euclidean --threshold 1.1 --seed 42
& $py ".\aed_rms_experiment.py" train --method rms --threshold 1.1 --seed 42
```

每次只能运行一条训练命令，不要并行训练。若模型文件已经存在，脚本会停止并提示；只有明确需要覆盖该组结果时才使用 `--force`。

全部 30 组训练完成后汇总结果：

```powershell
& $py ".\aed_rms_experiment.py" summarize
```

汇总报告保存在 `output/reports_gpu_lite/`，包含 ACC、SEN、Specificity、Precision、F1、MCC、ROC-AUC 和 AUPR 等指标。

## 注意事项

- 旧版完整输入结果保存在 `output/results/`，不得与 GPU 精简结果直接混合比较；
- 公平比较时，所有方法、阈值和随机种子都必须使用同一套 GPU 精简配置重新训练；
- GPU 精简模式只能降低系统负载，不能完全排除显卡驱动异常；
- 如果再次出现 `DPC_WATCHDOG_VIOLATION (0x133)` 蓝屏，应停止在本机使用 GPU 训练，改用 CPU 或其他计算设备。
