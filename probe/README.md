# 探针实验

基于 Qwen3-0.6B 与 MSRA NER 的**线性探针**实验，用于分析各层表示对命名实体识别任务的编码能力。

## 核心流程

1. **冻结目标模型**：Qwen3 仅前向传播，不更新参数  
2. **提取层表示**：对每个样本提取指定层的 token 表示（hidden states）  
3. **标签对齐**：MSRA 为字符级标注，通过 `offset_mapping` 对齐到子词 token  
4. **独立训练探针**：每层一个线性分类器，输入 1024 维，输出 7 类 NER 标签  
5. **Z-score 归一化**：训练集统计量归一化，测试集复用，保证层间可比性  

## 环境

```bash
pip install -r requirements.txt
```

## 运行

**完整实验**（28 层 + embedding，耗时较长）：

```bash
python -m probe.main
```

**快速验证**（仅 5 层，约 2000 训练句）：

```bash
python -m probe.main --layers "0,7,14,21,28" --max_train 500 --max_test 200 --epochs 20
```

**常用参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model_path` | Qwen3-0.6B | 模型目录 |
| `--train_file` | dataset/MSRA命名实体识别数据集/train.txt | 训练集 |
| `--test_file` | dataset/MSRA命名实体识别数据集/test.txt | 测试集 |
| `--max_train` | 2000 | 最大训练句子数 |
| `--max_test` | 500 | 最大测试句子数 |
| `--layers` | all | 层索引，如 `5,10,15` 或 `all` |
| `--epochs` | 50 | 探针训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--no_normalize` | - | 关闭 Z-score 归一化 |

## 输出

- 终端：每层训练 loss 与 Train/Test 准确率  
- `probe_results.txt`：汇总结果表格  
