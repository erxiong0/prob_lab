# 探针实验

基于 Qwen3-0.6B 的**线性探针**实验，统一入口适配 `dataset/` 下各类数据。

## 支持数据集

| 名称 | 类型 | 说明 |
|------|------|------|
| msra | token | MSRA NER，中文命名实体识别 |
| chnsenti | sentence | ChnSentiCorp，中文情感 |
| imdb | sentence | IMDB，英文电影评论情感 |
| sst2 | sentence | SST-2，英文情感 |
| mrpc | sentence | MRPC，句对同义判断 |
| commonsense_qa | sentence | CommonsenseQA，英文常识推理 5 选 1 |
| csqa | sentence | CSQA，中文常识推理 5 选 1 |

## 统一运行（推荐）

```bash
# 列出可用数据集
python -m probe.run --list

# 指定数据集运行
python -m probe.run --dataset msra
python -m probe.run --dataset chnsenti
python -m probe.run --dataset imdb
python -m probe.run --dataset sst2
python -m probe.run --dataset mrpc
python -m probe.run --dataset commonsense_qa
python -m probe.run --dataset csqa

# 快速验证
python -m probe.run -d imdb --layers "5" --max_train 500 --epochs 10
```

## 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `-d` / `--dataset` | 必填 | 数据集名称 |
| `--data_dir` | 自动 | 覆盖数据目录 |
| `--max_train` | 2000 | 最大训练样本数 |
| `--max_test` | 500 | 最大测试样本数 |
| `--layers` | all | 层索引，如 `5,10,15` |
| `--extract_batch_size` | 16 | 提取表示时的 batch，显存不足可调小 |
| `--device` | 自动 | 指定 CUDA 卡，如 cuda:0 |

## 输出

结果保存为 `probe_results_{数据集名}.txt`，可用 `-o` 指定路径。
