"""
数据集注册表：名称 -> 配置
"""
from dataclasses import dataclass
from typing import Callable, Optional

from . import loaders


@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    task_type: str  # "token" | "sentence"
    num_labels: int
    default_max_length: int
    load_fn: Callable
    prepare_fn: Optional[Callable] = None  # None 时 sentence 用 prepare_sentence_dataset
    description: str = ""


DATASET_REGISTRY = {
    "msra": DatasetConfig(
        name="msra",
        data_dir="dataset/MSRA命名实体识别数据集",
        task_type="token",
        num_labels=7,
        default_max_length=128,
        load_fn=None,  # 特殊处理，用 data_loader
        description="MSRA NER，中文命名实体识别",
    ),
    "chnsenti": DatasetConfig(
        name="chnsenti",
        data_dir="dataset/ChnSentiCorp",
        task_type="sentence",
        num_labels=2,
        default_max_length=256,
        load_fn=loaders.load_chnsenti,
        description="ChnSentiCorp，中文情感分类",
    ),
    "imdb": DatasetConfig(
        name="imdb",
        data_dir="dataset/imdb",
        task_type="sentence",
        num_labels=2,
        default_max_length=512,
        load_fn=loaders.load_imdb,
        description="IMDB，英文电影评论情感",
    ),
    "sst2": DatasetConfig(
        name="sst2",
        data_dir="dataset/sst2",
        task_type="sentence",
        num_labels=2,
        default_max_length=256,
        load_fn=loaders.load_sst2,
        description="SST-2，英文情感分类",
    ),
    "mrpc": DatasetConfig(
        name="mrpc",
        data_dir="dataset/mrpc",
        task_type="sentence",
        num_labels=2,
        default_max_length=256,
        load_fn=loaders.load_mrpc,
        description="MRPC，句对同义判断",
    ),
    "commonsense_qa": DatasetConfig(
        name="commonsense_qa",
        data_dir="dataset/commonsense_qa",
        task_type="sentence",
        num_labels=5,
        default_max_length=512,
        load_fn=loaders.load_commonsense_qa,
        description="CommonsenseQA，英文常识推理 5 选 1",
    ),
    "csqa": DatasetConfig(
        name="csqa",
        data_dir="dataset/CSQA_preprocessed",
        task_type="sentence",
        num_labels=5,
        default_max_length=512,
        load_fn=loaders.load_csqa,
        description="CSQA 中文常识推理 5 选 1",
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    key = name.lower().strip()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"未知数据集: {name}，可选: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[key]


def list_datasets() -> str:
    lines = ["可用数据集:", "-" * 50]
    for k, v in DATASET_REGISTRY.items():
        lines.append(f"  {k:<15} {v.task_type:<8} {v.num_labels} 类  {v.description}")
    return "\n".join(lines)
