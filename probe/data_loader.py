"""
MSRA NER 数据加载与预处理
- 解析 MSRA 格式（字符级标注）
- 与 Qwen3 tokenizer 对齐：字符标签 → 子词标签（首个字符决定标签，特殊 token 标 -100）
"""
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

# MSRA NER 标签：O, B-LOC, I-LOC, B-ORG, I-ORG, B-PER, I-PER
LABEL2ID = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-ORG": 3, "I-ORG": 4, "B-PER": 5, "I-PER": 6}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


def load_msra_sentences(file_path: str, max_sentences: Optional[int] = None) -> List[Tuple[str, List[int]]]:
    """
    加载 MSRA 数据集，返回 (句子文本, 字符级标签id列表) 的列表。
    句子以空行分隔。
    """
    sentences = []
    current_chars = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_chars:
                    text = "".join(current_chars)
                    sentences.append((text, current_labels.copy()))
                    current_chars = []
                    current_labels = []
                    if max_sentences and len(sentences) >= max_sentences:
                        break
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue
            char, label = parts[0], parts[1]
            if label not in LABEL2ID:
                label = "O"
            current_chars.append(char)
            current_labels.append(LABEL2ID[label])

        if current_chars and (not max_sentences or len(sentences) < max_sentences):
            text = "".join(current_chars)
            sentences.append((text, current_labels.copy()))

    return sentences


def align_labels_to_tokens(
    char_labels: List[int],
    offset_mapping: List[Tuple[int, int]],
    special_token_ids: set,
) -> List[int]:
    """
    将字符级标签对齐到 token 级。
    - 每个 token 的 offset (start, end)：若 start==0 且 end==0 表示特殊 token，标 -100
    - 否则取该 token 覆盖的首个字符的标签
    """
    token_labels = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            token_labels.append(-100)
        else:
            token_labels.append(char_labels[start])
    return token_labels


@dataclass
class ProbeSample:
    """单个探针样本：已对齐的 token_ids、attention_mask、token 级标签"""
    input_ids: List[int]
    attention_mask: List[int]
    token_labels: List[int]
    text: str


def prepare_probe_dataset(
    tokenizer,
    sentences: List[Tuple[str, List[int]]],
    max_length: int = 128,
) -> List[ProbeSample]:
    """
    将 (文本, 字符标签) 列表转为探针可用的样本。
    使用 return_offsets_mapping 做字符-token 对齐。
    """
    samples = []
    for text, char_labels in sentences:
        if not text or len(char_labels) != len(text):
            continue

        encoding = tokenizer(
            text,
            return_tensors=None,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]

        # 字符级标签，需要扩展到与 offset 对应的位置
        # offset 是 (start, end)，对于 subword 可能 span 多个字符，取 start 位置的标签
        token_labels = []
        for start, end in offset_mapping:
            if start == 0 and end == 0:
                token_labels.append(-100)
            else:
                if start < len(char_labels):
                    token_labels.append(char_labels[start])
                else:
                    token_labels.append(-100)

        samples.append(
            ProbeSample(
                input_ids=input_ids,
                attention_mask=encoding["attention_mask"],
                token_labels=token_labels,
                text=text,
            )
        )
    return samples
