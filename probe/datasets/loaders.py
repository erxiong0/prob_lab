"""
统一数据集加载逻辑，适配 dataset/ 下各类数据
"""
import json
import os
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

# 通用句子级样本（与 ChnSentSample/ImdbSample 等结构一致）
@dataclass
class SentenceSample:
    input_ids: List[int]
    attention_mask: List[int]
    label: int
    text: str


def _load_parquet(
    data_dir: str,
    split: str,
    max_samples: Optional[int],
    subdirs: List[str] = ("data", "plain_text", ""),
    text_col: str = "text",
    label_col: str = "label",
) -> List[Tuple[str, int]]:
    """通用 parquet 加载，返回 [(text, label), ...]"""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需安装 pyarrow 和 pandas: pip install pyarrow pandas")

    for sub in subdirs:
        parquet_dir = os.path.join(data_dir, sub) if sub else data_dir
        if not os.path.isdir(parquet_dir):
            continue
        files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
        target = None
        for f in files:
            if split in f:
                target = os.path.join(parquet_dir, f)
                break
        if target is None and files:
            target = os.path.join(parquet_dir, files[0])
        if target is None:
            continue
        df = pd.read_parquet(target)
        tc = text_col if text_col in df.columns else ("sentence" if "sentence" in df.columns else df.columns[0])
        lc = label_col if label_col in df.columns else df.columns[1]
        out = []
        for _, row in df.iterrows():
            text = str(row[tc]).strip()
            if not text:
                continue
            out.append((text, int(row[lc])))
            if max_samples and len(out) >= max_samples:
                break
        return out
    raise FileNotFoundError(f"未找到 {split} parquet: {data_dir}")


def load_sst2(data_dir: str, split: str, max_samples: Optional[int]) -> List[Tuple[str, int]]:
    """SST-2 情感二分类，列名通常为 sentence + label"""
    return _load_parquet(data_dir, split, max_samples, subdirs=["data", ""], text_col="sentence", label_col="label")


def load_imdb(data_dir: str, split: str, max_samples: Optional[int]) -> List[Tuple[str, int]]:
    """IMDB 情感二分类，数据在 plain_text/"""
    return _load_parquet(data_dir, split, max_samples, subdirs=["plain_text", "data", ""])


def load_chnsenti(data_dir: str, split: str, max_samples: Optional[int]) -> List[Tuple[str, int]]:
    """ChnSentiCorp 中文情感二分类"""
    return _load_parquet(data_dir, split, max_samples, subdirs=["data", ""])


def load_mrpc(data_dir: str, split: str, max_samples: Optional[int]) -> List[Tuple[str, int]]:
    """MRPC 句对同义二分类，拼接 text1 + text2"""
    ext = "jsonl"
    for fname in [f"{split}.{ext}", "train.jsonl"]:
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    t1 = obj.get("text1", obj.get("sentence1", ""))
                    t2 = obj.get("text2", obj.get("sentence2", ""))
                    lab = obj.get("label", -1)
                    if lab < 0:
                        continue
                    text = f"{t1} [SEP] {t2}".strip()
                    if text:
                        out.append((text, int(lab)))
                        if max_samples and len(out) >= max_samples:
                            break
                except Exception:
                    continue
        return out
    raise FileNotFoundError(f"未找到 {split} jsonl: {data_dir}")


def _format_choices(row: Any) -> Tuple[str, int]:
    """CommonsenseQA/CSQA 格式：question + choices -> (text, label)"""
    question = str(row.get("question", "")).strip()
    choices = row.get("choices", {})
    answer_key = str(row.get("answerKey", "A")).strip().upper()
    lab_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    parts = []
    if isinstance(choices, dict):
        labels = choices.get("label", ["A", "B", "C", "D", "E"])
        texts = choices.get("text", [""] * 5)
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        if not isinstance(labels, (list, tuple)):
            labels = [labels] if labels else ["A", "B", "C", "D", "E"]
        if not isinstance(texts, (list, tuple)):
            texts = [texts] if texts else [""] * 5
        for i in range(min(5, len(labels), len(texts))):
            l = str(labels[i]) if i < len(labels) else chr(65 + i)
            t = str(texts[i]) if i < len(texts) else "?"
            parts.append(f"{l}) {t}")
    while len(parts) < 5:
        parts.append(f"{chr(65 + len(parts))}) ?")
    text = f"Question: {question} Choices: {' '.join(parts[:5])}".strip()
    return text, lab_map.get(answer_key[0] if answer_key else "A", 0)


def load_commonsense_qa(data_dir: str, split: str, max_samples: Optional[int]) -> List[Tuple[str, int]]:
    """CommonsenseQA 常识推理 5 选 1"""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需安装 pyarrow 和 pandas")
    for sub in ["data", ""]:
        parquet_dir = os.path.join(data_dir, sub) if sub else data_dir
        if not os.path.isdir(parquet_dir):
            continue
        files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
        target = next((os.path.join(parquet_dir, f) for f in files if split in f), None)
        if target is None and files:
            target = os.path.join(parquet_dir, files[0])
        if target is None:
            continue
        df = pd.read_parquet(target)
        out = []
        for _, row in df.iterrows():
            try:
                text, label = _format_choices(row)
                if text:
                    out.append((text, label))
                    if max_samples and len(out) >= max_samples:
                        break
            except Exception:
                continue
        return out
    raise FileNotFoundError(f"未找到 parquet: {data_dir}")


def load_csqa(data_dir: str, split: str, max_samples: Optional[int]) -> List[Tuple[str, int]]:
    """CSQA_preprocessed 中文常识推理，格式同 CommonsenseQA"""
    return load_commonsense_qa(data_dir, split, max_samples)


def prepare_sentence_dataset(tokenizer, samples: List[Tuple[str, int]], max_length: int) -> List[SentenceSample]:
    """通用句子级样本准备"""
    out = []
    for text, label in samples:
        enc = tokenizer(text, return_tensors=None, truncation=True, max_length=max_length, add_special_tokens=True)
        out.append(SentenceSample(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], label=label, text=text))
    return out
