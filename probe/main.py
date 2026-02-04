"""
探针实验主脚本
- 使用 Qwen3-0.6B + MSRA NER
- 对指定层（默认全部 28 层）独立训练线性探针
- 输出每层训练/测试准确率
"""
import argparse
import os
import sys

import torch

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.data_loader import (
    load_msra_sentences,
    prepare_probe_dataset,
    NUM_LABELS,
)
from probe.probe_model import LinearProbe
from probe.train import (
    extract_and_stack,
    train_probe,
    evaluate_probe,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen3-0.6B")
    parser.add_argument("--train_file", type=str, default="dataset/MSRA命名实体识别数据集/train.txt")
    parser.add_argument("--test_file", type=str, default="dataset/MSRA命名实体识别数据集/test.txt")
    parser.add_argument("--max_train", type=int, default=2000, help="最大训练句子数（轻量化）")
    parser.add_argument("--max_test", type=int, default=500)
    parser.add_argument("--layers", type=str, default="all", help="要训练的层，如 '5,10,15' 或 'all'")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--normalize", action="store_true", default=True, help="对表示做 Z-score 归一化")
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    train_file = os.path.join(root, args.train_file) if not os.path.isabs(args.train_file) else args.train_file
    test_file = os.path.join(root, args.test_file) if not os.path.isabs(args.test_file) else args.test_file

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 1. 加载模型和分词器
    print("Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 探针用 float32 即可，避免与 bf16 混用
        device_map=None,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    print(f"Model: {num_layers} layers, hidden_size={hidden_size}")

    # 2. 加载数据
    print("Loading MSRA data...")
    train_sentences = load_msra_sentences(train_file, max_sentences=args.max_train)
    test_sentences = load_msra_sentences(test_file, max_sentences=args.max_test)
    print(f"Train: {len(train_sentences)} sentences, Test: {len(test_sentences)} sentences")

    train_samples = prepare_probe_dataset(tokenizer, train_sentences, max_length=args.max_length)
    test_samples = prepare_probe_dataset(tokenizer, test_sentences, max_length=args.max_length)
    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

    # 3. 确定要训练的层
    if args.layers == "all":
        layer_indices = list(range(num_layers + 1))  # 0=embedding, 1~N=transformer
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    # 4. 逐层训练与评估
    results = []
    for layer_idx in layer_indices:
        layer_name = "embedding" if layer_idx == 0 else f"layer_{layer_idx}"
        print(f"\n{'='*50} {layer_name} {'='*50}")

        # 提取表示（训练集计算归一化统计量，测试集复用）
        train_reprs, train_labels, norm_stats = extract_and_stack(
            model, train_samples, layer_idx, device,
            normalize=args.normalize, pad_token_id=pad_token_id
        )
        norm_mean = norm_stats[0].to(device) if norm_stats else None
        norm_std = norm_stats[1].to(device) if norm_stats else None
        test_reprs, test_labels, _ = extract_and_stack(
            model, test_samples, layer_idx, device,
            normalize=args.normalize, pad_token_id=pad_token_id,
            norm_mean=norm_mean, norm_std=norm_std
        )

        n_train = (train_labels != -100).sum().item()
        n_test = (test_labels != -100).sum().item()
        print(f"  Train tokens: {n_train}, Test tokens: {n_test}")

        if n_train == 0:
            print(f"  Skip (no valid labels)")
            results.append((layer_name, 0.0, 0.0))
            continue

        # 初始化并训练探针
        probe = LinearProbe(input_dim=hidden_size, num_labels=NUM_LABELS).to(device)
        train_probe(probe, train_reprs, train_labels, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

        train_acc = evaluate_probe(probe, train_reprs, train_labels, device)
        test_acc = evaluate_probe(probe, test_reprs, test_labels, device)
        print(f"  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        results.append((layer_name, train_acc, test_acc))

    # 5. 汇总
    print("\n" + "=" * 60)
    print("Results summary:")
    print(f"{'Layer':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 40)
    for name, ta, tea in results:
        print(f"{name:<15} {ta:<12.4f} {tea:<12.4f}")

    # 可选：保存结果
    out_path = os.path.join(root, "probe_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Layer\tTrain_Acc\tTest_Acc\n")
        for name, ta, tea in results:
            f.write(f"{name}\t{ta:.4f}\t{tea:.4f}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
