"""
统一探针实验入口：适配 dataset/ 下各类数据
用法：python -m probe.run --dataset <名称> [其他参数]
"""
import argparse
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

try:
    from .datasets import get_dataset_config, list_datasets
except ImportError:
    from probe.datasets import get_dataset_config, list_datasets


def main():
    parser = argparse.ArgumentParser(description="探针实验 - 统一入口")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="数据集名称，用 --list 查看")
    parser.add_argument("--model_path", type=str, default="Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default=None, help="覆盖默认数据目录")
    parser.add_argument("--max_train", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=500)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--extract_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")
    parser.add_argument("--output", "-o", type=str, default=None, help="结果输出文件")
    parser.add_argument("--list", action="store_true", help="列出可用数据集后退出")
    args = parser.parse_args()

    if args.list:
        print(list_datasets())
        return

    if not args.dataset:
        parser.error("请指定 --dataset，用 --list 查看可用数据集")

    cfg = get_dataset_config(args.dataset)
    data_dir = args.data_dir or os.path.join(_root, cfg.data_dir)
    max_length = args.max_length if args.max_length is not None else cfg.default_max_length

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Dataset: {cfg.name} ({cfg.description})")
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.join(_root, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map=None, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_labels = cfg.num_labels

    if cfg.task_type == "token":
        # MSRA NER
        from .data_loader import load_msra_sentences, prepare_probe_dataset

        train_file = os.path.join(data_dir, "train.txt")
        test_file = os.path.join(data_dir, "test.txt")
        if not os.path.isfile(train_file):
            raise FileNotFoundError(f"未找到 {train_file}")
        train_sentences = load_msra_sentences(train_file, max_sentences=args.max_train)
        test_sentences = load_msra_sentences(test_file, max_sentences=args.max_test)
        train_samples = prepare_probe_dataset(tokenizer, train_sentences, max_length=max_length)
        test_samples = prepare_probe_dataset(tokenizer, test_sentences, max_length=max_length)
        from .train import extract_and_stack, train_probe, evaluate_probe
        extract_fn = extract_and_stack
    else:
        # 句子级
        from .datasets.loaders import prepare_sentence_dataset

        train_raw = cfg.load_fn(data_dir, "train", args.max_train)
        test_raw = cfg.load_fn(data_dir, "test", args.max_test)
        if len(test_raw) == 0:
            test_raw = cfg.load_fn(data_dir, "validation", args.max_test)
        train_samples = prepare_sentence_dataset(tokenizer, train_raw, max_length)
        test_samples = prepare_sentence_dataset(tokenizer, test_raw, max_length)
        from .train import extract_and_stack_sentence, train_probe, evaluate_probe
        extract_fn = extract_and_stack_sentence

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    layer_indices = list(range(num_layers + 1)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    results = []

    for layer_idx in layer_indices:
        layer_name = "embedding" if layer_idx == 0 else f"layer_{layer_idx}"
        print(f"\n{'='*50} {layer_name} {'='*50}")

        train_reprs, train_labels, norm_stats = extract_fn(
            model, train_samples, layer_idx, device,
            normalize=args.normalize, pad_token_id=pad_token_id,
            batch_size=args.extract_batch_size,
        )
        norm_mean = norm_stats[0].to(device) if norm_stats else None
        norm_std = norm_stats[1].to(device) if norm_stats else None
        test_reprs, test_labels, _ = extract_fn(
            model, test_samples, layer_idx, device,
            normalize=args.normalize, pad_token_id=pad_token_id,
            norm_mean=norm_mean, norm_std=norm_std,
            batch_size=args.extract_batch_size,
        )

        n_train = (train_labels != -100).sum().item() if cfg.task_type == "token" else len(train_labels)
        n_test = (test_labels != -100).sum().item() if cfg.task_type == "token" else len(test_labels)
        print(f"  Train: {n_train}, Test: {n_test}")

        if n_train == 0:
            results.append((layer_name, 0.0, 0.0))
            continue

        from .probe_model import LinearProbe
        probe = LinearProbe(input_dim=hidden_size, num_labels=num_labels).to(device)
        train_probe(probe, train_reprs, train_labels, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

        train_acc = evaluate_probe(probe, train_reprs, train_labels, device)
        test_acc = evaluate_probe(probe, test_reprs, test_labels, device)
        print(f"  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        results.append((layer_name, train_acc, test_acc))

    print("\n" + "=" * 60)
    print(f"Results ({cfg.name}):")
    print(f"{'Layer':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 40)
    for name, ta, tea in results:
        print(f"{name:<15} {ta:<12.4f} {tea:<12.4f}")

    out_path = args.output or os.path.join(_root, f"probe_results_{cfg.name}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Layer\tTrain_Acc\tTest_Acc\n")
        for name, ta, tea in results:
            f.write(f"{name}\t{ta:.4f}\t{tea:.4f}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
