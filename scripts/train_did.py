"""Standalone training script for the ECAPA-TDNN dialect identification (DID) branch.

Trains ``ECAPA_TDNN_DID`` directly on ViMD log-mel features to classify the
3 regions (North/Central/South) as its own experiment, independent from the
PhoWhisper ASR fine-tuning pipeline in ``run.py``. Tracks Accuracy, macro-F1
and a confusion matrix on the validation split every epoch, and on the test
split at the end logs the same metrics plus a t-SNE projection of the DID
embeddings colored by region.

Usage:
    uv run python scripts/train_did.py --data-dir data/ViMD_Dataset/data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from datasets import DatasetDict
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from dialect_asr import (
    DataCollatorDIDWithPadding,
    ECAPA_TDNN_DID,
    load_vietnamese_processor,
    load_vimd,
    prepare_did_dataset,
    seed_everything,
)


LOGGER = logging.getLogger(__name__)
REGION_NAMES = ["North", "Central", "South"]  # Index == class ID, see data.REGION_TO_LABEL.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ViMD_Dataset/data")
    parser.add_argument("--pretrained-model-name", default="vinai/PhoWhisper-base")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="outputs/did-ecapa-tdnn")
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--embedding-size", type=int, default=192)
    parser.add_argument("--res2net-scale", type=int, default=8)
    parser.add_argument("--se-bottleneck-channels", type=int, default=128)
    parser.add_argument("--attention-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--wandb-project", default="dialect-asr")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="did-ecapa-tdnn")
    parser.add_argument("--wandb-group", default="did")
    parser.add_argument("--wandb-tags", nargs="*", default=["vimd", "did", "ecapa-tdnn"])
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs phải > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("--batch-size/--eval-batch-size phải > 0")
    return args


def build_dataloaders(
    args: argparse.Namespace,
    processor: Any,
) -> dict[str, DataLoader]:
    raw_dataset = load_vimd(args.data_dir, sampling_rate=args.sampling_rate)
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
        "test": args.max_test_samples,
    }

    # Select each split's raw-audio subset *before* preprocessing so
    # --max-*-samples actually skips decoding/mel-extracting the rest of the
    # split instead of preprocessing everything and discarding most of it.
    selected_splits = {}
    for split, limit in limits.items():
        split_dataset = raw_dataset[split]
        if limit is not None:
            if limit <= 0:
                raise ValueError("max_*_samples phải > 0 hoặc None")
            split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))
        selected_splits[split] = split_dataset

    dataset = prepare_did_dataset(
        DatasetDict(selected_splits), processor, num_proc=args.num_proc
    )
    collator = DataCollatorDIDWithPadding(processor.feature_extractor)

    return {
        split: DataLoader(
            dataset[split],
            batch_size=args.batch_size if split == "train" else args.eval_batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        for split in limits
    }


def run_epoch(
    model: ECAPA_TDNN_DID,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Run one train (``optimizer`` set) or eval (``optimizer=None``) epoch."""
    model.train(optimizer is not None)

    total_loss, total_examples = 0.0, 0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_embeddings: list[np.ndarray] = []

    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in dataloader:
            # [B, num_mel_bins, T_frame], [B, T_frame], [B].
            input_features = batch["input_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            region_labels = batch["region_labels"].to(device)

            # [B, num_mel_bins, T_frame] -> [B, T_frame, num_mel_bins] to match
            # ECAPA_TDNN_DID's [B, T_frame, H] contract.
            hidden_states = input_features.transpose(1, 2)
            logits, embedding = model(hidden_states, attention_mask)
            loss = criterion(logits, region_labels)  # [B, R], [B] -> scalar [].

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = region_labels.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            all_labels.extend(region_labels.detach().cpu().tolist())
            all_embeddings.append(embedding.detach().cpu().numpy())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1_macro": f1_score(all_labels, all_predictions, average="macro"),
        "predictions": all_predictions,
        "labels": all_labels,
        "embeddings": np.concatenate(all_embeddings, axis=0),
    }


def tsne_figure(embeddings: np.ndarray, labels: list[int], perplexity: float) -> Figure:
    """Project ``embeddings`` [N, embedding_size] to 2D and color by region."""
    # perplexity must stay below the sample count for TSNE to be well-defined.
    effective_perplexity = min(perplexity, max(len(embeddings) - 1, 1))
    projection = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=42,
    ).fit_transform(embeddings)  # [N, embedding_size] -> [N, 2].

    figure, axis = plt.subplots(figsize=(6, 6))
    labels_array = np.asarray(labels)
    for class_id, region_name in enumerate(REGION_NAMES):
        mask = labels_array == class_id
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            label=region_name,
            alpha=0.7,
            s=12,
        )
    axis.set_title("t-SNE của DID embedding (test split)")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend()
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    seed_everything(args.seed, deterministic=False)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = load_vietnamese_processor(
        args.pretrained_model_name,
        local_files_only=args.local_files_only,
    )
    dataloaders = build_dataloaders(args, processor)

    model = ECAPA_TDNN_DID(
        hidden_size=processor.feature_extractor.feature_size,
        num_regions=len(REGION_NAMES),
        channels=args.channels,
        embedding_size=args.embedding_size,
        res2net_scale=args.res2net_scale,
        se_bottleneck_channels=args.se_bottleneck_channels,
        attention_channels=args.attention_channels,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=vars(args),
    )
    final_checkpoint_path = output_dir / "final_model.pt"
    best_val_f1_macro = -1.0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, dataloaders["train"], device, criterion, optimizer)
            val_metrics = run_epoch(model, dataloaders["validation"], device, criterion)
            LOGGER.info(
                "Epoch %d/%d — train loss %.4f, val loss %.4f, val acc %.4f, val F1-macro %.4f",
                epoch,
                args.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["f1_macro"],
            )
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/f1_macro": train_metrics["f1_macro"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1_macro": val_metrics["f1_macro"],
                    "val/confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=val_metrics["labels"],
                        preds=val_metrics["predictions"],
                        class_names=REGION_NAMES,
                    ),
                }
            )

            # Tracked only for the W&B summary; the checkpoint itself is saved
            # once at the very end, from whatever epoch training finishes on.
            best_val_f1_macro = max(best_val_f1_macro, val_metrics["f1_macro"])

        torch.save(model.state_dict(), final_checkpoint_path)
        LOGGER.info("Saved final checkpoint to %s", final_checkpoint_path)

        test_metrics = run_epoch(model, dataloaders["test"], device, criterion)
        LOGGER.info(
            "Test — loss %.4f, accuracy %.4f, F1-macro %.4f",
            test_metrics["loss"],
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
        )

        figure = tsne_figure(test_metrics["embeddings"], test_metrics["labels"], args.tsne_perplexity)
        run.log(
            {
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/f1_macro": test_metrics["f1_macro"],
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=test_metrics["labels"],
                    preds=test_metrics["predictions"],
                    class_names=REGION_NAMES,
                ),
                "test/tsne_embedding": wandb.Image(figure),
            }
        )
        plt.close(figure)
        run.summary["best_val_f1_macro"] = best_val_f1_macro
    finally:
        run.finish()


if __name__ == "__main__":
    main()
