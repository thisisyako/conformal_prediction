import argparse
from pathlib import Path
import torch
import torch.nn as nn

from src.classification.data import split_dataset, data_augmentation
from src.classification.models import load_model
from src.classification.train import train_classifier, train_backbone
from src.classification.conformal import (
    test_conformal_prediction,
    per_class_coverage,
    plot_coverage_histogram,
)


def train(dataset_path, model_path, alpha, seed, device, is_cuda):
    # 1. Split dataset
    train_path, cal_path, test_path = split_dataset(dataset_path, seed=seed)

    # 2. Setup data loaders with augmentation
    train_loader, cal_loader, test_loader = data_augmentation(
        train_path, cal_path, test_path, is_cuda
    )

    # 3. Load model and criterion
    model = load_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 4. Train classifier head (FC)
    print("\nPhase 1: Training classifier head...")
    train_classifier(model, train_loader, criterion, device, is_cuda)

    # 5. Fine-tune backbone
    print("\nPhase 2: Fine-tuning backbone...")
    train_backbone(model, train_loader, criterion, device, is_cuda)

    # 6. Save model weights
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel successfully saved to {model_path}")

    # 7. Quick evaluation post-training
    print("\nEvaluating Conformal Prediction on Test Set...")
    test_conformal_prediction(model, cal_loader, test_loader, alpha=alpha)


def infer(dataset_path, model_path, alpha, seed, device, is_cuda):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model weights not found at {model_path}. Please train first."
        )

    # 1. Split dataset
    train_path, cal_path, test_path = split_dataset(dataset_path, seed=seed)

    # 2. Setup data loaders
    _, cal_loader, test_loader = data_augmentation(
        train_path, cal_path, test_path, is_cuda
    )

    # 3. Load model and weights
    model = load_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 4. Run Conformal Prediction evaluation
    test_conformal_prediction(model, cal_loader, test_loader, alpha=alpha)


def main():
    parser = argparse.ArgumentParser(
        description="Main script for training and inferring RealWaste classification with Conformal Prediction."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "infer", "per_class", "plot_histogram"],
        help="Operation mode: 'train' (trains model), 'infer' (evaluates conformal prediction), 'per_class' (per-class evaluation), 'plot_histogram' (runs multi-seed histogram script).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the root dataset directory containing class subfolders.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/models/mobilenet_v3_realwaste_stage2.pt",
        help="Path to save or load model weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target significance level for conformal prediction (e.g., 0.05 for 95% coverage).",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    dataset_path = Path(args.dataset_path)

    if args.mode == "train":
        print("--- Starting Training Pipeline ---")
        train(dataset_path, args.model_path, args.alpha, args.seed, device, is_cuda)

    elif args.mode == "infer":
        print("--- Starting Inference / Conformal Evaluation Pipeline ---")
        infer(dataset_path, args.model_path, args.alpha, args.seed, device, is_cuda)

    elif args.mode == "per_class":
        print("--- Starting Per-Class Coverage Analysis ---")
        per_class_coverage(dataset_path)

    elif args.mode == "plot_histogram":
        print("--- Generating Multi-Seed Coverage Histogram ---")
        plot_coverage_histogram(
            dataset_path, device, is_cuda, num_seeds=20, target_alpha=args.alpha
        )


if __name__ == "__main__":
    main()
