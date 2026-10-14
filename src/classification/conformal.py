"""
L'objectif est d'encadrer les prédictions d'un modèle de classification d'images
de déchets (9 classes) avec une garantie statistique de couverture (ex: 95%).
On prédit ainsi un ensemble de classes (prediction set) dont la taille varie selon l'incertitude du modèle.
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.classification.data import split_dataset, data_augmentation
from src.classification.models import load_model
from src.classification.train import train_classifier, train_backbone

NUM_CLASSES = 9


CLASS_LABELS = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Miscellaneous Trash",
    "Textile Trash",
    "Vegetation",
]

DATASET_NAME = "realwaste"


def score(model, X, y):
    """
    Calcule le score de non-conformité (score d'étrangeté) pour un batch d'images.

    - On trie les classes par probabilité décroissante.
    - On cherche la position de la vraie classe (y[i]) dans ce tri.
    - Le score est la somme des probabilités des classes qui sont plus probables que la vraie classe. Si le modèle a vu juste (position 0), le score est 0.
    """
    with torch.no_grad():
        outputs = model(X)
        probs = nn.functional.softmax(outputs, dim=1).numpy()
    scores = []
    for i, p in enumerate(probs):
        order = np.argsort(p)[::-1]
        true_pos = np.where(order.copy() == y[i])[0][0]
        scores.append(p[order[:true_pos]].sum())
    return scores


def calibration(model, cal_loader, alpha):
    """
    Calcule le seuil de quantile q_hat sur le jeu de calibration.

    - On applique la correction d'échantillon fini pour garantir le taux de couverture (1 - alpha), par exemple 95% si alpha = 0.05.
    """
    all_scores = []
    for images, labels in cal_loader:
        all_scores += score(model, images, labels)
    all_scores = np.array(all_scores)
    n = len(all_scores)
    q_hat = np.quantile(all_scores, np.ceil((n + 1) * (1 - alpha)) / n)
    return q_hat


def test_conformal_prediction(model, cal_loader, test_loader, alpha):
    """
    Évalue la prédiction conforme sur le jeu de test en utilisant le q_hat calibré.
    """

    model.eval()
    q_hat = calibration(model, cal_loader, alpha)

    total_predicted = 0
    coverage_correct = 0
    set_sizes = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs_batch = nn.functional.softmax(outputs, dim=1).numpy()
            for i in range(len(images)):
                probs = probs_batch[i]

                pred_set = []
                order = np.argsort(probs)[::-1]
                cum = 0.0

                for idx in order:
                    if cum > q_hat:
                        break
                    pred_set.append(idx)
                    cum += probs[idx]

                true_label = labels[i]

                total_predicted += 1
                set_sizes.append(len(pred_set))
                if true_label in pred_set:
                    coverage_correct += 1
    set_sizes = np.array(set_sizes)
    final_coverage = 100.0 * coverage_correct / total_predicted
    average_set_size = np.mean(set_sizes)

    print(f"Final Coverage: {final_coverage:.2f}%")
    print(f"Mean Set Size: {average_set_size:.2f}")
    print(f"Median Set Size: {np.median(set_sizes):.2f}")
    print(f"Min Set Size: {np.min(set_sizes):.2f}")
    print(f"Max Set Size: {np.max(set_sizes):.2f}")
    print(f"Quantile q_hat: {q_hat:.4f}")
    return final_coverage, average_set_size, q_hat


def plot_coverage_histogram(
    dataset_path, device, is_cuda, num_seeds=20, target_alpha=0.05
):
    """
    Répète l'entrainement et la prédiction conforme sur plusieurs splits aléatoires (multi-seed) pour vérifier la stabilité empirique du coverage.
    """

    seeds = range(num_seeds)
    coverages = []
    avg_set_sizes = []
    q_hats = []

    for seed in seeds:
        train_path, cal_path, test_path = split_dataset(dataset_path, seed=seed)
        train_loader, cal_loader, test_loader = data_augmentation(
            train_path, cal_path, test_path, is_cuda
        )
        model = load_model(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        train_classifier(model, train_loader, criterion, device, is_cuda)
        train_backbone(model, train_loader, criterion, device, is_cuda)
        coverage, avg_set_size, q_hat = test_conformal_prediction(
            model, cal_loader, test_loader, target_alpha
        )
        coverages.append(coverage)
        avg_set_sizes.append(avg_set_size)
        q_hats.append(q_hat)
    print(coverages)
    print(avg_set_sizes)
    print(q_hats)

    plt.figure(figsize=(9, 5))
    plt.hist(coverages, bins=12, color="royalblue", edgecolor="black", alpha=0.85)

    # Ligne verticale indiquant la couverture cible théorique
    plt.axvline(
        x=100.0 * (1 - target_alpha),
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Couverture cible (95%)",
    )

    plt.title("Histogramme du taux de couverture sur 20 splits aléatoires")
    plt.xlabel("Taux de couverture observé")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde et affichage
    plt.savefig("histogram_coverage.png", dpi=300)
    plt.show()
    print("Histogramme généré et sauvegardé avec succès !")


def evaluate_per_class_coverage(model, test_loader, q_hat, n_labels, device):
    """
    Analyse par classe : calcule la couverture, la taille moyenne des sets, et identifie les confusions (top hesitated classes).

    - Permet de voir si certaines catégories de déchets (ex: plastique vs métal) posent plus de problèmes d'incertidude au modèle.
    """
    model.eval()

    class_total = defaultdict(int)
    class_coverage = defaultdict(int)
    class_set_size = defaultdict(list)

    class_hesitations = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs_batch = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(images)):
                probs = probs_batch[i]
                true_label = labels_np[i]

                order = np.argsort(probs)[::-1]
                pred_set = []
                cum = 0.0

                for idx in order:
                    if cum > q_hat:
                        break
                    pred_set.append(idx)
                    cum += probs[idx]

                class_total[true_label] += 1
                class_set_size[true_label].append(len(pred_set))
                if true_label in pred_set:
                    class_coverage[true_label] += 1

                for pred_idx in pred_set:
                    if pred_idx != true_label:
                        class_hesitations[true_label][pred_idx] += 1

    results = {}
    header_title = "Class Name"
    print(
        f"{header_title:<20} | {'Coverage (%)':<15} | {'Avg Set Size':<15} | {'Samples':<8} | {'Top 2 Hesitated Classes':<30}"
    )
    print("-" * 104)

    for cls in range(n_labels):
        total = class_total[cls]
        c_label = CLASS_LABELS[cls]

        if total > 0:
            cov = 100.0 * class_coverage[cls] / total
            avg_size = np.mean(class_set_size[cls])

            sorted_hesitations = sorted(
                class_hesitations[cls].items(), key=lambda x: x[1], reverse=True
            )[:2]

            top_hesitated_formatted = []
            for h_cls, count in sorted_hesitations:
                h_name = CLASS_LABELS[h_cls]
                top_hesitated_formatted.append(f"{h_name} ({count})")

            top_hesitated_str = (
                ", ".join(top_hesitated_formatted)
                if top_hesitated_formatted
                else "None"
            )

            results[cls] = {
                "coverage": cov,
                "avg_set_size": avg_size,
                "count": total,
                "top_hesitated": sorted_hesitations,
            }
            print(
                f"{c_label:<20} | {cov:<15.2f} | {avg_size:<15.2f} | {total:<8} | {top_hesitated_str:<30}"
            )
        else:
            results[cls] = {
                "coverage": 0.0,
                "avg_set_size": 0.0,
                "count": 0,
                "top_hesitated": [],
            }
            print(f"{c_label:<20} | {'N/A':<15} | {'N/A':<15} | {0:<8} | {'N/A':<30}")

    return results


def per_class_coverage(dataset_path, model_path, target_alpha, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    train_path, cal_path, test_path = split_dataset(dataset_path, seed=seed)

    _, cal_loader, test_loader = data_augmentation(
        train_path, cal_path, test_path, is_cuda
    )

    model = load_model(device)
    model.load_state_dict(torch.load(model_path))

    _, _, q_hat = test_conformal_prediction(
        model, cal_loader, test_loader, target_alpha
    )

    evaluate_per_class_coverage(
        model=model,
        test_loader=test_loader,
        q_hat=q_hat,
        n_labels=9,
        device=device,
    )
