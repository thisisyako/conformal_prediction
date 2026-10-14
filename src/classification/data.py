import os
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision import datasets, transforms

# MobileNetV3 attend du 224x224 par défaut
IMG_SIZE = 224
BATCH_SIZE = 64

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


CAL_PROP = 0.2
TEST_PROP = 0.1


def split_dataset(dataset_path, seed):

    parent_dir = dataset_path.resolve().parent
    split_path = parent_dir / "split"
    train_path = split_path / "train"
    cal_path = split_path / "cal"
    test_path = split_path / "test"

    if split_path.exists():
        shutil.rmtree(split_path)
    split_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------- Helper -------------------------------- #
    def ensure_dirs(base_path, classes):
        for cls in classes:
            (base_path / cls).mkdir(parents=True, exist_ok=True)

    # ---------------------------- Split -------------------------------- #
    ensure_dirs(train_path, CLASS_LABELS)
    ensure_dirs(test_path, CLASS_LABELS)
    ensure_dirs(cal_path, CLASS_LABELS)

    print("Découpage du dataset en train / val / test / cal")

    for cls in CLASS_LABELS:
        src_dir = dataset_path / cls
        files = [f for f in os.listdir(src_dir) if (src_dir / f).is_file()]

        # 1. Séparation Train vs (Cal + Test)
        train_files, temp_files = train_test_split(
            files,
            test_size=(TEST_PROP + CAL_PROP),
            random_state=seed,
            shuffle=True,
        )

        # 2. Séparation Cal vs Test
        relative_test = TEST_PROP / (TEST_PROP + CAL_PROP)
        cal_files, test_files = train_test_split(
            temp_files, test_size=relative_test, random_state=seed, shuffle=True
        )

        # 3. Copie simple des fichiers (sans retouche ni augmentation)
        for f in train_files:
            shutil.copy(src_dir / f, train_path / cls / f)

        for f in test_files:
            shutil.copy(src_dir / f, test_path / cls / f)

        for f in cal_files:
            shutil.copy(src_dir / f, cal_path / cls / f)

    return train_path, cal_path, test_path


def data_augmentation(train_path, cal_path, test_path, is_cuda):
    # Augmentation dynamique à la volée
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1

    num_workers = 2 if is_cuda else 4

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.1,
                scale=(0.02, 0.1),
            ),
        ]
    )

    val_tf = weights.transforms()

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    test_ds = datasets.ImageFolder(test_path, transform=val_tf)
    cal_ds = datasets.ImageFolder(cal_path, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )

    cal_loader = DataLoader(
        cal_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    return train_loader, cal_loader, test_loader
