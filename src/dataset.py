import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# Clinically plausible confusions, used by noise_type='class_dependent'
# (indices follow CLASSES: MEL=0 NV=1 BCC=2 AK=3 BKL=4 DF=5 VASC=6 SCC=7)
CONFUSABLE = {0: [1], 1: [0, 4], 2: [7], 3: [7], 4: [1], 5: [1], 6: [1], 7: [3]}

NOISE_TYPES = ('uniform_global', 'uniform_classwise', 'class_dependent')


class ISIC2019Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = df['image'].values
        self.has_clean_labels = 'is_clean' in df.columns
        self.is_clean = df['is_clean'].values if self.has_clean_labels else np.ones(len(df), dtype=bool)
        self.labels = df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.image_names[idx])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx]), idx, int(bool(self.is_clean[idx]))


def get_transforms(image_size):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(), norm])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), norm])
    return train_transform, test_transform


class SimCLRTransform:
    """Dermoscopy-friendly SimCLR views: vertical flips + rotation are valid for
    lesions, and hue jitter is kept small because colour is diagnostic."""
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.02)], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def inject_noise(df, noise_level, num_classes, seed=42, noise_type='uniform_global'):
    """
    uniform_global    : exactly noise_level*N random samples flip to a uniformly random other class.
                        Rare classes get flooded (they receive as many wrong labels as big ones).
    uniform_classwise : every sample flips with prob noise_level -> every class has ~noise_level noise.
    class_dependent   : like classwise, but 80% of flips go to a clinically confusable class.
    """
    assert noise_type in NOISE_TYPES + ('none',), noise_type
    noisy_df = df.copy()
    noisy_df['original_label'] = noisy_df['label']
    noisy_df['is_clean'] = True
    if noise_level <= 0.0 or noise_type == 'none':
        return noisy_df

    rng = np.random.default_rng(seed)
    labels = noisy_df['label'].values.astype(np.int64)
    n = len(labels)
    if noise_type == 'uniform_global':
        flip = np.zeros(n, dtype=bool)
        flip[rng.choice(n, int(noise_level * n), replace=False)] = True
    else:
        flip = rng.random(n) < noise_level

    new = labels.copy()
    for i in np.where(flip)[0]:
        y = labels[i]
        if noise_type == 'class_dependent' and rng.random() < 0.8:
            new[i] = rng.choice(CONFUSABLE[y])
        else:
            new[i] = rng.choice([c for c in range(num_classes) if c != y])
    noisy_df['label'] = new
    noisy_df['is_clean'] = (new == labels)
    return noisy_df


def prepare_data(config):
    csv_path = config['csv_path']
    if not os.path.exists(csv_path):
        csv_path = os.path.join(config['data_dir'], 'ISIC_2019_Training_GroundTruth.csv')
    df = pd.read_csv(csv_path)

    def get_label(row):
        for i, c in enumerate(CLASSES):
            if c in row and row[c] == 1.0:
                return i
        return 0

    if 'label' not in df.columns:
        df['label'] = df.apply(get_label, axis=1)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=config['seed'], stratify=df['label'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=config['seed'], stratify=test_df['label'])
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True), len(CLASSES))


def _img_dir(config):
    d = config['image_dir']
    return d if os.path.exists(d) else os.path.join(config['data_dir'], 'ISIC_2019_Training_Input')


def get_dataloaders(config, noise_level=0.0, noise_type='uniform_global', val_mode='clean', seed=None):
    """
    val_mode: 'clean' -> clean validation set (best-epoch selection uses true labels)
              'noisy' -> validation labels get the same noise (realistic: no clean data)
              'last'  -> validation is not used; the last epoch is evaluated
    Returns a dict. 'score' is a NON-augmented, NON-shuffled, drop_last=False loader over the
    noisy training set; position i in its outputs == dataset index i.
    """
    seed = config['seed'] if seed is None else seed
    train_df, val_df, test_df, num_classes = prepare_data(config)

    train_noisy = inject_noise(train_df, noise_level, num_classes, seed, noise_type)
    if val_mode == 'noisy':
        val_df = inject_noise(val_df, noise_level, num_classes, seed + 1000, noise_type)

    train_tf, test_tf = get_transforms(config['image_size'])
    img_dir = _img_dir(config)
    train_ds = ISIC2019Dataset(train_noisy, img_dir, transform=train_tf)
    score_ds = ISIC2019Dataset(train_noisy, img_dir, transform=test_tf)
    val_ds = ISIC2019Dataset(val_df, img_dir, transform=test_tf)
    test_ds = ISIC2019Dataset(test_df, img_dir, transform=test_tf)

    counts = np.bincount(train_noisy['label'].values, minlength=num_classes)
    class_weights = torch.FloatTensor(len(train_noisy) / (num_classes * np.maximum(counts, 1)))

    bs, nw = config['batch_size'], config['num_workers']
    pin = torch.cuda.is_available()
    return dict(
        train=DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True, pin_memory=pin),
        val=DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin),
        test=DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin),
        score=DataLoader(score_ds, batch_size=bs * 2, shuffle=False, num_workers=nw, drop_last=False, pin_memory=pin),
        num_classes=num_classes, class_weights=class_weights, train_dataset=train_ds)


def get_simclr_dataloader(config):
    train_df, _, _, _ = prepare_data(config)
    ds = ISIC2019Dataset(train_df, _img_dir(config), transform=SimCLRTransform(config['image_size']))
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=True,
                      num_workers=config['num_workers'], drop_last=True)
