import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class ISIC2019Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_noisy=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_noisy = is_noisy
        
        # 'image' or 'image_name' column depending on dataset, usually 'image'
        self.image_names = df['image'].values
        
        # If 'is_clean' is in df, keep track for evaluation
        self.has_clean_labels = 'is_clean' in df.columns
        if self.has_clean_labels:
            self.is_clean = df['is_clean'].values
        else:
            self.is_clean = np.ones(len(df), dtype=bool)
            
        self.labels = df['label'].values
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_name = str(self.image_names[idx])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        
        # For evaluation/analysis, return idx as well
        if self.has_clean_labels:
            clean_status = bool(self.is_clean[idx])
            return image, label, idx, int(clean_status)
        return image, label, idx, 1

def get_transforms(image_size):
    # SimCLR needs two views, but for regular dataset we return one
    # I'll handle SimCLR dataset separately or with a specific transform
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

class SimCLRTransform:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def inject_noise(df, noise_level, num_classes, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    noisy_df = df.copy()
    noisy_df['is_clean'] = True
    noisy_df['original_label'] = noisy_df['label']
    
    if noise_level > 0.0:
        n_samples = len(noisy_df)
        n_noisy = int(noise_level * n_samples)
        
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        
        for idx in noisy_indices:
            current_label = noisy_df.iloc[idx]['label']
            possible_labels = [c for c in range(num_classes) if c != current_label]
            new_label = np.random.choice(possible_labels)
            noisy_df.iat[idx, noisy_df.columns.get_loc('label')] = new_label
            noisy_df.iat[idx, noisy_df.columns.get_loc('is_clean')] = False
            
    return noisy_df

def prepare_data(config):
    csv_path = config['csv_path']
    # Check if csv exists, else might be named ISIC_2019_Training_GroundTruth.csv
    if not os.path.exists(csv_path):
        csv_path = os.path.join(config['data_dir'], 'ISIC_2019_Training_GroundTruth.csv')
    
    df = pd.read_csv(csv_path)
    
    # Process labels
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    
    # convert one-hot to categorical
    def get_label(row):
        for i, c in enumerate(classes):
            if c in row and row[c] == 1.0:
                return i
        return 0 # Default fallback
        
    if 'label' not in df.columns:
        df['label'] = df.apply(get_label, axis=1)
        
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=config['seed'], stratify=df['label'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=config['seed'], stratify=test_df['label'])
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), len(classes)

def get_dataloaders(config, noise_level=0.0):
    train_df, val_df, test_df, num_classes = prepare_data(config)
    
    # Inject noise into training data only
    train_df_noisy = inject_noise(train_df, noise_level, num_classes, config['seed'])
    
    # Check image dir
    img_dir = config['image_dir']
    if not os.path.exists(img_dir):
        img_dir = os.path.join(config['data_dir'], 'ISIC_2019_Training_Input')
    
    train_transform, test_transform = get_transforms(config['image_size'])
    
    train_dataset = ISIC2019Dataset(train_df_noisy, img_dir, transform=train_transform, is_noisy=True)
    val_dataset = ISIC2019Dataset(val_df, img_dir, transform=test_transform)
    test_dataset = ISIC2019Dataset(test_df, img_dir, transform=test_transform)
    
    # Calculate class weights for class balancing
    labels = train_df_noisy['label'].values
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    return train_loader, val_loader, test_loader, num_classes, class_weights, train_dataset

def get_simclr_dataloader(config):
    train_df, _, _, _ = prepare_data(config)
    img_dir = config['image_dir']
    if not os.path.exists(img_dir):
        img_dir = os.path.join(config['data_dir'], 'ISIC_2019_Training_Input')
        
    simclr_transform = SimCLRTransform(config['image_size'])
    # labels don't matter for SimCLR
    train_dataset = ISIC2019Dataset(train_df, img_dir, transform=simclr_transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    
    return train_loader
