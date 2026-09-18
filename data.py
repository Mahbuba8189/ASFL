# asfl/data.py

import os
import pickle  
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')



# 1. WESAD Dataset

class WESADDataset(Dataset):
    """
    WESAD Dataset from .pkl files.
    Uses chest data (8 channels at 700Hz) and study protocol labels.
    """
    def __init__(self, data_dir, window_size=8, sample_rate=700, 
                 normalize='zscore', seed=42):
        np.random.seed(seed)
        
        self.data_dir = data_dir
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.samples_per_window = window_size * sample_rate  # 5600
        self.input_dim = self.samples_per_window * 8  # 44,800
        self.normalize = normalize
        
        self.data = []
        self.labels = []
        self.subject_ids = []
        self.window_indices = []
        
        # Find all subject folders (S2-S11, S13-S17)
        subject_folders = sorted([f for f in os.listdir(data_dir) 
                                  if os.path.isdir(os.path.join(data_dir, f)) and f.startswith('S')])
        
        # Exclude S1 and S12 (missing due to sensor malfunction)
        subject_folders = [f for f in subject_folders if f not in ['S1', 'S12']]
        
        print(f"Loading WESAD from {len(subject_folders)} subjects: {subject_folders}")
        
        for subject_folder in subject_folders:
            subject_id = subject_folder
            pkl_path = os.path.join(data_dir, subject_folder, f'{subject_id}.pkl')
            
            if not os.path.exists(pkl_path):
                #print(f"  Skipping subject {subject_id}: No .pkl file")
                continue
            
            try:
                # Load .pkl file - pickle is now imported
                with open(pkl_path, 'rb') as f:
                    subject_data = pickle.load(f, encoding='latin1')
                
                # Get chest data
                chest = subject_data['signal']['chest']
                
                # Combine all chest channels into one array
                acc = chest['ACC']  # (samples, 3)
                ecg = chest['ECG']  # (samples, 1)
                emg = chest['EMG']  # (samples, 1)
                eda = chest['EDA']  # (samples, 1)
                temp = chest['Temp']  # (samples, 1)
                resp = chest['Resp']  # (samples, 1)
                
                chest_data = np.concatenate([acc, ecg, emg, eda, temp, resp], axis=1)
                
                # Get labels
                label = subject_data['label']
                
                # Segment into non-overlapping windows
                num_windows = len(chest_data) // self.samples_per_window
                
                for i in range(num_windows):
                    start = i * self.samples_per_window
                    end = start + self.samples_per_window
                    
                    window_data = chest_data[start:end]
                    window_labels = label[start:end]
                    
                    # Only keep labels for 3-class classification
                    valid_window_labels = window_labels[(window_labels == 1) | (window_labels == 2) | (window_labels == 3)]
                    
                    if len(valid_window_labels) == 0:
                        continue
                    
                    unique, counts = np.unique(valid_window_labels, return_counts=True)
                    majority_label = unique[np.argmax(counts)]
                    
                    if majority_label == 1:
                        label_remapped = 0
                    elif majority_label == 2:
                        label_remapped = 1
                    elif majority_label == 3:
                        label_remapped = 2
                    
                    window_flat = window_data.flatten()
                    
                    self.data.append(window_flat.astype(np.float32))
                    self.labels.append(label_remapped)
                    self.subject_ids.append(subject_id)
                    self.window_indices.append(i)
                    
            except Exception as e:
                #print(f"  Error loading subject {subject_id}: {e}")
                continue
        
        # Convert to numpy arrays
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_ids = np.array(self.subject_ids)
        self.window_indices = np.array(self.window_indices)
        
        # Normalize per subject per channel
        if self.normalize == 'zscore' and len(self.data) > 0:
            reshaped = self.data.reshape(-1, self.samples_per_window, 8)
            
            for subject in np.unique(self.subject_ids):
                mask = self.subject_ids == subject
                subject_data = reshaped[mask]
                
                for c in range(8):
                    channel_mean = subject_data[:, :, c].mean()
                    channel_std = subject_data[:, :, c].std()
                    if channel_std > 1e-8:
                        subject_data[:, :, c] = (subject_data[:, :, c] - channel_mean) / channel_std
                
                reshaped[mask] = subject_data
            
            self.data = reshaped.reshape(-1, self.input_dim)
            
        elif self.normalize == 'minmax' and len(self.data) > 0:
            reshaped = self.data.reshape(-1, self.samples_per_window, 8)
            for c in range(8):
                channel_min = reshaped[:, :, c].min()
                channel_max = reshaped[:, :, c].max()
                if channel_max - channel_min > 1e-8:
                    reshaped[:, :, c] = (reshaped[:, :, c] - channel_min) / (channel_max - channel_min)
            self.data = reshaped.reshape(-1, self.input_dim)
        
        print(f"\nLoaded {len(self.data)} samples from {len(np.unique(self.subject_ids))} subjects")
        if len(self.data) > 0:
            print(f"  Class distribution:")
            print(f"  Feature shape: {self.data.shape}")
            print(f"  Input dimension: {self.input_dim}")
            print(f"  Normalization: {self.normalize}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_subject_data(self, subject_id):
        mask = self.subject_ids == subject_id
        return self.data[mask], self.labels[mask], self.window_indices[mask]
    
    def get_class_weights(self):
        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        if total == 0:
            return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_stats(self):
        return {
            'total_samples': len(self.data),
            'total_subjects': len(np.unique(self.subject_ids)),
            'input_dim': self.input_dim,
            'class_distribution': {
                int(c): int(np.sum(self.labels == c)) 
                for c in np.unique(self.labels)
            }
        }


def get_wesad_loaders(data_dir, oversample_target=None, num_clients=15, batch_size=32, 
                      partition_mode='iid', non_iid_alpha=None, 
                      window_size=8, normalize='zscore',
                      val_split=0.2, seed=99):
    """
    Get WESAD data loaders with IID or Dirichlet Non-IID partitioning.
    
    Args:
        partition_mode: 'iid' for IID distribution, 'dirichlet' for Non-IID
        non_iid_alpha: Concentration parameter for Dirichlet (e.g., 0.5 for non-IID)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = WESADDataset(data_dir, window_size=window_size, normalize=normalize)
    
    if len(dataset) == 0:
        raise ValueError("No data loaded. Check data directory.")
    
    if oversample_target is not None:
        print(f"Warning: oversample_target is set but WESAD does not use oversampling. Ignoring.")
    indices = list(range(len(dataset)))
    
    if partition_mode == 'iid':
        indices_shuffled = np.random.permutation(indices)
        client_indices = np.array_split(indices_shuffled, num_clients)
        
    elif partition_mode == 'dirichlet':
        if non_iid_alpha is None:
            raise ValueError("non_iid_alpha must be provided for dirichlet mode")
        
        labels = dataset.labels
        num_classes = len(np.unique(labels))
        
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            proportions = np.random.dirichlet([non_iid_alpha] * num_clients)
            samples_for_class = class_indices[c]
            np.random.shuffle(samples_for_class)
            
            split_points = np.cumsum(proportions * len(samples_for_class)).astype(int)[:-1]
            split_indices = np.split(samples_for_class, split_points)
            for i in range(num_clients):
                client_indices[i].extend(split_indices[i].tolist())
        
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
        
        # Filter out empty clients
        client_indices = [idx for idx in client_indices if len(idx) > 0]
    
    train_loaders = []
    val_loaders = []
    
    for indices in client_indices:
        if len(indices) == 0:
            continue
        
        # Ensure we have enough samples for both train and val
        if len(indices) < 2:
            continue
        
        np.random.shuffle(indices)
        
        # Robust split: ensure at least 1 sample for both train and val
        val_size = max(1, int(len(indices) * val_split))
        val_size = min(val_size, len(indices) - 1)  # Leave at least 1 for training
        train_size = len(indices) - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Client-level oversampling is DISABLED - we do global oversampling instead
        train_dataset = Subset(dataset, train_indices)
        
        train_loaders.append(DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False))
    
    return train_loaders, val_loaders


# 2. BIDMC-PPG Dataset

class BIDMCPPGDataset(Dataset):
    """
    BIDMC-PPG Dataset from CSV files.
    Uses ' PLETH' column (PPG signal at 125Hz) and ' HR' column (heart rate at 1Hz).
    """
    def __init__(self, data_dir, window_size=8, sample_rate=125, 
                 normalize='minmax', seed=99):
        np.random.seed(seed)
        
        self.data_dir = data_dir
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.samples_per_window = window_size * sample_rate
        self.input_dim = self.samples_per_window
        self.normalize = normalize
        
        self.data = []
        self.labels = []
        self.subject_ids = []
        self.window_indices = []
        
        signal_files = sorted([f for f in os.listdir(data_dir) 
                              if f.endswith('_Signals.csv')])
        
        print(f"Loading BIDMC-PPG from {len(signal_files)} subjects...")
        
        for signal_file in signal_files:
            subject_id = signal_file.split('_')[1]
            
            numerics_file = f'bidmc_{subject_id}_Numerics.csv'
            numerics_path = os.path.join(data_dir, numerics_file)
            
            if not os.path.exists(numerics_path):
                continue
            
            try:
                df_signals = pd.read_csv(os.path.join(data_dir, signal_file))
                df_numerics = pd.read_csv(numerics_path)
                
                if ' PLETH' in df_signals.columns:
                    ppg = df_signals[' PLETH'].values.astype(np.float32)
                elif 'PLETH' in df_signals.columns:
                    ppg = df_signals['PLETH'].values.astype(np.float32)
                
                if ' HR' in df_numerics.columns:
                    hr = df_numerics[' HR'].values.astype(np.float32)
                elif 'HR' in df_numerics.columns:
                    hr = df_numerics['HR'].values.astype(np.float32)
                
                if np.isnan(hr).any():
                    hr_series = pd.Series(hr)
                    hr = hr_series.ffill().bfill().values
                
                num_windows = len(ppg) // self.samples_per_window
                
                for i in range(num_windows):
                    start = i * self.samples_per_window
                    end = start + self.samples_per_window
                    
                    window_ppg = ppg[start:end]
                    
                    hr_start_idx = start // self.sample_rate
                    hr_end_idx = end // self.sample_rate
                    window_hr = hr[hr_start_idx:hr_end_idx]
                    window_hr = window_hr[~np.isnan(window_hr)]
                    
                    if len(window_hr) == 0:
                        continue
                    
                    hr_median = np.median(window_hr)

                    if hr_median < 60:
                        label = 0
                    elif hr_median <= 100:
                        label = 1
                    else:
                        label = 2
              
                    self.data.append(window_ppg)
                    self.labels.append(label)
                    self.subject_ids.append(subject_id)
                    self.window_indices.append(i)
                    
            except Exception as e:

                continue
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_ids = np.array(self.subject_ids)
        self.window_indices = np.array(self.window_indices)
        
        if self.normalize == 'minmax' and len(self.data) > 0:
            for i in range(len(self.data)):
                min_val = self.data[i].min()
                max_val = self.data[i].max()
                if max_val - min_val > 1e-8:
                    self.data[i] = (self.data[i] - min_val) / (max_val - min_val)
                    
        elif self.normalize == 'zscore' and len(self.data) > 0:
            for i in range(len(self.data)):
                mean_val = self.data[i].mean()
                std_val = self.data[i].std()
                if std_val > 1e-8:
                    self.data[i] = (self.data[i] - mean_val) / std_val

            print(f"  Feature shape: {self.data.shape}")
            print(f"  Input dimension: {self.input_dim}")
            print(f"  Normalization: {self.normalize}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_subject_data(self, subject_id):
        mask = self.subject_ids == subject_id
        return self.data[mask], self.labels[mask], self.window_indices[mask]
    
    def get_class_weights(self):
        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        if total == 0:
            return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_stats(self):
        return {
            'total_samples': len(self.data),
            'total_subjects': len(np.unique(self.subject_ids)),
            'input_dim': self.input_dim,
            'class_distribution': {
                int(c): int(np.sum(self.labels == c)) 
                for c in np.unique(self.labels)
            }
        }


def get_bidmc_loaders(data_dir, num_clients=53, batch_size=32, 
                      partition_mode='iid', non_iid_alpha=None,
                      window_size=8, normalize='minmax',
                      val_split=0.2, seed=99,
                      oversample_target=None):
    print(f"DEBUG: get_bidmc_loaders called with oversample_target={oversample_target}")
    """
    Get BIDMC-PPG data loaders with IID or Dirichlet Non-IID partitioning.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = BIDMCPPGDataset(data_dir, window_size=window_size, normalize=normalize)
    
    if len(dataset) == 0:
        raise ValueError("No data loaded. Check data directory.")
    
    if oversample_target is not None:
        print(f"Applying GLOBAL SMOTE with target={oversample_target}")
        X = dataset.data
        y = dataset.labels
        print(f"Before global SMOTE: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}, Class 2={np.sum(y==2)}")
        
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(
            sampling_strategy={0: oversample_target},
            random_state=seed,
            k_neighbors=5
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        dataset.data = X_resampled
        dataset.labels = y_resampled
        print(f"After global SMOTE: Class 0={np.sum(y_resampled==0)}, Class 1={np.sum(y_resampled==1)}, Class 2={np.sum(y_resampled==2)}")
    
    indices = list(range(len(dataset)))
    
    if partition_mode == 'iid':
        indices_shuffled = np.random.permutation(indices)
        client_indices = np.array_split(indices_shuffled, num_clients)
        
    elif partition_mode == 'dirichlet':
        if non_iid_alpha is None:
            raise ValueError("non_iid_alpha must be provided for dirichlet mode")
        
        labels = dataset.labels
        num_classes = len(np.unique(labels))
        
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            proportions = np.random.dirichlet([non_iid_alpha] * num_clients)
            samples_for_class = class_indices[c]
            np.random.shuffle(samples_for_class)
            
            split_points = np.cumsum(proportions * len(samples_for_class)).astype(int)[:-1]
            split_indices = np.split(samples_for_class, split_points)
            for i in range(num_clients):
                client_indices[i].extend(split_indices[i].tolist())
        
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
        
        # Filter out empty clients
        client_indices = [idx for idx in client_indices if len(idx) > 0]
    
    train_loaders = []
    val_loaders = []
    
    for client_idx, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue
        
        # Ensure we have enough samples for both train and val
        if len(indices) < 2:
            continue
        
        np.random.shuffle(indices)
        
        # Robust split: ensure at least 1 sample for both train and val
        val_size = max(1, int(len(indices) * val_split))
        val_size = min(val_size, len(indices) - 1)  # Leave at least 1 for training
        train_size = len(indices) - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Client-level oversampling is DISABLED - we do global oversampling instead
        train_dataset = Subset(dataset, train_indices)
        
        
        val_dataset = Subset(dataset, val_indices)
        
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
    
    return train_loaders, val_loaders


# 3. MHEALTH Dataset

class MHEALTHDataset(Dataset):
    """
    MHEALTH Dataset from UCI.
    10 subjects, 12 activities, 3 sensors (chest, wrist, ankle).
    
    Sensor placement:
    - Chest: acceleration (3 axes), ECG (2 leads)
    - Left ankle: acceleration (3 axes), gyro (3 axes)
    - Right wrist: acceleration (3 axes)
    
    Total: 14 channels
    Input dimension: 14 channels * 750 samples = 10500
    """
    def __init__(self, data_dir, window_size=15, sample_rate=50,
                 normalize='zscore', seed=99):
        np.random.seed(seed)
        
        self.data_dir = data_dir
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.samples_per_window = window_size * sample_rate  # 750
        self.input_dim = self.samples_per_window * 14  # 10500
        self.normalize = normalize
        
        self.data = []
        self.labels = []
        self.subject_ids = []
        self.window_indices = []
        
        # Label mapping: 1->0, 2->1, ..., 12->11
        self.label_map = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
            7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11
        }
        
        # Select 14 channels for ASFL:
        # 0-2: chest acceleration, 3-4: ECG (lead 1, lead 2)
        # 5-7: left ankle acceleration, 8-10: left ankle gyro
        # 14-16: right wrist acceleration
        selected_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16]
        
        # Find subject files
        subject_files = sorted([f for f in os.listdir(data_dir) 
                               if f.startswith('mHealth_subject') and f.endswith('.log')])
        
        print(f"Loading MHEALTH from {len(subject_files)} subjects...")
        
        for subject_file in subject_files:
            subject_id = subject_file.split('_')[1].replace('.log', '')
            
            try:
                data = np.loadtxt(os.path.join(data_dir, subject_file))
                
                # Extract selected channels
                signal_data = data[:, selected_cols]
                
                # Labels are in column 23
                labels = data[:, 23].astype(int)
                
                # Remap labels: 1->0, 2->1, ..., 12->11
                remapped_labels = np.array([self.label_map.get(l, -1) for l in labels])
                
                # Remove null labels (0) and invalid labels (-1)
                valid_mask = remapped_labels >= 0
                signal_data = signal_data[valid_mask]
                remapped_labels = remapped_labels[valid_mask]
                
                if len(signal_data) == 0:
                    continue
                
                # Segment into windows
                num_windows = len(signal_data) // self.samples_per_window
                
                for i in range(num_windows):
                    start = i * self.samples_per_window
                    end = start + self.samples_per_window
                    
                    window_data = signal_data[start:end]
                    window_labels = remapped_labels[start:end]
                    
                    # Majority label for window
                    unique, counts = np.unique(window_labels, return_counts=True)
                    majority_label = unique[np.argmax(counts)]
                    
                    window_flat = window_data.flatten()
                    
                    self.data.append(window_flat.astype(np.float32))
                    self.labels.append(majority_label)
                    self.subject_ids.append(subject_id)
                    self.window_indices.append(i)
                    
            except Exception as e:
                print(f"  Error loading subject {subject_id}: {e}")
                continue
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_ids = np.array(self.subject_ids)
        self.window_indices = np.array(self.window_indices)
        
        # Normalize
        if self.normalize == 'zscore' and len(self.data) > 0:
            for i in range(self.data.shape[1]):
                mean = self.data[:, i].mean()
                std = self.data[:, i].std()
                if std > 1e-8:
                    self.data[:, i] = (self.data[:, i] - mean) / std
                    
        elif self.normalize == 'minmax' and len(self.data) > 0:
            for i in range(self.data.shape[1]):
                min_val = self.data[:, i].min()
                max_val = self.data[:, i].max()
                if max_val - min_val > 1e-8:
                    self.data[:, i] = (self.data[:, i] - min_val) / (max_val - min_val)
        
        print(f"\nLoaded {len(self.data)} samples from {len(np.unique(self.subject_ids))} subjects")
        if len(self.data) > 0:
            print(f"  Class distribution:")
            #for c in range(12):
                #count = np.sum(self.labels == c)
                #print(f"    Class {c}: {count} samples")
            print(f"  Feature shape: {self.data.shape}")
            print(f"  Input dimension: {self.input_dim}")
            print(f"  Normalization: {self.normalize}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_subject_data(self, subject_id):
        mask = self.subject_ids == subject_id
        return self.data[mask], self.labels[mask], self.window_indices[mask]
    
    def get_class_weights(self):
        class_counts = np.bincount(self.labels, minlength=12)
        total = len(self.labels)
        
        weights = []
        for count in class_counts:
            if count > 0:
                weights.append(total / (12 * count))
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_stats(self):
        return {
            'total_samples': len(self.data),
            'total_subjects': len(np.unique(self.subject_ids)),
            'input_dim': self.input_dim,
            'class_distribution': {
                int(c): int(np.sum(self.labels == c)) 
                for c in np.unique(self.labels)
            }
        }


def get_mhealth_loaders(data_dir, oversample_target=None, num_clients=10, batch_size=32,
                        partition_mode='iid', non_iid_alpha=None,
                        window_size=15, normalize='zscore',
                        val_split=0.2, seed=99):
    """
    Get MHEALTH data loaders with IID or Dirichlet Non-IID partitioning.
    
    Args:
        data_dir: Path to MHEALTH dataset
        num_clients: Number of clients (default: 10)
        batch_size: Batch size for training
        partition_mode: 'iid' or 'dirichlet'
        non_iid_alpha: Concentration parameter for Dirichlet (e.g., 0.5 for non-IID)
        window_size: Window size in seconds (default: 15)
        normalize: 'zscore' or 'minmax'
        val_split: Validation split ratio
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = MHEALTHDataset(data_dir, window_size=window_size, sample_rate=50, normalize=normalize, seed=seed)
    
    if len(dataset) == 0:
        raise ValueError("No data loaded. Check data directory.")
    if oversample_target is not None:
        print(f"Warning: oversample_target is set but MHEALTH does not use oversampling. Ignoring.")
        
    indices = list(range(len(dataset)))
    
    if partition_mode == 'iid':
        indices_shuffled = np.random.permutation(indices)
        client_indices = np.array_split(indices_shuffled, num_clients)
        
    elif partition_mode == 'dirichlet':
        if non_iid_alpha is None:
            raise ValueError("non_iid_alpha must be provided for dirichlet mode")
        
        labels = dataset.labels
        num_classes = len(np.unique(labels))
        
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            proportions = np.random.dirichlet([non_iid_alpha] * num_clients)
            samples_for_class = class_indices[c]
            np.random.shuffle(samples_for_class)
            
            split_points = np.cumsum(proportions * len(samples_for_class)).astype(int)[:-1]
            split_indices = np.split(samples_for_class, split_points)
            for i in range(num_clients):
                client_indices[i].extend(split_indices[i].tolist())
        
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
        
        # Filter out empty clients
        client_indices = [idx for idx in client_indices if len(idx) > 0]
    
    train_loaders = []
    val_loaders = []
    
    for indices in client_indices:
        if len(indices) == 0:
            continue
        
        # Ensure we have enough samples for both train and val
        if len(indices) < 2:
            continue
        
        np.random.shuffle(indices)
        
        # Robust split: ensure at least 1 sample for both train and val
        val_size = max(1, int(len(indices) * val_split))
        val_size = min(val_size, len(indices) - 1)  # Leave at least 1 for training
        train_size = len(indices) - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Client-level oversampling is DISABLED - we do global oversampling instead
        train_dataset = Subset(dataset, train_indices)
        
        train_loaders.append(DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False))
    
    return train_loaders, val_loaders


# Verification Functions

def verify_dataset(dataset, name):
    print("\n" + "=" * 60)
    print(f"{name} Verification")
    print("=" * 60)
    print(f"  Samples: {len(dataset)}")
    if hasattr(dataset, 'input_dim'):
        print(f"  Input dim: {dataset.input_dim}")
    if hasattr(dataset, 'subject_ids'):
        print(f"  Subjects: {len(np.unique(dataset.subject_ids))}")
    if hasattr(dataset, 'patient_ids'):
        print(f"  Patients: {len(np.unique(dataset.patient_ids))}")
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
        print(f"  Classes: {np.unique(labels)}")
        for c in np.unique(labels):
            print(f"    Class {c}: {np.sum(labels == c)} samples")
    print("=" * 60)