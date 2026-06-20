import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import os
from dynamic_dataset_class import DynamicKeywordDataset


def prepare_dataloaders(full_data_list, noise_dir, class_mapping, batch_size=64, val_split=0.1, plot_distribution_per_class=False,pin_memory=True):
    """
    Splits the raw data list and returns PyTorch DataLoaders optimized for GPU.
    full_data_list format: [("path/yes_1.wav", 0), ("path/no_1.wav", 1), ...]
    class_mapping format: {"yes": 0, "no": 1, "up": 2, ...}
    """
    
    # 1. Extract labels for stratification
    labels = [item[1] for item in full_data_list]
    
    # 2. Perform the Stratified Split
    train_list, val_list = train_test_split(
        full_data_list, 
        test_size=val_split, 
        stratify=labels, 
        random_state=42
    )
    
    print(f"Total samples: {len(full_data_list)}")
    print(f"Training samples: {len(train_list)}")
    print(f"Validation samples: {len(val_list)}")

    # 2.5. Plot class distribution if requested
    if plot_distribution_per_class:
        # Count class distribution
        train_labels = [item[1] for item in train_list]
        val_labels = [item[1] for item in val_list]
        
        train_counts = Counter(train_labels)
        val_counts = Counter(val_labels)
        
        # Create reverse mapping (int -> class name)
        reverse_mapping = {v: k for k, v in class_mapping.items()}
        
        # Get all unique classes and sort by their integer values
        all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()))
        
        # Prepare data for plotting with class names
        train_values = [train_counts.get(cls, 0) for cls in all_classes]
        val_values = [val_counts.get(cls, 0) for cls in all_classes]
        class_names = [reverse_mapping[cls] for cls in all_classes]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Train plot
        ax1.bar(class_names, train_values, color='steelblue', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Class Label', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xticklabels(class_names, ha='right')
        
        # Validation plot
        ax2.bar(class_names, val_values, color='coral', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Class Label', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Set Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_xticklabels(class_names, ha='right')
        
        
        plt.tight_layout()
        
        # Create plots folder if it doesn't exist
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Save figure
        save_path = os.path.join(plots_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
        plt.close()

    # 3. Instantiate the Datasets
    train_dataset = DynamicKeywordDataset(
        data_list=train_list, 
        noise_dir=noise_dir, 
        is_training=True
    )
    
    val_dataset = DynamicKeywordDataset(
        data_list=val_list, 
        noise_dir=noise_dir, 
        is_training=False
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader