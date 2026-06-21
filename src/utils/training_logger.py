import logging
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt


log_path = '/home/st1/Documents/training_cnn_for_waveform/logs'


class TrainingLogger:
    def __init__(self, log_dir=log_path):
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.epochs = []
        
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.csv_path = os.path.join(self.log_dir, 'training_metrics.csv')
        
        # Initialize CSV with headers
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Accuracy'])

    def log_epoch(self, epoch, train_loss, val_loss, val_acc):
        # Store in memory for plotting
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        # Save to CSV for persistent logging
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_loss, val_acc])

    def plot_metrics(self, save_path=None):
        if not self.epochs:
            print("No data to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Training vs Validation Loss
        ax1.plot(self.epochs, self.train_losses, label='Train Loss', color='blue', marker='o', markersize=3)
        ax1.plot(self.epochs, self.val_losses, label='Validation Loss', color='red', marker='o', markersize=3)
        ax1.set_title('Cross Entropy Loss per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Validation Accuracy
        ax2.plot(self.epochs, self.val_accs, label='Validation Accuracy', color='green', marker='o', markersize=3)
        ax2.set_title('Validation Accuracy per Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.log_dir, save_path))
            print(f"Plot saved to {os.path.join(self.log_dir, save_path)}")
            
        plt.show()