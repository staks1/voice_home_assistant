
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
from utils.gpu_utils import run_gpu_monitor


# Assuming you have imported your model, dataloaders, and logger
from utils.training_logger import TrainingLogger
model_path = "/home/st1/Documents/training_cnn_for_waveform/models"
plots_path = "/home/st1/Documents/training_cnn_for_waveform/plots"

def train_model(model, train_loader, val_loader, num_epochs=2, device='cuda:0', save_path=os.path.join(model_path,'best_keyword_model.pth')):
    # Initialize the logger
    logger = TrainingLogger()
    
    # 1. Hyperparameters
    # CrossEntropyLoss automatically applies Softmax, do not add Softmax to your model's final layer
    criterion = nn.CrossEntropyLoss()
    
    # 1e-3 is the standard starting LR for Adam on fresh weights.
    # weight_decay=1e-4 adds L2 regularization to fight overfitting on your small dataset.
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Reduces the learning rate by 50% every 20 epochs to help the model settle into the minimum
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    model = model.to(device)
    
    # Track the highest validation accuracy achieved across all epochs
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ==========================
        # TRAINING PHASE
        # ==========================
        model.train() # Critical: Enables Dropout and BatchNorm updates
        running_train_loss = 0.0
        
        # Use tqdm for a clean progress bar instead of printing every batch
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for data, labels in train_bar:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, labels)
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * data.size(0)
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Step the learning rate scheduler
        scheduler.step()

        # ==========================
        # VALIDATION PHASE
        # ==========================
        model.eval() # Critical: Disables Dropout, freezes BatchNorm stats
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad(): # Critical: Disables gradient calculation to save VRAM
            for data, labels in val_bar:
                data = data.to(device)
                labels = labels.to(device)

                scores = model(data)
                loss = criterion(scores, labels)
                
                running_val_loss += loss.item() * data.size(0)
                
                # Calculate Accuracy
                _, predicted = torch.max(scores.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_predictions / total_predictions) * 100.0

        # ==========================
        # LOGGING & SAVING
        # ==========================
        logger.log_epoch(
            epoch=epoch+1, 
            train_loss=epoch_train_loss, 
            val_loss=epoch_val_loss, 
            val_acc=epoch_val_acc
        )
        
        # Save the best model based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> New best validation accuracy: {best_val_acc:.2f}%. Model saved to {save_path}")
        
        # monitor gpu at end of epoch
        run_gpu_monitor()

    print("Training Complete.")

    # NEW: Trigger the plot rendering after the loop finishes
    logger.plot_metrics(save_path=os.path.join(plots_path,'training_curves.png'))
    return model