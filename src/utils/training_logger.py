import logging
import os
import csv
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs", run_name=None):
        """
        Initializes the logger. Creates a log directory if it doesn't exist.
        Writes to both a .log file and a .csv file.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Use timestamp if no run name is provided to prevent overwriting old logs
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.log_file = os.path.join(log_dir, f"train_{run_name}.log")
        self.csv_file = os.path.join(log_dir, f"metrics_{run_name}.csv")
        
        # Setup standard Python logging
        self.logger = logging.getLogger(f"TrainingLogger_{run_name}")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate logs if instantiated multiple times
        if getattr(self.logger, '_init_done', False):
            return
            
        # Formatter ensures consistent timestamps
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # File Handler (writes to text file)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console Handler (writes to terminal)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger._init_done = True
        
        # Internal storage for CSV metrics
        self.metrics = []
        
        self.info(f"Logging initialized. Tracking run: {run_name}")
        self.info(f"Log file: {self.log_file}")
        self.info(f"Metrics file: {self.csv_file}")

    def info(self, message):
        """Log a standard informational message."""
        self.logger.info(message)
        
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Log an error message."""
        self.logger.error(message)

    def log_epoch(self, epoch, train_loss, val_loss=None, val_acc=None):
        """
        Log metrics for a specific epoch, prints them to console/file, 
        and safely dumps the history to a CSV.
        """
        metric_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss if val_loss is not None else '',
            'val_acc': val_acc if val_acc is not None else ''
        }
        self.metrics.append(metric_entry)
        
        log_msg = f"Epoch [{epoch}] | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        if val_acc is not None:
            log_msg += f" | Val Acc: {val_acc:.2f}%"
            
        self.info(log_msg)
        
        # Auto-save CSV after every epoch to prevent data loss upon crash
        self._save_csv()

    def _save_csv(self):
        """Internal method to write metrics to a CSV file."""
        if not self.metrics:
            return
            
        keys = self.metrics[0].keys()
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics)