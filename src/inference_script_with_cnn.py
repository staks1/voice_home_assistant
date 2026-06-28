import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import os 
import time 

from models import MobileNetV2Custom

class KeywordInferencer:
    def __init__(self, model_path='/home/st1/Documents/training_cnn_for_waveform/models/best_keyword_model.pth', num_classes=14, device='cuda:3', sample_rate=16000):
        """
        Initializes the model and the exact audio transformation pipeline used during training.
        """
        self.device = device
        self.sample_rate = sample_rate
        
        # Max length mathematically mapped from your dataset class (53248 samples)
        self.max_length = 53248
        
        # 1. Initialize the Model
        self.model = MobileNetV2Custom(num_classes=num_classes, freeze_base=False)
        
        # 2. Load the highly-trained weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    
        
        # CRITICAL: Set model to evaluation mode (disables Dropout, freezes BatchNorm)
        self.model.eval()
        
        # 3. Initialize the exact Mel-Spectrogram transformer from the dataset class
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64,
            n_fft=512,
            hop_length=160 
        ).to(self.device)
        
        # Initialize DB converter
        self.db_transform = T.AmplitudeToDB(stype='power', top_db=80).to(self.device)

    def _pad_or_truncate(self, waveform):
        """
        Forces the 1D waveform to be exactly self.max_length samples.
        Must mathematically mirror the DynamicKeywordDataset logic.
        """
        length = waveform.shape[1]
        if length > self.max_length:
            # Truncate
            return waveform[:, :self.max_length]
        elif length < self.max_length:
            # Pad with silence (zeros) at the end
            pad_amount = self.max_length - length
            return F.pad(waveform, (0, pad_amount))
        return waveform

    def predict(self, audio_path):
        """
        Processes a single .wav file and returns the predicted class ID and confidence scores.
        """
        # 1. Load Audio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

        # Handle sample rate mismatches (if the file isn't exactly 16000Hz)
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 2. Standardize Length
        waveform = self._pad_or_truncate(waveform)
        
        # Move waveform to the target GPU
        waveform = waveform.to(self.device)

        

        # 3. Audio to Spectrogram Pipeline
        with torch.no_grad():
            # Convert to Mel-Spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # Convert to Decibels
            mel_spec_db = self.db_transform(mel_spec)

            # 4. Normalization (Exact mirror of dataset Z-score logic)
            mean = mel_spec_db.mean()
            std = mel_spec_db.std()

            if std > 1e-6:
                mel_spec_normalized = (mel_spec_db - mean) / std
            else:
                mel_spec_normalized = mel_spec_db - mean
            
            # 5. Model Inference
            # The model expects a batch dimension: [Batch, Channel, Mels, Time]
            # Currently we have [Channel, Mels, Time], so we unsqueeze to add Batch=1
            input_tensor = mel_spec_normalized.unsqueeze(0)
            
            # Forward pass
            logits = self.model(input_tensor)
            
            # Convert raw logits to probabilities using Softmax
            probabilities = F.softmax(logits, dim=1)
            
            # Get the highest probability and its corresponding class ID
            confidence, predicted_class = torch.max(probabilities, 1)
            
        return predicted_class.item(), confidence.item()

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Ensure this matches your Phase 2 saved model filename
    MODEL_PATH = '/home/st1/Documents/training_cnn_for_waveform/models/phase2_finetuned_best.pth'
    TARGET_DEVICE = 'cpu'
    
    # Initialize the inferencer
    inferencer = KeywordInferencer(
        model_path=MODEL_PATH, 
        num_classes=14, 
        device=TARGET_DEVICE
    )
    
    # Map your integer class IDs back to human-readable labels
    # Integers as keys, strings as values
    CLASS_MAP = {
        0: 'anoikse_fos_kanape',
        1: 'anoikse_fos_krevati',
        2: 'anoikse_fos_saloni',
        3: 'kleise_fos_kanape',
        4: 'kleise_fos_krevati',
        5: 'kleise_fos_saloni',
        6: 'kleise_musiki',
        7: 'kleise_ola',
        8: 'noise',
        9: 'paikse_musiki',
        10: 'xamilose_fos_kanape',
        11: 'xamilose_fos_krevati',
        12: 'xamilose_fos_saloni',
        13: 'google_speech'
    }
    
    test_audio_file = "/home/st1/Documents/training_cnn_for_waveform/data/need_cleaning_or_slided_forward_chunks/chunk_050.wav"
    
    if os.path.exists(test_audio_file):
        print(f"Analyzing {test_audio_file}...")

        st = time.time()
        
        pred_id, conf = inferencer.predict(test_audio_file)
        print(f"Time taken for inference is : {time.time()-st} seconds")
        
        if pred_id is not None:
            human_label = CLASS_MAP.get(pred_id, f"Class {pred_id}")
            print(f"\n--- Prediction Results ---")
            print(f"Predicted Class : {human_label} (ID: {pred_id})")
            print(f"Confidence      : {conf * 100:.2f}%")
            
            # Filtering logic example for your final application
            if pred_id == 12 or pred_id == 13:
                print("Action: IGNORED (Noise or Background Speech detected)")
            elif conf < 0.70:
                print("Action: IGNORED (Confidence too low)")
            else:
                print(f"Action: EXECUTING COMMAND for '{human_label}'")
    else:
        print(f"Test file not found: {test_audio_file}")