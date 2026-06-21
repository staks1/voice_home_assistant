import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import random
import os
import glob

class DynamicKeywordDataset(Dataset):
    def __init__(self, data_list, noise_dir=None, sample_rate=16000, max_seconds=3.0, is_training=True):
        """
        data_list: List of tuples -> [(filepath, label_integer), ...]
        noise_dir: Path to directory containing real environmental noise .wav files
        is_training: Boolean. If True, applies augmentations. If False, just pads and converts to Mel.
        """
        self.data_list = data_list
        self.sample_rate = sample_rate
        self.max_length = 53248 # steps calculated experimentally #int(sample_rate * max_seconds) # 3.0s * 16000 = 48000 samples
        self.is_training = is_training
        
        # Load real noise file paths if provided
        self.noise_files = glob.glob(os.path.join(noise_dir, "*.wav")) if noise_dir else []
        
        # Pre-initialize torchaudio PitchShifters to avoid recreating them every iteration
        # Steps: -2, -1, 1, 2 (0 is handled by not applying the transform)
        self.pitch_shifters = {
            step: T.PitchShift(sample_rate, n_steps=step) 
            for step in [-2, -1, 1, 2]
        }
        
        # Initialize MelSpectrogram transformer
        # TODO : check what values we need to use in the transformation
        # self.mel_transform = T.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_mels=64,
        #     n_fft=1024,
        #     hop_length=512
        # )


        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=64,
            n_fft=512,
            # win_length=512 is applied implicitly when omitted
            hop_length=160 # 69% overlap - 10ms step size for 16000 sample rate # or 256 for 50% overlap, we start with more detailed overlap 
        )

    def __len__(self):
        return len(self.data_list)
        
    def _pad_or_truncate(self, waveform):
        """Forces the 1D waveform to be exactly self.max_length samples."""
        length = waveform.shape[1]
        if length > self.max_length:
            # Truncate
            return waveform[:, :self.max_length]
        elif length < self.max_length:
            # Pad with silence (zeros) at the end
            pad_amount = self.max_length - length
            return torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def _apply_real_noise(self, waveform):
        """Overlays a random noise file scaled to a random amplitude factor."""
        if not self.noise_files:
            return waveform
            
        noise_path = random.choice(self.noise_files)
        noise_wave, _ = torchaudio.load(noise_path)
        
        # Ensure noise matches the target length
        noise_wave = self._pad_or_truncate(noise_wave)
        
        # Random scale factor between 0.1 and 0.5 (10% to 50% volume)
        scale_factor = random.uniform(0.1, 0.5)
        
        # Add and clamp to valid audio range [-1.0, 1.0]
        augmented = waveform + (noise_wave * scale_factor)
        return torch.clamp(augmented, -1.0, 1.0)

    def __getitem__(self, idx):


        # 1. Load Audio and the class label ground truth (assume it takes tuples of (path_to_wav,class))
        audio_path, label = self.data_list[idx]
        #print(audio_path)
        #print(label)
        waveform, _ = torchaudio.load(audio_path)
        #print(waveform)
        # detach so we dont have differentiation issues when stacking (collate fn)
        waveform = waveform.detach()
        
        
        # 2. Standardize Length (Crucial for CNN batching)
        waveform = self._pad_or_truncate(waveform)
        
        # 3. Apply Augmentations (Only during training)
        if self.is_training:

            # Pitch Shifting
            pitch_step = random.choice([-2, -1, 0, 1, 2])
            if pitch_step != 0:
                waveform = self.pitch_shifters[pitch_step](waveform)
                
            # White Noise (0.005 to 0.015)
            if random.random() < 0.5:
                noise_level = random.uniform(0.005, 0.015) # we can also test up to 0.02
                white_noise = torch.randn_like(waveform)
                waveform = torch.clamp(waveform + (white_noise * noise_level), -1.0, 1.0)
                
            # Real Environmental Noise Overlay
            if random.random() < 0.5:
                waveform = self._apply_real_noise(waveform)
                
            # Time Slicing / Roll (0.1 to 0.5)
            # CAUTION: Using only positive shifts to avoid wrapping the start of the audio to the end,
            # assuming your audio starts at 0.0s and has silence padding at the end.
            # TODO : we can also test ~0.6 and also potentially wrap around 
            if random.random() < 0.5:
                shift_pct = random.uniform(0.1, 0.5) 
                shift_samples = int(self.max_length * shift_pct)
                waveform = torch.roll(waveform, shifts=shift_samples, dims=1)

        waveform = waveform.detach()

        # convert labels to tensors for consistency with collate fn 
        # Convert labels to tensor
        label = torch.tensor(label, dtype=torch.long)
        

        #return waveform,label

        ## 4. Convert to Mel-Spectrogram 
        ## Final shape: [1, 64, ~94] (Channels, Mels, TimeFrames)
        mel_spec = self.mel_transform(waveform)
        
        # Convert to Decibels for better numerical stability
        mel_spec_db = T.AmplitudeToDB(stype='power', top_db=80)(mel_spec)

        # add normalization 
        # 2. Calculate the mean and standard deviation of this specific spectrogram
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()

        # 3. Apply Z-score normalization (avoiding division by zero)
        if std > 1e-6:
            mel_spec_normalized = (mel_spec_db - mean) / std
        else:
            mel_spec_normalized = mel_spec_db - mean
                
        return mel_spec_normalized, label

# ==========================================
# Example Initialization
# ==========================================
# data_list = [("path/to/yes_01.wav", 0), ("path/to/no_01.wav", 1), ...]
# train_dataset = DynamicKeywordDataset(data_list, noise_dir="path/to/noise", is_training=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)