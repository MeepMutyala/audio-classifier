import torch
import torch.nn as nn
from preprocessing.audio_utils import ESC50Preprocessor

class MelToTubeletProcessor:
    def __init__(self, tubelet_size=16, overlap=8):
        self.tubelet_size = tubelet_size
        self.overlap = overlap
        
    def process_mel_to_tubelets(self, mel_spec):
        """
        Convert mel-spectrogram to 3D tubelets for V-JEPA2
        Args:
            mel_spec: [time_steps, n_mels]
        Returns:
            tubelets: [1, num_tubelets, tubelet_size, n_mels] 
        """
        time_steps, n_mels = mel_spec.shape
        
        # Create overlapping windows
        stride = self.tubelet_size - self.overlap
        num_tubelets = (time_steps - self.tubelet_size) // stride + 1
        
        tubelets = []
        for i in range(num_tubelets):
            start = i * stride
            end = start + self.tubelet_size
            if end <= time_steps:
                tubelet = mel_spec[start:end, :]  # [tubelet_size, n_mels]
                tubelets.append(tubelet)
        
        if tubelets:
            tubelets = torch.stack(tubelets)  # [num_tubelets, tubelet_size, n_mels]
            # Reshape to "video-like" format: [1, num_tubelets, tubelet_size, n_mels]
            tubelets = tubelets.unsqueeze(0).permute(0, 2, 1, 3)  # [1, tubelet_size, num_tubelets, n_mels]
        else:
            # Fallback for very short sequences
            tubelets = mel_spec.unsqueeze(0).unsqueeze(0)  # [1, 1, time_steps, n_mels]
        
        return tubelets

class VJEPA2AudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, esc50_path, preprocessor=None, tubelet_processor=None):
        self.df = dataframe.reset_index(drop=True)
        self.esc50_path = Path(esc50_path)
        self.preprocessor = preprocessor or ESC50Preprocessor()
        self.tubelet_processor = tubelet_processor or MelToTubeletProcessor()
        
        # Create class mapping
        self.classes = sorted(self.df['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess audio to mel-spectrogram
        audio_path = self.esc50_path / "audio" / row['filename']
        mel_spec = self.preprocessor.load_and_preprocess(audio_path)
        
        # Convert to tubelets
        tubelets = self.tubelet_processor.process_mel_to_tubelets(mel_spec)
        
        # Get label
        label = self.class_to_idx[row['category']]
        
        return tubelets, torch.tensor(label, dtype=torch.long)
