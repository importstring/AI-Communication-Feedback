## Updated AI Model with Factor Integration ##
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Configuration matching your project structure
CONFIG = {
    "raw_data_dir": "/Users/simon/AI-Presentation-Feedback/data/recordings",
    "expert_scores_path": "/Users/simon/AI-Presentation-Feedback/data/expert_scores.csv",
    "output_dir": "results/",
    "modality_mapping": {
        'visual': ['body_language', 'emotion'],
        'audio': ['speech_patterns', 'tonal', 'volume'],
        'text': ['coherence', 'structure']
    }
}

class EnhancedFusion(nn.Module):
    """Updated to handle actual factor dimensions"""
    def __init__(self, input_dims):
        super().__init__()
        self.visual_dim = input_dims['visual']
        self.audio_dim = input_dims['audio']
        self.text_dim = input_dims['text']
        
        # Modality-specific projections
        self.visual_proj = nn.Linear(self.visual_dim, 512)
        self.audio_proj = nn.Linear(self.audio_dim, 512)
        self.text_proj = nn.Linear(self.text_dim, 512)
        
        # Temporal modeling (assuming 5-second clips at 30fps)
        self.temporal_gru = nn.GRU(input_size=512, hidden_size=256, 
                                 bidirectional=True, batch_first=True)
        
        # Multi-modal attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        
        # Final regression layer
        self.regressor = nn.Sequential(
            nn.Linear(512*3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Split into modalities
        visual = x[:, :self.visual_dim]
        audio = x[:, self.visual_dim:self.visual_dim+self.audio_dim]
        text = x[:, self.visual_dim+self.audio_dim:]
        
        # Project modalities
        v_proj = self.visual_proj(visual)
        a_proj = self.audio_proj(audio)
        t_proj = self.text_proj(text)
        
        # Temporal modeling
        v_temp, _ = self.temporal_gru(v_proj.unsqueeze(1))
        a_temp, _ = self.temporal_gru(a_proj.unsqueeze(1))
        
        # Cross-modal attention
        attn_out, _ = self.cross_attn(
            torch.cat([v_temp, a_temp], dim=1).transpose(0,1),
            t_proj.unsqueeze(1).transpose(0,1),
            t_proj.unsqueeze(1).transpose(0,1)
        )
        
        # Combine features
        combined = torch.cat([
            v_temp.mean(dim=1),
            a_temp.mean(dim=1),
            attn_out.squeeze().mean(dim=1)
        ], dim=1)
        
        return self.regressor(combined)

class FactorDataset(Dataset):
    """Handles timestamp-based factor loading"""
    def __init__(self, config):
        self.config = config
        self.timestamps = self._get_valid_timestamps()
        self.scaler = StandardScaler()
        self.feature_dim = self._calculate_feature_dim()
        self._preload_data()

    def _get_valid_timestamps(self):
        score_df = pd.read_csv(self.config['expert_scores_path'])
        valid_ts = []
        
        for ts in score_df['timestamp']:
            ts_path = os.path.join(self.config['raw_data_dir'], ts)
            if os.path.exists(ts_path):
                valid_ts.append(ts)
        
        return valid_ts

    def _calculate_feature_dim(self):
        sample_ts = self.timestamps[0]
        sample_features = self._load_timestamp_features(sample_ts)
        return sum(len(v) for v in sample_features.values())

    def _load_timestamp_features(self, timestamp):
        features = {'visual': [], 'audio': [], 'text': []}
        ts_dir = os.path.join(self.config['raw_data_dir'], timestamp)
        
        for modality, factors in self.config['modality_mapping'].items():
            for factor in factors:
                factor_path = os.path.join(ts_dir, f"{factor}.parquet")
                if os.path.exists(factor_path):
                    df = pd.read_parquet(factor_path)
                    features[modality].extend(df.values.flatten())
        
        return features

    def _preload_data(self):
        self.data = []
        self.scores = pd.read_csv(self.config['expert_scores_path'])
        
        # First pass to fit scaler
        all_features = []
        for ts in self.timestamps:
            features = self._load_timestamp_features(ts)
            combined = np.concatenate([features['visual'], features['audio'], features['text']])
            all_features.append(combined)
        
        self.scaler.fit(np.vstack(all_features))

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        ts = self.timestamps[idx]
        features = self._load_timestamp_features(ts)
        combined = np.concatenate([features['visual'], features['audio'], features['text']])
        scaled = self.scaler.transform(combined.reshape(1, -1))
        
        score = self.scores[self.scores['timestamp'] == ts]['score'].values[0]
        return torch.FloatTensor(scaled).squeeze(), torch.FloatTensor([score])

def main():
    # Initialize dataset
    dataset = FactorDataset(CONFIG)
    
    # Calculate input dimensions for model initialization
    input_dims = {
        'visual': len(dataset._load_timestamp_features(dataset.timestamps[0])['visual']),
        'audio': len(dataset._load_timestamp_features(dataset.timestamps[0])['audio']),
        'text': len(dataset._load_timestamp_features(dataset.timestamps[0])['text'])
    }
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedFusion(input_dims).to(device)
    
    # Training setup
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.HuberLoss()
    
    # Training loop
    for epoch in range(15):
        model.train()
        epoch_loss = 0
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_features, val_targets = next(iter(DataLoader(dataset, batch_size=len(dataset))))
            predictions = model(val_features.to(device))
            correlation = pearsonr(predictions.cpu().numpy().flatten(), 
                                 val_targets.numpy().flatten())[0]
            
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Correlation: {correlation:.2f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'presentation_feedback_model.pth'))

if __name__ == "__main__":
    main()
