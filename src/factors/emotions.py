from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import torchaudio
import librosa
from transformers import pipeline
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.nn as nn
import torch

class MultimodalEmotionAnalyzer:
    def __init__(self):
        self.text_analyzer = pipeline("text-classification", 
                                    model="j-hartmann/emotion-english-distilroberta-base",
                                    top_k=None)
        
        self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        
        self.audio_projection = nn.Linear(256, 7)  

    def _map_acoustic_features(self, hidden_states):
        """Implements temporal attention pooling with learnable weights"""
        attention_weights = torch.softmax(self.attention_mlp(hidden_states.mean(dim=1)), dim=-1)
        weighted_embedding = torch.matmul(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return torch.softmax(self.audio_projection(weighted_embedding), dim=-1)
