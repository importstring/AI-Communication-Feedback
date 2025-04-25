from transformers import AutoProcessor, AutoModelForAudioClassification
import torch
import torchaudio
import librosa
from transformers import pipeline
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.nn as nn
from typing import Dict, Optional, Union
import numpy as np

class MultimodalEmotionAnalyzer:
    def __init__(self, text_weight=0.7):
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
        self.attention_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.text_weight = text_weight

    def _map_acoustic_features(self, hidden_states):
        """Implements temporal attention pooling with learnable weights"""
        attention_weights = torch.softmax(self.attention_mlp(hidden_states.mean(dim=1)), dim=-1)
        weighted_embedding = torch.matmul(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return torch.softmax(self.audio_projection(weighted_embedding), dim=-1)
        
    def analyze(self, text, audio):
        """
        Analyze both text and audio to determine emotions
        
        Args:
            text: Text content to analyze
            audio: Audio data to analyze
            
        Returns:
            Dictionary containing combined emotion analysis
        """
        text_emotions = self._analyze_text(text)
        audio_emotions = self._analyze_audio(audio)
        return self.combine_emotions(text_emotions, audio_emotions)
        
    def _analyze_text(self, text):
        """Analyze emotions from text"""
        emotion_results = self.text_analyzer(text)
        
        emotions = {}
        for result in emotion_results:
            label = result['label']
            score = result['score']
            emotions[label] = score
            
        return emotions
        
    def _analyze_audio(self, audio):
        """Analyze emotions from audio"""
        # Process audio to match model requirements
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=16000)
        
        # Ensure sample rate is 16kHz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Extract features
        inputs = self.audio_processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
        
        # Process outputs
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Map to emotion classes
        emotion_classes = [
            "neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"
        ]
        
        # Extract and organize results
        emotions = {}
        for i, emotion in enumerate(emotion_classes):
            emotions[emotion] = float(probabilities[i])
            
        return emotions
        
    def combine_emotions(self, text_emotion, audio_emotion):
        """
        Combine text and audio emotion analyses with weighting
        
        Args:
            text_emotion: Dictionary of text-based emotion scores
            audio_emotion: Dictionary of audio-based emotion scores
            
        Returns:
            Dictionary with combined emotion analysis
        """
        if isinstance(text_emotion, dict) and isinstance(audio_emotion, dict):
            combined = {}
            
            # Get all unique emotions
            all_emotions = set(list(text_emotion.keys()) + list(audio_emotion.keys()))
            
            # Combine with weights
            for emotion in all_emotions:
                text_score = text_emotion.get(emotion, 0.0)
                audio_score = audio_emotion.get(emotion, 0.0)
                combined[emotion] = text_score * self.text_weight + audio_score * (1 - self.text_weight)
            
            # Find primary emotion
            primary_emotion = max(combined, key=combined.get)
            confidence = combined[primary_emotion]
            
            return {
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'all_emotions': combined
            }
        
        # Fallback if one of the inputs is not a dictionary
        if self.text_weight >= 0.5:
            return text_emotion
        else:
            return audio_emotion

class Emotion:
    """
    Detects and classifies emotions from text or audio data.
    """
    def __init__(self):
        self.emotion_analyzer = MultimodalEmotionAnalyzer()
        self.text_emotion_classifier = pipeline(
            "text-classification", 
            model="arpanghoshal/EmoRoberta"
        )
        self.audio_classifier = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        self.emotion = None

    def detect_emotion(self, text: Optional[str] = None, 
                      audio: Optional[Union[str, np.ndarray]] = None) -> Dict:
        """
        Uses sentiment analysis (text) or audio processing (tone)
        to classify emotions.
        
        Args:
            text: Optional text content to analyze
            audio: Optional audio data to analyze
            
        Returns:
            dict: Dictionary containing emotion analysis results
        """
        results = {}
        
        # Use imported MultimodalEmotionAnalyzer if both text and audio are provided
        if text is not None and audio is not None:
            multimodal_result = self.emotion_analyzer.analyze(text=text, audio=audio)
            results['multimodal'] = multimodal_result
        
        # Text-based emotion analysis
        if text is not None:
            text_emotions = self._analyze_text_emotion(text)
            results['text'] = text_emotions
        
        # Audio-based emotion analysis
        if audio is not None:
            audio_emotions = self._analyze_audio_emotion(audio)
            results['audio'] = audio_emotions
        
        # Combine results if both modalities were analyzed
        if 'text' in results and 'audio' in results:
            combined = self._combine_emotion_analyses(results['text'], results['audio'])
            results['combined'] = combined
            self.emotion = combined['primary_emotion']
        elif 'text' in results:
            self.emotion = results['text']['primary_emotion']
        elif 'audio' in results:
            self.emotion = results['audio']['primary_emotion']
        else:
            self.emotion = None
            
        return results
    
    def _analyze_text_emotion(self, text: str) -> Dict:
        """Analyze emotion from text"""
        # Use the Hugging Face emotion classification pipeline
        emotion_results = self.text_emotion_classifier(text)
        
        # Extract and organize results
        emotions = {}
        primary_emotion = None
        max_score = 0
        
        for result in emotion_results:
            label = result['label']
            score = result['score']
            emotions[label] = score
            
            if score > max_score:
                max_score = score
                primary_emotion = label
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': max_score,
            'all_emotions': emotions
        }
    
    def _analyze_audio_emotion(self, audio: Union[str, np.ndarray]) -> Dict:
        """Analyze emotion from audio"""
        # Process audio to match model requirements
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=16000)
        
        # Ensure sample rate is 16kHz
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Extract features
        inputs = self.audio_feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.audio_classifier(**inputs)
        
        # Process outputs
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Map to emotion classes
        emotion_classes = [
            "neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"
        ]
        
        # Extract and organize results
        emotions = {}
        for i, emotion in enumerate(emotion_classes):
            emotions[emotion] = float(probabilities[i])
        
        # Find primary emotion
        primary_emotion = emotion_classes[torch.argmax(probabilities).item()]
        confidence = float(torch.max(probabilities))
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'all_emotions': emotions
        }
    
    def _combine_emotion_analyses(self, text_emotions: Dict, 
                                audio_emotions: Dict) -> Dict:
        """Combine text and audio emotion analyses"""
        # Simple weighted combination
        combined_emotions = {}
        
        # Get all unique emotions
        all_emotions = set(list(text_emotions['all_emotions'].keys()) + 
                          list(audio_emotions['all_emotions'].keys()))
        
        # Combine with weights (text: 0.6, audio: 0.4)
        for emotion in all_emotions:
            text_score = text_emotions['all_emotions'].get(emotion, 0.0)
            audio_score = audio_emotions['all_emotions'].get(emotion, 0.0)
            combined_emotions[emotion] = 0.6 * text_score + 0.4 * audio_score
        
        # Find primary emotion
        primary_emotion = max(combined_emotions, key=combined_emotions.get)
        confidence = combined_emotions[primary_emotion]
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'all_emotions': combined_emotions
        }

    def save_data(self, emotion_data):
        """
        Save emotion analysis data to a parquet file
        
        Args:
            emotion_data: Dictionary containing emotion analysis results
            
        Returns:
            Path to the saved file
        """
        from .helper import save_factor_data
        return save_factor_data(emotion_data, 'emotions')
    
    def analyze_and_save(self, text=None, audio=None):
        """
        Analyze emotions from text/audio and save the results
        
        Args:
            text: Optional text to analyze
            audio: Optional audio to analyze
            
        Returns:
            Dictionary with analysis results and save path
        """
        # Analyze emotions
        results = self.detect_emotion(text=text, audio=audio)
        
        # Save results
        save_path = self.save_data(results)
        
        # Add save path to results
        results['save_path'] = save_path
        
        return results
