import pandas as pd
import numpy as np
from datetime import datetime
from .helper import save_factor_data
import nltk
from nltk.tokenize import sent_tokenize
import re
from pathlib import Path
import typing
import os
from .helper import save_factor_data, get_video_path, get_audio_path, get_transcript_path

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

class CoherenceAnalyzer:
    """Analyzes text coherence using topic consistency and discourse markers."""
    
    def __init__(self, timestamp: str):
        self.coherence_metrics = {}
        self.timestamp = timestamp 
        self.discourse_markers = {
            'causal': ['because', 'therefore', 'thus', 'consequently', 'as a result', 'so'],
            'contrast': ['however', 'but', 'nevertheless', 'on the other hand', 'in contrast', 'yet'],
            'addition': ['moreover', 'furthermore', 'additionally', 'in addition', 'also', 'besides'],
            'temporal': ['first', 'second', 'finally', 'next', 'then', 'previously'],
            'exemplification': ['for example', 'for instance', 'such as', 'specifically', 'to illustrate']
        }

    def analyze_coherence(self, text: str) -> dict:
        """
        Analyze text coherence through multiple linguistic metrics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing coherence metrics
        """
        if not text.strip():
            return self._empty_metrics()
            
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return self._empty_metrics()
        
        # Calculate metrics
        metrics = {
            'topic_consistency': self._topic_consistency(sentences),
            'logical_flow': self._logical_flow(text, sentences),
            'sentence_metrics': self._sentence_analysis(sentences),
            'sentence_count': len(sentences)
        }
        
        # Calculate weighted overall score
        metrics['overall_coherence'] = round(
            0.4 * metrics['topic_consistency'] + 
            0.4 * metrics['logical_flow'] + 
            0.2 * metrics['sentence_metrics']['length_consistency'], 2
        )
        
        self.coherence_metrics = metrics
        return metrics

    def _empty_metrics(self) -> dict:
        """Return baseline metrics for empty/invalid input"""
        return {
            'topic_consistency': 0.0,
            'logical_flow': 0.0,
            'sentence_metrics': {
                'avg_length': 0.0,
                'length_consistency': 0.0,
                'min_length': 0,
                'max_length': 0
            },
            'overall_coherence': 0.0,
            'sentence_count': 0
        }

    def _topic_consistency(self, sentences: list) -> float:
        """Calculate Jaccard similarity between adjacent sentences."""
        tokens = [set(re.findall(r'\w+', s.lower())) for s in sentences]
        similarities = []
        
        for i in range(len(tokens)-1):
            intersect = len(tokens[i] & tokens[i+1])
            union = len(tokens[i] | tokens[i+1])
            similarities.append(intersect/union if union else 0)
            
        avg_sim = np.mean(similarities) if similarities else 0.0
        return min(1.0, avg_sim * 2.5)  # Scale low natural overlap

    def _logical_flow(self, text: str, sentences: list) -> float:
        """Analyze discourse marker usage patterns."""
        lower_text = text.lower()
        counts = {
            k: sum(1 for m in v if m in lower_text)
            for k, v in self.discourse_markers.items()
        }
        
        # Calculate diversity and density
        diversity = sum(1 for c in counts.values() if c > 0) / len(counts)
        density = sum(counts.values()) / (len(sentences) + 1e-6)  # Prevent division by zero
        
        return round(0.6 * diversity + 0.4 * min(density, 1.0), 2)

    def _sentence_analysis(self, sentences: list) -> dict:
        """Analyze sentence length distribution."""
        lengths = [len(re.findall(r'\w+', s)) for s in sentences]
        if not lengths:
            return self._empty_metrics()['sentence_metrics']
            
        avg = np.mean(lengths)
        std = np.std(lengths)
        
        return {
            'avg_length': round(avg, 1),
            'length_consistency': round(1 - min(std/avg, 1) if avg else 0, 2),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }

    def save_data(self, data: dict = None) -> str:
        """Save metrics to parquet format using helper function."""
        data = data or self.coherence_metrics
        df = pd.DataFrame([self._flatten_dict(data)])
        save_factor_data(df, 'coherence', self.timestamp)

    def _flatten_dict(self, data: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flatten nested dictionary structure."""
        items = []
        for k, v in data.items():
            key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, key, sep).items())
            else:
                items.append((key, v))
        return dict(items)

    def read_transcript(self):
        transcript_path = self.get_transcript_path()
        file = open(transcript_path, 'r')
        text = file.read()
        file.close()
        return text

    def analyze_and_save(self, text: str) -> dict:
        """Convenience method to analyze and save in one call."""
        text = self.read_transcript()
        metrics = self.analyze_coherence(text)
        self.save_data(metrics)
