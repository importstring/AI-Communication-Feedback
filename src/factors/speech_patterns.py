import pandas as pd
import numpy as np
from datetime import datetime
from .helper import save_factor_data
import re
from collections import Counter

class SpeechPatternAnalyzer:
    """Analyzes speech patterns like pace, pauses, and verbal tics."""
    
    def __init__(self):
        self.patterns = {}
        self.metrics = {}
        self.filler_words = {
            'um', 'uh', 'like', 'you know', 'so', 'actually', 
            'basically', 'right', 'I mean', 'well', 'kind of'
        }
    
    def analyze_speech_patterns(self, text, audio=None, audio_duration=None, pauses=None):
        """
        Analyze speech patterns from text and optional audio.
        
        Args:
            text: Transcribed text to analyze
            audio: Optional audio data for timing analysis
            audio_duration: Length of audio in seconds
            pauses: List of pause durations
            
        Returns:
            Dictionary with speech pattern metrics
        """
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        if audio_duration:
            speaking_rate_wpm = len(words) / (audio_duration / 60) if audio_duration > 0 else 0
            speaking_rate_wps = len(words) / audio_duration if audio_duration > 0 else 0
        else:
            speaking_rate_wpm = len(words) / (len(text) / 1000)
            speaking_rate_wps = speaking_rate_wpm / 60
        
        filler_word_counts = {
            word: sum(1 for w in words if w == word) 
            for word in self.filler_words if word in words
        }
        filler_count = sum(filler_word_counts.values())
        
        pause_analysis = {}
        if pauses:
            total_pause = sum(pauses)
            avg_pause = total_pause / len(pauses) if pauses else 0
            pause_analysis = {
                'total_pause_seconds': round(total_pause, 2),
                'average_pause_duration': round(avg_pause, 2),
                'pause_count': len(pauses),
                'longest_pause': round(max(pauses), 2) if pauses else 0
            }
        
        if pauses and len(words) > 0:
            filler_ratio = filler_count / len(words)
            pause_ratio = total_pause / audio_duration if audio_duration else 0
            fluency_score = 1.0 - (0.5 * filler_ratio + 0.5 * min(1.0, pause_ratio))
        else:
            filler_ratio = filler_count / len(words) if len(words) > 0 else 0
            fluency_score = 1.0 - min(1.0, filler_ratio)
        
        self.metrics = {
            'words_per_minute': round(speaking_rate_wpm, 2),
            'words_per_second': round(speaking_rate_wps, 2),
            'word_count': len(words),
            'top_words': dict(word_freq.most_common(10)),
            'filler_word_frequency': filler_word_counts,
            'total_filler_words': filler_count,
            'filler_word_ratio': round(filler_count / len(words), 4) if len(words) > 0 else 0,
            'pause_analysis': pause_analysis,
            'speech_fluency_score': round(fluency_score, 2)
        }
        
        return self.metrics
    
    def save_data(self, data=None):
        """
        Save speech pattern analysis to a parquet file.
        
        Args:
            data: Optional data to save, uses self.metrics if None
            
        Returns:
            Path to the saved file
        """
        if data is None:
            data = self.metrics
            
        if isinstance(data, dict):
            flat_data = self._flatten_dict(data)
            data = pd.DataFrame([flat_data])
            
        return save_factor_data(data, 'speech_patterns')
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary for DataFrame conversion"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def analyze_and_save(self, text, audio=None, audio_duration=None, pauses=None):
        """
        Analyze speech patterns and save results.
        
        Args:
            text: Text to analyze
            audio: Optional audio data
            audio_duration: Length of audio in seconds
            pauses: List of pause durations
            
        Returns:
            Dictionary with speech pattern metrics
        """
        results = self.analyze_speech_patterns(text, audio, audio_duration, pauses)
        save_path = self.save_data(results)
        
        results['save_path'] = save_path
        
        return results
