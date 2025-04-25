import pandas as pd
import numpy as np
from datetime import datetime
from .helper import save_factor_data
import nltk
from nltk.tokenize import sent_tokenize
import re

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class CoherenceAnalyzer:
    """Analyzes the coherence of presentation content and speech."""
    
    def __init__(self):
        self.coherence_score = None
        self.coherence_metrics = {}
        self.discourse_markers = {
            'causal': ['because', 'therefore', 'thus', 'consequently', 'as a result', 'so'],
            'contrast': ['however', 'but', 'nevertheless', 'on the other hand', 'in contrast', 'yet'],
            'addition': ['moreover', 'furthermore', 'additionally', 'in addition', 'also', 'besides'],
            'temporal': ['first', 'second', 'finally', 'next', 'then', 'previously'],
            'exemplification': ['for example', 'for instance', 'such as', 'specifically', 'to illustrate']
        }
    
    def analyze_coherence(self, text):
        """
        Analyze text coherence by measuring topic consistency and flow.
        
        Args:
            text: Transcribed text of the presentation
            
        Returns:
            Dictionary with coherence metrics
        """
        if not text or len(text.strip()) == 0:
            return self._empty_metrics()
            
        # Split text into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return self._empty_metrics()
        
        # Analyze topic consistency
        topic_consistency = self._analyze_topic_consistency(sentences)
        
        # Analyze logical flow using discourse markers
        logical_flow = self._analyze_logical_flow(text, sentences)
        
        # Analyze sentence length distribution for readability
        sentence_metrics = self._analyze_sentences(sentences)
        
        # Calculate overall coherence score
        overall_coherence = (
            0.4 * topic_consistency + 
            0.4 * logical_flow + 
            0.2 * sentence_metrics['sentence_length_consistency']
        )
        
        # Store the results
        self.coherence_metrics = {
            'topic_consistency': round(topic_consistency, 2),
            'logical_flow': round(logical_flow, 2),
            'sentence_metrics': sentence_metrics,
            'overall_coherence': round(overall_coherence, 2),
            'sentence_count': len(sentences)
        }
        
        self.coherence_score = overall_coherence
        
        return self.coherence_metrics
    
    def _empty_metrics(self):
        """Return empty metrics when text is insufficient"""
        return {
            'topic_consistency': 0.0,
            'logical_flow': 0.0,
            'sentence_metrics': {
                'avg_sentence_length': 0.0,
                'sentence_length_consistency': 0.0,
                'min_sentence_length': 0,
                'max_sentence_length': 0
            },
            'overall_coherence': 0.0,
            'sentence_count': 0
        }
    
    def _analyze_topic_consistency(self, sentences):
        """Analyze how consistently topics are maintained across sentences"""
        # Simple bag-of-words approach
        word_sets = [set(re.findall(r'\b\w+\b', sentence.lower())) for sentence in sentences]
        
        # Calculate overlap between adjacent sentences
        overlaps = []
        for i in range(len(word_sets) - 1):
            if not word_sets[i] or not word_sets[i+1]:
                continue
                
            # Jaccard similarity between adjacent sentences
            intersection = len(word_sets[i] & word_sets[i+1])
            union = len(word_sets[i] | word_sets[i+1])
            overlaps.append(intersection / union if union > 0 else 0)
        
        # Calculate average overlap as a measure of topic consistency
        avg_overlap = np.mean(overlaps) if overlaps else 0.0
        
        # Scale to 0-1 where higher is better
        return min(1.0, avg_overlap * 2.5)  # Scale up since natural overlap is often low
    
    def _analyze_logical_flow(self, text, sentences):
        """Analyze the logical flow using discourse markers"""
        # Count discourse markers by type
        marker_counts = {
            marker_type: sum(1 for marker in markers if marker in text.lower())
            for marker_type, markers in self.discourse_markers.items()
        }
        
        # Calculate marker diversity (0-1)
        total_marker_types = len(self.discourse_markers)
        types_used = sum(1 for count in marker_counts.values() if count > 0)
        marker_diversity = types_used / total_marker_types if total_marker_types > 0 else 0
        
        # Calculate marker density relative to sentence count
        total_markers = sum(marker_counts.values())
        marker_density = min(1.0, total_markers / (len(sentences) * 0.8))
        
        # Combine diversity and density for logical flow score
        return 0.6 * marker_diversity + 0.4 * marker_density
    
    def _analyze_sentences(self, sentences):
        """Analyze sentence length and structure"""
        # Calculate lengths
        lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        
        if not lengths:
            return {
                'avg_sentence_length': 0.0,
                'sentence_length_consistency': 0.0,
                'min_sentence_length': 0,
                'max_sentence_length': 0
            }
        
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # Calculate consistency (lower std_dev/mean ratio is more consistent)
        length_consistency = 1.0 - min(1.0, (std_length / avg_length) if avg_length > 0 else 0)
        
        return {
            'avg_sentence_length': round(avg_length, 1),
            'sentence_length_consistency': round(length_consistency, 2),
            'min_sentence_length': min(lengths),
            'max_sentence_length': max(lengths)
        }
    
    def save_data(self, data=None):
        """
        Save coherence analysis data to a parquet file.
        
        Args:
            data: Optional data to save, uses self.coherence_metrics if None
            
        Returns:
            Path to the saved file
        """
        # Use class data if none provided
        if data is None:
            data = self.coherence_metrics
            
        # Convert to DataFrame if it's a dict
        if isinstance(data, dict):
            # Flatten nested dictionary
            flat_data = self._flatten_dict(data)
            data = pd.DataFrame([flat_data])
            
        return save_factor_data(data, 'coherence')
    
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
    
    def analyze_and_save(self, text):
        """
        Analyze coherence and save results.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with coherence metrics
        """
        results = self.analyze_coherence(text)
        save_path = self.save_data(results)
        
        # Add save path to results
        results['save_path'] = save_path
        
        return results
