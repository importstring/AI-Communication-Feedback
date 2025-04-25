import pandas as pd
import numpy as np
from datetime import datetime
from .helper import save_factor_data
import re

class EthicsAnalyzer:
    """Analyzes ethical aspects of presentation content."""
    
    def __init__(self):
        self.ethics_score = None
        self.ethics_metrics = {}
        
        # Define term sets for different ethical aspects
        self.bias_terms = {
            'always', 'never', 'all', 'every', 'none', 'dominate', 'superior', 
            'inferior', 'better than', 'worse than', 'only', 'best', 'worst',
            'perfect', 'impossible', 'entirely', 'totally', 'completely',
            'obviously', 'clearly', 'undoubtedly', 'certainly', 'absolutely'
        }
        
        self.inclusivity_terms = {
            'diverse', 'diversity', 'inclusive', 'inclusivity', 'equitable', 
            'equity', 'accessible', 'accessibility', 'representation',
            'equal', 'equality', 'fair', 'fairness', 'respectful', 'respect',
            'different perspectives', 'various viewpoints', 'all backgrounds'
        }
        
        self.potentially_offensive_terms = {
            'hate', 'discriminate', 'prejudice', 'stereotype', 'bias',
            'offensive', 'inappropriate', 'insensitive', 'divisive'
        }
        
        # Define more specific problematic terms
        # This would typically include sensitive or offensive language
        self.problematic_terms = {
            # This is just a placeholder - in a real implementation,
            # this would include actual problematic terms
            'offensive_example': ['placeholder']
        }
    
    def analyze_ethics(self, text):
        """
        Analyze text for ethical considerations.
        
        Args:
            text: Transcribed text of the presentation
            
        Returns:
            Dictionary with ethics metrics
        """
        if not text or len(text.strip()) == 0:
            return self._empty_metrics()
            
        text_lower = text.lower()
        
        # Measure absolutist language (potential bias indicator)
        bias_count = sum(self._count_term_usage(text_lower, term) for term in self.bias_terms)
        bias_score = self._calculate_bias_score(bias_count, len(text.split()))
        
        # Measure inclusive language
        inclusivity_count = sum(self._count_term_usage(text_lower, term) for term in self.inclusivity_terms)
        inclusivity_score = self._calculate_inclusivity_score(inclusivity_count, len(text.split()))
        
        # Check for potentially offensive or problematic language
        offensive_terms = [term for term in self.potentially_offensive_terms 
                          if self._count_term_usage(text_lower, term) > 0]
        
        # Check for specific problematic terms (this would be expanded in a real implementation)
        problematic_terms_found = []
        
        # Calculate overall ethics score (higher is better)
        overall_ethics_score = 0.5 + (0.3 * inclusivity_score - 0.2 * bias_score)
        # Clamp between 0 and 1
        overall_ethics_score = max(0.0, min(1.0, overall_ethics_score))
        
        # Store results
        self.ethics_metrics = {
            'bias_score': round(bias_score, 2),  # Lower is better
            'inclusivity_score': round(inclusivity_score, 2),  # Higher is better
            'offensive_language_detected': len(offensive_terms) > 0,
            'offensive_terms': offensive_terms,
            'problematic_terms': problematic_terms_found,
            'overall_ethics_score': round(overall_ethics_score, 2)  # Higher is better
        }
        
        self.ethics_score = overall_ethics_score
        
        return self.ethics_metrics
    
    def _empty_metrics(self):
        """Return empty metrics when text is insufficient"""
        return {
            'bias_score': 0.0,
            'inclusivity_score': 0.0,
            'offensive_language_detected': False,
            'offensive_terms': [],
            'problematic_terms': [],
            'overall_ethics_score': 0.5  # Neutral score for empty text
        }
    
    def _count_term_usage(self, text, term):
        """Count occurrences of a term in text, handling word boundaries"""
        if ' ' in term:  # For multi-word terms
            return text.count(term)
        else:  # For single words, use regex with word boundaries
            return len(re.findall(r'\b' + re.escape(term) + r'\b', text))
    
    def _calculate_bias_score(self, bias_count, word_count):
        """Calculate bias score based on absolutist language frequency"""
        if word_count == 0:
            return 0.0
        
        # Normalize by text length and scale to 0-1 (higher score = more biased)
        normalized_count = bias_count / (word_count / 100)  # Per 100 words
        return min(1.0, normalized_count / 5)  # Assume 5+ per 100 words is very biased
    
    def _calculate_inclusivity_score(self, inclusivity_count, word_count):
        """Calculate inclusivity score based on inclusive language frequency"""
        if word_count == 0:
            return 0.0
        
        # Normalize by text length and scale to 0-1 (higher score = more inclusive)
        normalized_count = inclusivity_count / (word_count / 100)  # Per 100 words
        return min(1.0, normalized_count / 3)  # Assume 3+ per 100 words is very inclusive
    
    def save_data(self, data=None):
        """
        Save ethics analysis data to a parquet file.
        
        Args:
            data: Optional data to save, uses self.ethics_metrics if None
            
        Returns:
            Path to the saved file
        """
        # Use class data if none provided
        if data is None:
            data = self.ethics_metrics
            
        # Convert to DataFrame if it's a dict
        if isinstance(data, dict):
            # Handle list values for DataFrame conversion
            processed_data = data.copy()
            for key, value in processed_data.items():
                if isinstance(value, list):
                    processed_data[key] = ','.join(value) if value else ''
            
            data = pd.DataFrame([processed_data])
            
        return save_factor_data(data, 'ethics')
    
    def analyze_and_save(self, text):
        """
        Analyze ethics and save results.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with ethics metrics
        """
        results = self.analyze_ethics(text)
        save_path = self.save_data(results)
        
        # Add save path to results
        results['save_path'] = save_path
        
        return results
