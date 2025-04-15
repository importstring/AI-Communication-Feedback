import re
from typing import Dict, List, Optional, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class Structure:
    """
    Analyzes the structure and organization of a presentation or speech.
    """
    def __init__(self):
        self.transitions = [
            'first', 'second', 'third', 'finally',
            'in addition', 'furthermore', 'moreover',
            'however', 'on the other hand', 'nevertheless',
            'therefore', 'consequently', 'as a result',
            'in conclusion', 'to summarize', 'in summary'
        ]
        self.closing_techniques = [
            'summary', 'call_to_action', 'future_outlook',
            'circle_back_to_opening'
        ]
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_structure(self, message: str) -> Dict:
        """
        Analyzes the structure of a presentation or speech.
        
        Args:
            message: The text content to analyze
            
        Returns:
            dict: Dictionary containing structure analysis results
        """
        # Split message into sentences
        sentences = sent_tokenize(message)
        
        # Analyze opening
        opening_score = self._analyze_opening(sentences[0] if sentences else "")
        
        # Analyze transitions
        transition_metrics = self._analyze_transitions(message)
        
        # Analyze closing
        closing_text = sentences[-1] if sentences else ""
        closing_technique = self._identify_closing_technique(closing_text)
        closing_score = self._score_closing(closing_text, closing_technique)
        
        # Analyze information hierarchy
        hierarchy_metrics = self._analyze_information_hierarchy(message)
        
        # Analyze coherence
        coherence_score = self._analyze_coherence(message)
        
        # Calculate overall structure score
        structure_score = self._calculate_structure_score(
            opening_score, 
            transition_metrics['score'],
            closing_score,
            coherence_score
        )
        
        return {
            'opening_score': opening_score,
            'transition_metrics': transition_metrics,
            'closing_technique': closing_technique,
            'closing_score': closing_score,
            'hierarchy_metrics': hierarchy_metrics,
            'coherence_score': coherence_score,
            'structure_score': structure_score
        }
    
    def _analyze_opening(self, opening_text: str) -> float:
        """Analyze the effectiveness of the opening"""
        if not opening_text:
            return 0.0
            
        # Check for attention-grabbing elements
        attention_indicators = [
            r'[!?]',  # Exclamation or question marks
            r'\b(imagine|picture|consider)\b',  # Engaging verbs
            r'\b(shocking|surprising|amazing)\b'  # Impact words
        ]
        
        attention_score = sum(
            1 for pattern in attention_indicators 
            if re.search(pattern, opening_text, re.IGNORECASE)
        ) / len(attention_indicators)
        
        # Check for clear purpose statement
        purpose_indicators = [
            r'\b(today|in this|we will|I will)\b',
            r'\b(present|discuss|explore|examine)\b'
        ]
        
        purpose_score = sum(
            1 for pattern in purpose_indicators 
            if re.search(pattern, opening_text, re.IGNORECASE)
        ) / len(purpose_indicators)
        
        return 0.6 * attention_score + 0.4 * purpose_score
    
    def _analyze_transitions(self, message: str) -> Dict:
        """Analyze the use of transitions"""
        # Count transitions
        transitions_found = sum(
            1 for transition in self.transitions 
            if transition.lower() in message.lower()
        )
        
        # Calculate transition density
        total_words = len(word_tokenize(message))
        total_transitions = len(self.transitions)
        transition_density = transitions_found / total_words if total_words > 0 else 0
        
        # Calculate transition diversity
        used_types = sum(1 for transition in self.transitions if transition.lower() in message.lower())
        diversity = used_types / len(self.transitions)
        
        # Calculate transition score
        score = 0.5 * min(1.0, transition_density * 2) + 0.5 * diversity
        
        return {
            'transitions_found': transitions_found,
            'total_transitions': total_transitions,
            'transition_density': transition_density,
            'diversity': diversity,
            'score': score
        }
    
    def _identify_closing_technique(self, closing_text: str) -> str:
        """Identify which closing technique is being used"""
        for technique in self.closing_techniques:
            if self._check_closing_technique(closing_text, technique):
                return technique
        return 'unidentified'
    
    def _check_closing_technique(self, text: str, technique: str) -> bool:
        """Check if text uses a specific closing technique"""
        technique_markers = {
            'summary': ['in summary', 'to summarize', 'in conclusion'],
            'call_to_action': ['should', 'must', 'take action', 'recommend'],
            'future_outlook': ['future', 'next', 'going forward', 'will'],
            'circle_back_to_opening': []  # This requires comparing to opening
        }
        
        markers = technique_markers.get(technique, [])
        for marker in markers:
            if marker.lower() in text.lower():
                return True
        return False
    
    def _score_closing(self, closing_text: str, technique: str) -> float:
        """Score the effectiveness of the closing"""
        if technique == 'unidentified':
            return 0.5
        
        length_score = min(1.0, len(closing_text.split()) / 30)
        technique_score = 0.8 if technique in self.closing_techniques else 0.5
        
        return 0.4 * length_score + 0.6 * technique_score
    
    def _analyze_information_hierarchy(self, message: str) -> Dict:
        """Analyze information hierarchy and organization"""
        sentences = sent_tokenize(message)
        words = word_tokenize(message)
        
        # Check for main points
        main_point_indicators = ['first', 'second', 'third', 'finally']
        main_points_found = sum(1 for word in words if word.lower() in main_point_indicators)
        
        # Check for supporting details
        detail_indicators = ['for example', 'specifically', 'in particular']
        details_found = sum(1 for sentence in sentences 
                          if any(indicator in sentence.lower() 
                                for indicator in detail_indicators))
        
        # Calculate scores
        main_points_score = min(1.0, main_points_found / 3)  # Expect at least 3 main points
        details_score = min(1.0, details_found / len(sentences))
        
        return {
            'main_points_score': main_points_score,
            'details_score': details_score,
            'score': 0.6 * main_points_score + 0.4 * details_score
        }
    
    def _analyze_coherence(self, message: str) -> float:
        """Analyze coherence of the message"""
        sentences = sent_tokenize(message)
        if len(sentences) < 2:
            return 0.5
            
        # Calculate sentence length consistency
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        length_std = np.std(lengths)
        length_score = 1.0 - min(1.0, length_std / 10)  # Normalize to 0-1
        
        # Calculate topic consistency
        topic_words = [word for word in word_tokenize(message) 
                      if word.lower() not in self.stop_words]
        unique_topics = len(set(topic_words))
        topic_score = min(1.0, unique_topics / len(topic_words))
        
        return 0.4 * length_score + 0.6 * topic_score
    
    def _calculate_structure_score(self, opening_score: float, 
                                 transition_score: float, 
                                 closing_score: float, 
                                 coherence_score: float) -> float:
        """Calculate overall structure score"""
        return 0.2 * opening_score + 0.3 * transition_score + \
               0.2 * closing_score + 0.3 * coherence_score 