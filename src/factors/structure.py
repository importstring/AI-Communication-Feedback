import re
from typing import Dict, List, Optional, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import glob
import os
from datetime import datetime
import numpy as np
from difflib import SequenceMatcher

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class Structure:
    """
    Analyzes the structure and organization of a presentation or speech.
    """
    def __init__(self, timestamp: str = None):
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
        self.base_audio_path = '../data/recordings/audio'
        self.opening_text = ""  # Store opening text for circle-back comparison

        self.timestamp = timestamp
        
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
        
        # Store opening for later comparison
        if sentences:
            # Consider the first 1-2 sentences as the opening
            opening_length = min(2, len(sentences))
            self.opening_text = ' '.join(sentences[:opening_length])
        
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
        if technique == 'circle_back_to_opening':
            # Compare with opening text
            if hasattr(self, 'opening_text') and self.opening_text:
                return self._check_circle_back(text, self.opening_text)
            return False
            
        technique_markers = {
            'summary': ['in summary', 'to summarize', 'in conclusion', 'to conclude', 'wrapping up', 'to sum up'],
            'call_to_action': ['should', 'must', 'take action', 'recommend', 'urge', 'encourage', 'call upon'],
            'future_outlook': ['future', 'next', 'going forward', 'will', 'plan', 'vision', 'look ahead']
        }
        
        markers = technique_markers.get(technique, [])
        for marker in markers:
            if marker.lower() in text.lower():
                return True
        return False
        
    def _check_circle_back(self, closing_text: str, opening_text: str) -> bool:
        """Check if closing text circles back to the opening"""
        # Extract key phrases (first and last few words)
        opening_words = opening_text.lower().split()[:5]
        closing_words = closing_text.lower().split()[-5:]
        
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, opening_words, closing_words).ratio()
        
        # Also check for thematic words from opening appearing in closing
        opening_themes = set([w for w in opening_text.lower().split() 
                           if len(w) > 4 and w not in self.stop_words])
        closing_themes = set([w for w in closing_text.lower().split() 
                           if len(w) > 4])
        
        theme_overlap = len(opening_themes.intersection(closing_themes)) / len(opening_themes) if opening_themes else 0
        
        # Return true if either similarity is high or thematic overlap is significant
        return similarity > 0.4 or theme_overlap > 0.3
    
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
    

    def get_path(self):
        """
        Get the path to the most recent audio file.
        
        When recorded here are the file names:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            video_filename = f'output_{timestamp}.mp4'
            audio_filename = f'output_{timestamp}.wav'

            base_dir = '../data/recordings'

            video_filename = base_dir + '/video/' video_filename
            audio_filename = base_dir + '/audio/' audio_filename
        
        Return the most recent .wav file
        """
        try:
            files = glob.glob(os.path.join(self.base_audio_path, 'output_*.wav'))
            if not files:
                raise FileNotFoundError(f"No WAV files found in {self.base_audio_path}")
            
            # Extract timestamps from filenames
            file_times = []
            for f in files:
                timestamp_str = os.path.basename(f).split('_')[1].split('.')[0]
                file_time = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
                file_times.append((f, file_time))
            
            # Sort by timestamp and return latest
            latest = max(file_times, key=lambda x: x[1])
            return latest[0]
        
        except Exception as e:
            print(f"Error finding audio file: {str(e)}")
            return None

    def get_message(self, filepath):
        from helper import AudioTranscriber
        """
        Get the message from the audio file.
        """
        audio_transcriber = AudioTranscriber(filepath)
        audio_transcriber.convert_to_wav()
        transcript = audio_transcriber.transcribe()
        if not transcript:
            raise ValueError(f"Transcription failed for file: {filepath}")
        return transcript

    def save_data(self, data):
        """
        Save the structure analysis data to a parquet file
        
        Args:
            data: Dictionary containing analysis results
        
        Returns:
            Path to the saved file
        """
        from .helper import save_factor_data
        save_factor_data(data, 'structure', self.timestamp)

    def calculate_and_save(self):
        """
        Calculate the structure score and save the data
        """
        message = self.get_message(self.get_path())
        data = self.analyze_structure(message)
        # Save the data to a file
        self.save_data(data)