import os
import sys
from datetime import datetime
import numpy as np

from body_language import JointMap
from coherence import CoherenceAnalyzer
from emotions import Emotion
from speech_patterns import SpeechPatternAnalyzer
from tonation import TonalAnalyzer
from volume import VolumeVarience
from structure import Structure

from .helper import save_factor_data, get_video_path, get_audio_path, get_transcript_path, read_transcript

class FactorAnalyzer:
    def __init__(self, timestamp: str):
        self.joint_map = JointMap(timestamp)
        self.coherence_analyzer = CoherenceAnalyzer(timestamp)
        self.emotion_analyzer = Emotion(timestamp)
        self.speech_pattern_analyzer = SpeechPatternAnalyzer(timestamp)
        self.speech_structure_analyzer = Structure(timestamp)
        self.tonal_analyzer = TonalAnalyzer(timestamp)
        self.volume_varience = VolumeVarience(timestamp)
        
        self.timestamp = timestamp

    def extract_facotrs(self):
        self.joint_map.analyze_and_save()
        self.coherence_analyzer.analyze_and_save()
        self.emotion_analyzer.analyze_and_save()
        self.speech_pattern_analyzer.analyze_and_save()
        self.speech_structure_analyzer.analyze_and_save()
        self.tonal_analyzer.analyze_and_save()
        self.volume_varience.analyze_and_save()