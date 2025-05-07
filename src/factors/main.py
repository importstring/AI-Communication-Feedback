import os
import sys
from datetime import datetime
import numpy as np

from body_language import JointMap
from coherence import CoherenceAnalyzer
from emotions import MultimodalEmotionAnalyzer, Emotion
from speech_patterns import SpeechPatternAnalyzer
from tonation import TonalAnalyzer
from volume import VolumeVarience

class FactorAnalyzer:
    def __init__(self, timestamp: str):
        self.joint_map = JointMap()
        self.coherence_analyzer = CoherenceAnalyzer()
        self.emotion_analyzer = MultimodalEmotionAnalyzer()
        self.speech_pattern_analyzer = SpeechPatternAnalyzer()
        self.tonal_analyzer = TonalAnalyzer()
        self.volume_varience = VolumeVarience()
        self.timestamp = timestamp

    def analyze(self):
        self.joint_map.map_recording('video.mp4') 
        

    def analyze_and_save(self):
        self.analyze()
        self.save_factors()