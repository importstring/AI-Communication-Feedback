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
from structure import Structure

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

    def analyze(self):
        self.joint_map.XYZ()
        self.coherence_analyzer.analyze_and_save()
        self.emotion_analyzer.analyze_and_save()
        self.speech_pattern_analyzer.analyze_and_save()
        self.speech_structure_analyzer.analyze_and_save()
        self.XYZ.XYZ()
        self.XYZ.XYZ()
        self.XYZ.XYZ()
        self.XYZ.XYZ()
        self.XYZ.XYZ()



    def analyze_and_save(self):
        self.analyze()
        self.save_factors()

