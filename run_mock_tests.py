import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Mock modules that might be causing issues
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['mediapipe.tasks'] = MagicMock()
sys.modules['mediapipe.tasks.python'] = MagicMock()
sys.modules['mediapipe.tasks.python.vision'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()
sys.modules['wave'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test classes manually to avoid discovery issues
from tests.test_recording_tools import TestRecordVideo, test_record_function
from tests.test_factors.test_body_language import TestJointMap
from tests.test_factors.test_emotions import TestEmotion, TestMultimodalEmotionAnalyzer
from tests.test_factors.test_volume import TestVolumeVarience
from tests.test_factors.test_helper import TestHelper

if __name__ == "__main__":
    # Create test suite manually
    suite = unittest.TestSuite()
    
    # Add tests from TestRecordVideo
    suite.addTest(unittest.makeSuite(TestRecordVideo))
    
    # Add test_record_function
    suite.addTest(unittest.FunctionTestCase(test_record_function))
    
    # Add tests from TestJointMap
    suite.addTest(unittest.makeSuite(TestJointMap))
    
    # Add tests from TestEmotion
    suite.addTest(unittest.makeSuite(TestEmotion))
    
    # Add tests from TestMultimodalEmotionAnalyzer
    suite.addTest(unittest.makeSuite(TestMultimodalEmotionAnalyzer))
    
    # Add tests from TestVolumeVarience
    suite.addTest(unittest.makeSuite(TestVolumeVarience))
    
    # Add tests from TestHelper
    suite.addTest(unittest.makeSuite(TestHelper))
    
    # Run the suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 