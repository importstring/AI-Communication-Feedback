import unittest
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSimple(unittest.TestCase):
    
    def test_addition(self):
        """Test that Python can add numbers correctly"""
        self.assertEqual(1 + 1, 2)
        
    def test_subtraction(self):
        """Test that Python can subtract numbers correctly"""
        self.assertEqual(5 - 3, 2)
        
    def test_multiplication(self):
        """Test that Python can multiply numbers correctly"""
        self.assertEqual(2 * 3, 6)
        
    def test_division(self):
        """Test that Python can divide numbers correctly"""
        self.assertEqual(10 / 2, 5)

if __name__ == "__main__":
    unittest.main() 