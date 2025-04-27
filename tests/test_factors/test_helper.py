import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.factors.helper import save_factor_data

class TestHelper(unittest.TestCase):
    
    @patch('src.factors.helper.pd.DataFrame.to_csv')
    @patch('src.factors.helper.os.path.exists')
    @patch('src.factors.helper.os.makedirs')
    def test_save_factor_data(self, mock_makedirs, mock_exists, mock_to_csv):
        """Test saving factor data to CSV"""
        # Setup mocks
        mock_exists.return_value = False
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0],
            'value': [0.1, 0.2, 0.3]
        })
        
        # Call the function
        result = save_factor_data(test_data, "test_factor", "test_id")
        
        # Assertions
        mock_exists.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_to_csv.assert_called_once()
        
        # Check the returned path
        self.assertEqual(result, "factors/test_factor/test_id.csv")
    
    @patch('src.factors.helper.pd.DataFrame.to_csv')
    @patch('src.factors.helper.os.path.exists')
    @patch('src.factors.helper.os.makedirs')
    def test_save_factor_data_existing_dir(self, mock_makedirs, mock_exists, mock_to_csv):
        """Test saving factor data when directory already exists"""
        # Setup mocks
        mock_exists.return_value = True
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0],
            'value': [0.1, 0.2, 0.3]
        })
        
        # Call the function
        result = save_factor_data(test_data, "test_factor", "test_id")
        
        # Assertions
        mock_exists.assert_called_once()
        mock_makedirs.assert_not_called()
        mock_to_csv.assert_called_once()
        
        # Check the returned path
        self.assertEqual(result, "factors/test_factor/test_id.csv")

if __name__ == "__main__":
    unittest.main() 