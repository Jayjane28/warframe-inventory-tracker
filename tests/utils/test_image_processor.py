#!/usr/bin/env python3
"""
Tests for the image processor module.
"""

import os
import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from src.warframe_tracker.image_processor import ImageProcessor, Item


class TestImageProcessor(unittest.TestCase):
    """Test cases for the ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock for Tesseract to avoid actual OCR during tests
        self.tesseract_patcher = patch('src.warframe_tracker.image_processor.pytesseract')
        self.mock_tesseract = self.tesseract_patcher.start()
        
        # Configure the mock to return predictable values
        self.mock_tesseract.get_tesseract_version.return_value = '5.0.0'
        self.mock_tesseract.image_to_string.return_value = 'Test Item'
        
        # Create an instance of ImageProcessor with the mock
        self.image_processor = ImageProcessor()
        
        # Create test data directory if it doesn't exist
        self.test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create a sample test image (plain black 800x600 image)
        self.test_image_path = os.path.join(self.test_data_dir, 'test_blank.png')
        blank_image = np.zeros((600, 800, 3), np.uint8)
        cv2.imwrite(self.test_image_path, blank_image)
    
    def tearDown(self):
        """Clean up after each test method."""
        self.tesseract_patcher.stop()
        
        # Remove test image if it exists
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_image_processor_initialization(self):
        """Test that ImageProcessor initializes correctly."""
        self.assertIsNotNone(self.image_processor)
        self.assertIsNotNone(self.image_processor.corrections_manager)
    
    def test_item_class(self):
        """Test the Item class functionality."""
        # Create a test item
        item = Item(name="Test Item", item_type="Resource", quantity=5)
        
        # Check properties
        self.assertEqual(item.name, "Test Item")
        self.assertEqual(item.type, "Resource")
        self.assertEqual(item.quantity, 5)
        
        # Check the to_dict method
        item_dict = item.to_dict()
        self.assertEqual(item_dict["name"], "Test Item")
        self.assertEqual(item_dict["type"], "Resource")
        self.assertEqual(item_dict["quantity"], 5)
    
    @patch('src.warframe_tracker.image_processor.ImageProcessor._preprocess_image')
    @patch('src.warframe_tracker.image_processor.ImageProcessor.segment_inventory_grid')
    @patch('src.warframe_tracker.image_processor.ImageProcessor._parse_cell_text')
    def test_process_screenshot_basic(self, mock_parse, mock_segment, mock_preprocess):
        """Test basic screenshot processing with mocks."""
        # Configure mocks
        mock_preprocess.return_value = np.zeros((600, 800, 3), np.uint8)
        mock_segment.return_value = [
            {'cell_image': np.zeros((100, 100, 3), np.uint8), 'position': (0, 0)}
        ]
        mock_parse.return_value = ("Test Item", "Resource", "Test Item", 5)
        
        # Call the method
        result = self.image_processor.process_screenshot(self.test_image_path)
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "Test Item")
        self.assertEqual(result[0]["quantity"], 5)
        
        # Verify mocks were called correctly
        mock_preprocess.assert_called_once()
        mock_segment.assert_called_once()
        mock_parse.assert_called_once()
    
    def test_extract_quantity_standard_pattern(self):
        """Test extracting quantity with standard pattern 'x5'."""
        # Mock the private _perform_ocr method to avoid actual OCR
        with patch.object(self.image_processor, '_perform_ocr') as mock_ocr:
            mock_ocr.return_value = "Resource x5"
            quantity = self.image_processor._extract_quantity("Resource x5")
            self.assertEqual(quantity, 5)
    
    def test_extract_quantity_number_at_start(self):
        """Test extracting quantity with number at start pattern '5 Resource'."""
        quantity = self.image_processor._extract_quantity("5 Resource")
        self.assertEqual(quantity, 5)
    
    def test_extract_quantity_number_at_end(self):
        """Test extracting quantity with number at end pattern 'Resource 5'."""
        quantity = self.image_processor._extract_quantity("Resource 5")
        self.assertEqual(quantity, 5)
    
    def test_extract_quantity_with_comma(self):
        """Test extracting quantity with comma separator '1,000'."""
        quantity = self.image_processor._extract_quantity("Resource 1,000")
        self.assertEqual(quantity, 1000)
    
    def test_extract_quantity_separated_digits(self):
        """Test extracting quantity with separated digits '1 5 0 0 0'."""
        quantity = self.image_processor._extract_quantity("1 5 0 0 0 Resource")
        self.assertEqual(quantity, 15000)


if __name__ == '__main__':
    unittest.main()
