#!/usr/bin/env python3
"""
Tests for the corrections manager module.
"""

import os
import json
import unittest
import numpy as np
import cv2
from unittest.mock import patch, mock_open

# Import the module to test
from src.warframe_tracker.corrections_manager import CorrectionsManager


class TestCorrectionsManager(unittest.TestCase):
    """Test cases for the CorrectionsManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Use a temporary directory for testing
        self.test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data', 'temp_corrections')
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file = os.path.join(self.test_dir, 'test_corrections.json')
        
        # Create a test instance
        self.corrections_manager = CorrectionsManager(
            corrections_dir=self.test_dir, 
            corrections_file='test_corrections.json'
        )
        
        # Create a test image for visual corrections
        self.test_image = np.zeros((100, 100, 3), np.uint8)
        cv2.putText(self.test_image, 'Test', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove test file and directory
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_dir):
            # Only remove if empty (safety check)
            try:
                os.rmdir(self.test_dir)
            except OSError:
                # Directory not empty, that's fine for testing
                pass
    
    def test_initialization(self):
        """Test that CorrectionsManager initializes correctly."""
        self.assertIsNotNone(self.corrections_manager)
        self.assertEqual(self.corrections_manager.corrections_dir, self.test_dir)
        self.assertEqual(self.corrections_manager.corrections_file, self.test_file)
        self.assertEqual(self.corrections_manager.text_corrections, {})
        self.assertEqual(self.corrections_manager.visual_corrections, [])
    
    def test_add_text_correction(self):
        """Test adding a text-based correction."""
        # Add a correction
        self.corrections_manager.add_text_correction(
            ocr_text="Tost Item",
            correct_name="Test Item", 
            correct_type="Resource"
        )
        
        # Verify it was added properly
        self.assertIn("Tost Item", self.corrections_manager.text_corrections)
        correction = self.corrections_manager.text_corrections["Tost Item"]
        self.assertEqual(correction["name"], "Test Item")
        self.assertEqual(correction["type"], "Resource")
    
    def test_apply_text_correction(self):
        """Test applying a text-based correction."""
        # Add a correction first
        self.corrections_manager.add_text_correction(
            ocr_text="Tost Item",
            correct_name="Test Item", 
            correct_type="Resource"
        )
        
        # Test exact match
        corrected_name, corrected_type = self.corrections_manager.apply_text_correction("Tost Item")
        self.assertEqual(corrected_name, "Test Item")
        self.assertEqual(corrected_type, "Resource")
        
        # Test no match
        corrected_name, corrected_type = self.corrections_manager.apply_text_correction("Unknown Item")
        self.assertEqual(corrected_name, "Unknown Item")  # Should return original
        self.assertIsNone(corrected_type)  # Type is unknown
    
    def test_save_and_load_corrections(self):
        """Test saving and loading corrections from file."""
        # Add some corrections
        self.corrections_manager.add_text_correction(
            ocr_text="Tost Item",
            correct_name="Test Item", 
            correct_type="Resource"
        )
        
        # Save the corrections
        self.corrections_manager.save_corrections()
        
        # Create a new instance to load the saved corrections
        new_manager = CorrectionsManager(
            corrections_dir=self.test_dir, 
            corrections_file='test_corrections.json'
        )
        
        # Verify the corrections were loaded
        self.assertIn("Tost Item", new_manager.text_corrections)
        correction = new_manager.text_corrections["Tost Item"]
        self.assertEqual(correction["name"], "Test Item")
        self.assertEqual(correction["type"], "Resource")


if __name__ == '__main__':
    unittest.main()
