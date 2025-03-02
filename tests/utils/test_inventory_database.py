#!/usr/bin/env python3
"""
Tests for the inventory database module.
"""

import os
import json
import unittest
from unittest.mock import patch, mock_open

# Import the module to test
from src.warframe_tracker.inventory_database import InventoryDatabase


class TestInventoryDatabase(unittest.TestCase):
    """Test cases for the InventoryDatabase class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Use a temporary directory for testing
        self.test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data', 'temp_db')
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file = os.path.join(self.test_dir, 'test_inventory.json')
        
        # Create a test instance
        self.db = InventoryDatabase(
            db_dir=self.test_dir, 
            db_file='test_inventory.json'
        )
        
        # Sample test items
        self.test_items = [
            {
                "name": "Test Resource",
                "type": "Resource",
                "quantity": 100,
                "added": "2023-01-01T00:00:00",
                "last_updated": "2023-01-01T00:00:00"
            },
            {
                "name": "Test Component",
                "type": "Component",
                "quantity": 5,
                "added": "2023-01-01T00:00:00",
                "last_updated": "2023-01-01T00:00:00"
            }
        ]
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove test file and directory
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_dir):
            try:
                os.rmdir(self.test_dir)
            except OSError:
                # Directory not empty, that's fine for testing
                pass
    
    def test_initialization(self):
        """Test that InventoryDatabase initializes correctly."""
        self.assertIsNotNone(self.db)
        self.assertEqual(self.db.db_dir, self.test_dir)
        self.assertEqual(self.db.db_file, self.test_file)
        self.assertEqual(self.db.inventory, [])
    
    def test_add_item(self):
        """Test adding an item to the database."""
        # Add a test item
        self.db.add_item("Test Item", "Resource", 10)
        
        # Verify it was added properly
        self.assertEqual(len(self.db.inventory), 1)
        added_item = self.db.inventory[0]
        self.assertEqual(added_item["name"], "Test Item")
        self.assertEqual(added_item["type"], "Resource")
        self.assertEqual(added_item["quantity"], 10)
    
    def test_update_item_quantity(self):
        """Test updating an item's quantity."""
        # Add a test item first
        self.db.add_item("Test Item", "Resource", 10)
        
        # Update the quantity
        updated = self.db.update_item_quantity("Test Item", 20)
        
        # Verify it was updated properly
        self.assertTrue(updated)
        self.assertEqual(self.db.inventory[0]["quantity"], 20)
    
    def test_get_item(self):
        """Test getting an item from the database."""
        # Add a test item first
        self.db.add_item("Test Item", "Resource", 10)
        
        # Get the item
        item = self.db.get_item("Test Item")
        
        # Verify we got the correct item
        self.assertIsNotNone(item)
        self.assertEqual(item["name"], "Test Item")
        self.assertEqual(item["quantity"], 10)
        
        # Try getting an item that doesn't exist
        item = self.db.get_item("Non-existent Item")
        self.assertIsNone(item)
    
    def test_save_and_load_database(self):
        """Test saving and loading the database."""
        # Add some test items
        for item in self.test_items:
            self.db.add_item(item["name"], item["type"], item["quantity"])
        
        # Save the database
        self.db.save_database()
        
        # Create a new database instance to load the saved data
        new_db = InventoryDatabase(
            db_dir=self.test_dir, 
            db_file='test_inventory.json'
        )
        
        # Verify the data was loaded correctly
        self.assertEqual(len(new_db.inventory), len(self.test_items))
        for i, item in enumerate(self.test_items):
            loaded_item = new_db.get_item(item["name"])
            self.assertIsNotNone(loaded_item)
            self.assertEqual(loaded_item["name"], item["name"])
            self.assertEqual(loaded_item["type"], item["type"])
            self.assertEqual(loaded_item["quantity"], item["quantity"])


if __name__ == '__main__':
    unittest.main()
