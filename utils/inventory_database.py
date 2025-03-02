#!/usr/bin/env python3
"""
Inventory database module for storing and retrieving Warframe inventory items.
"""

import os
import json
import pandas as pd
from datetime import datetime

class InventoryDatabase:
    """Class to handle storage and retrieval of inventory items."""
    
    def __init__(self, db_dir='database', db_file='inventory.json'):
        """Initialize the database with the specified directory and file."""
        self.db_dir = db_dir
        self.db_file = os.path.join(db_dir, db_file)
        
        # Create database directory if it doesn't exist
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Initialize or load the database
        if os.path.exists(self.db_file):
            self.load_database()
        else:
            self.inventory = []
            self.save_database()
    
    def load_database(self):
        """Load the inventory database from the JSON file."""
        try:
            with open(self.db_file, 'r') as f:
                self.inventory = json.load(f)
            print(f"Loaded {len(self.inventory)} items from database.")
        except Exception as e:
            print(f"Error loading database: {e}")
            self.inventory = []
    
    def save_database(self):
        """Save the inventory database to the JSON file."""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.inventory, f, indent=4)
            print(f"Saved {len(self.inventory)} items to database.")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_items(self, items, overwrite=False):
        """
        Add items to the inventory database.
        
        Args:
            items (list): List of dictionaries containing item information
            overwrite (bool): Whether to overwrite existing items or update quantities
        
        Returns:
            int: Number of items added or updated
        """
        if not items:
            return 0
            
        count = 0
        for item in items:
            # Check if the item already exists
            existing_item = self._find_item(item['name'], item['type'])
            
            if existing_item:
                if overwrite:
                    # Replace the item entirely
                    existing_item['quantity'] = item['quantity']
                    existing_item['last_updated'] = datetime.now().isoformat()
                else:
                    # Update the quantity
                    existing_item['quantity'] += item['quantity']
                    existing_item['last_updated'] = datetime.now().isoformat()
            else:
                # Add timestamp to the item
                item['added'] = datetime.now().isoformat()
                item['last_updated'] = item['added']
                
                # Add the item to the inventory
                self.inventory.append(item)
            
            count += 1
        
        # Save the updated inventory
        self.save_database()
        
        return count
    
    def _find_item(self, name, item_type):
        """
        Find an item in the inventory by name and type.
        
        Args:
            name (str): The name of the item
            item_type (str): The type of the item
        
        Returns:
            dict or None: The item if found, None otherwise
        """
        for item in self.inventory:
            if item['name'].lower() == name.lower() and item['type'].lower() == item_type.lower():
                return item
        
        return None
    
    def search_items(self, query=None, item_type=None):
        """
        Search for items in the inventory.
        
        Args:
            query (str): Search query for item name (case-insensitive)
            item_type (str): Filter by item type (case-insensitive)
        
        Returns:
            list: List of items matching the search criteria
        """
        results = []
        
        for item in self.inventory:
            match = True
            
            if query and query.lower() not in item['name'].lower():
                match = False
            
            if item_type and item_type.lower() != item['type'].lower():
                match = False
            
            if match:
                results.append(item)
        
        return results
    
    def get_all_items(self):
        """
        Get all items in the inventory.
        
        Returns:
            list: List of all inventory items
        """
        return self.inventory
    
    def get_item_types(self):
        """
        Get all unique item types in the inventory.
        
        Returns:
            list: List of unique item types
        """
        types = set()
        for item in self.inventory:
            types.add(item['type'])
        
        return sorted(list(types))
    
    def export_to_csv(self, filename='inventory_export.csv'):
        """
        Export the inventory to a CSV file.
        
        Args:
            filename (str): The name of the CSV file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.DataFrame(self.inventory)
            export_path = os.path.join(self.db_dir, filename)
            df.to_csv(export_path, index=False)
            print(f"Exported inventory to {export_path}")
            return True
        except Exception as e:
            print(f"Error exporting inventory: {e}")
            return False
