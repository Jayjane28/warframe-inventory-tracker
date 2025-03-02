#!/usr/bin/env python3
"""
Script to run the Warframe Inventory Tracker application with error handling.
"""

import os
import sys
from warframe_tracker.user_interface import display_menu, process_choice
from warframe_tracker.image_processor import ImageProcessor
from warframe_tracker.inventory_database import InventoryDatabase

def main():
    """Main function that runs the inventory tracker."""
    print("===== Warframe Inventory Tracker =====")
    print("This application helps you track your Warframe inventory by processing screenshots.")
    
    # Set the Tesseract path
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Check if Tesseract is installed
    if not os.path.exists(tesseract_path):
        print("\nIMPORTANT: Tesseract OCR is required for image processing.")
        print(f"Could not find Tesseract at: {tesseract_path}")
        print("Please download and install Tesseract from:")
        print("https://github.com/tesseract-ocr/tesseract/releases")
        sys.exit(1)
    
    print(f"Using Tesseract OCR: {tesseract_path}")
    
    try:
        # Initialize the image processor and database
        image_processor = ImageProcessor(tesseract_path)
        db = InventoryDatabase()
        
        # Create screenshots directory if it doesn't exist
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
            print("\nCreated 'screenshots' directory.")
            print("Please add your inventory screenshots to this folder:")
            print(os.path.abspath('screenshots'))
        
        print("\nUsing database at:", os.path.abspath(db.db_file))
        
        # Main application loop
        while True:
            choice = display_menu()
            
            if choice == '7':
                print("Exiting Warframe Inventory Tracker. Goodbye!")
                sys.exit(0)
            
            process_choice(choice, image_processor, db)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
