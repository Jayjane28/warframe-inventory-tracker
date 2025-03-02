#!/usr/bin/env python3
"""
User interface module for the Warframe Inventory Tracker.
"""

import os
import sys
from tabulate import tabulate

def display_menu():
    """
    Display the main menu and get the user's choice.
    
    Returns:
        str: The user's choice
    """
    print("\n===== Warframe Inventory Tracker Menu =====")
    print("1. Process all screenshots in the 'screenshots' folder")
    print("2. Search inventory")
    print("3. View all inventory")
    print("4. Export inventory to CSV")
    print("5. Correct misidentified items")
    print("6. View correction statistics")
    print("7. Exit")
    
    choice = input("\nEnter your choice (1-7): ")
    return choice

def process_choice(choice, image_processor, db):
    """
    Process the user's menu choice.
    
    Args:
        choice (str): The user's choice
        image_processor (ImageProcessor): The image processor instance
        db (InventoryDatabase): The inventory database instance
    """
    if choice == '1':
        process_screenshots(image_processor, db)
    elif choice == '2':
        search_inventory(db)
    elif choice == '3':
        view_all_inventory(db)
    elif choice == '4':
        export_inventory(db)
    elif choice == '5':
        correct_items(image_processor, db)
    elif choice == '6':
        view_correction_stats(image_processor)
    elif choice == '7':
        print("Exiting Warframe Inventory Tracker. Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice. Please try again.")

def process_screenshots(image_processor, db):
    """
    Process all screenshots in the screenshots folder.
    
    Args:
        image_processor (ImageProcessor): The image processor instance
        db (InventoryDatabase): The inventory database instance
    """
    screenshots_dir = 'screenshots'
    
    if not os.listdir(screenshots_dir):
        print(f"No screenshots found in the '{screenshots_dir}' folder.")
        print(f"Please add screenshots to '{os.path.abspath(screenshots_dir)}'")
        return
    
    print(f"\nProcessing screenshots in '{screenshots_dir}' folder...")
    items = image_processor.process_all_screenshots(screenshots_dir)
    
    if not items:
        print("No items were extracted from the screenshots.")
        return
    
    print(f"\nExtracted {len(items)} items from screenshots.")
    
    # Ask whether to overwrite or update quantities
    overwrite = input("Do you want to overwrite existing items? (y/n): ").lower() == 'y'
    
    # Add items to the database
    added_count = db.add_items(items, overwrite)
    
    print(f"Added or updated {added_count} items in the database.")

def search_inventory(db):
    """
    Search for items in the inventory.
    
    Args:
        db (InventoryDatabase): The inventory database instance
    """
    print("\n===== Search Inventory =====")
    
    # Get all item types for filtering
    item_types = db.get_item_types()
    if item_types:
        print("\nAvailable item types:")
        for i, item_type in enumerate(item_types, 1):
            print(f"{i}. {item_type}")
    
    # Get search criteria
    query = input("\nEnter search query (or leave empty to skip): ")
    
    item_type = None
    if item_types:
        type_choice = input("Enter item type number (or leave empty to skip): ")
        if type_choice.isdigit() and 1 <= int(type_choice) <= len(item_types):
            item_type = item_types[int(type_choice) - 1]
    
    # Search for items
    results = db.search_items(query, item_type)
    
    if not results:
        print("No items found matching your search criteria.")
        return
    
    # Display results
    display_items(results)

def view_all_inventory(db):
    """
    View all items in the inventory.
    
    Args:
        db (InventoryDatabase): The inventory database instance
    """
    print("\n===== All Inventory Items =====")
    
    items = db.get_all_items()
    
    if not items:
        print("No items in the inventory.")
        return
    
    # Display all items
    display_items(items)

def display_items(items):
    """
    Display items in a tabular format.
    
    Args:
        items (list): List of items to display
    """
    # Prepare data for tabulate
    headers = ["Name", "Type", "Quantity", "Last Updated"]
    table_data = []
    
    for item in items:
        table_data.append([
            item['name'],
            item['type'],
            item['quantity'],
            item.get('last_updated', 'N/A').split('T')[0]  # Show just the date part
        ])
    
    # Sort by name
    table_data.sort(key=lambda x: x[0])
    
    # Display the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal items: {len(items)}")

def export_inventory(db):
    """
    Export the inventory to a CSV file.
    
    Args:
        db (InventoryDatabase): The inventory database instance
    """
    print("\n===== Export Inventory =====")
    
    # Get filename
    default_filename = 'inventory_export.csv'
    filename = input(f"Enter filename (default: {default_filename}): ")
    
    if not filename:
        filename = default_filename
    
    # Add .csv extension if not present
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    
    # Export to CSV
    success = db.export_to_csv(filename)
    
    if success:
        print(f"Inventory exported successfully to '{os.path.join(db.db_dir, filename)}'")
    else:
        print("Failed to export inventory.")

def correct_items(image_processor, db):
    """
    Allow the user to correct misidentified items.
    
    Args:
        image_processor (ImageProcessor): The image processor instance
        db (InventoryDatabase): The inventory database instance
    """
    print("\n===== Correct Misidentified Items =====")
    print("This feature allows you to correct items that were misidentified during OCR processing.")
    print("The system will learn from your corrections over time and improve its accuracy.")
    
    # Get the last processed cells
    cells = image_processor.get_last_processed_cells()
    
    if not cells:
        print("\nNo items have been processed yet. Please process some screenshots first.")
        return
    
    # Display the items that were identified
    print("\nItems identified in the last processing session:")
    print("------------------------------------------------")
    
    # Create a table of the items
    table_data = []
    for cell in cells:
        cell_id = cell['cell_id']
        item = cell['item']
        ocr_text = cell['ocr_text'].replace('\n', ' ')[:30]  # Truncate and clean OCR text
        
        if item:
            row = [
                cell_id,
                item.name,
                item.type,
                item.quantity,
                ocr_text
            ]
        else:
            row = [
                cell_id,
                "Unknown",
                "Unknown",
                "N/A",
                ocr_text
            ]
        
        table_data.append(row)
    
    # Display the table
    headers = ["Cell ID", "Item Name", "Item Type", "Quantity", "OCR Text"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Ask the user which item to correct
    while True:
        correct_id = input("\nEnter the Cell ID of the item to correct (or 'q' to quit): ")
        
        if correct_id.lower() == 'q':
            break
        
        try:
            cell_id = int(correct_id)
            # Find the cell
            cell_found = False
            for cell in cells:
                if cell['cell_id'] == cell_id:
                    cell_found = True
                    current_item = cell['item']
                    
                    print("\nCurrent identification:")
                    if current_item:
                        print(f"Name: {current_item.name}")
                        print(f"Type: {current_item.type}")
                        print(f"Quantity: {current_item.quantity}")
                    else:
                        print("Item was not recognized")
                    
                    # Get the correct information
                    correct_name = input("Enter the correct item name: ")
                    
                    # Get item type
                    print("\nItem Types:")
                    types = ["Blueprint", "Resource", "Component", "Mod", "Other"]
                    for i, type_name in enumerate(types, 1):
                        print(f"{i}. {type_name}")
                    
                    while True:
                        type_choice = input("Enter the item type number: ")
                        if type_choice.isdigit() and 1 <= int(type_choice) <= len(types):
                            correct_type = types[int(type_choice) - 1]
                            break
                        print("Invalid choice. Please try again.")
                    
                    # Add the correction
                    success = image_processor.add_correction(cell_id, correct_name, correct_type)
                    
                    if success:
                        print(f"\nCorrection added for Cell ID {cell_id}.")
                        print("The system will use this correction to improve future OCR accuracy.")
                        
                        # Update the database if needed
                        if current_item:
                            # Ask if the user wants to update the database
                            update_db = input("Do you want to update this item in the database? (y/n): ").lower() == 'y'
                            
                            if update_db:
                                # Get the quantity from the current item or ask the user
                                quantity = current_item.quantity
                                quantity_str = input(f"Enter the correct quantity (default: {quantity}): ")
                                if quantity_str.isdigit():
                                    quantity = int(quantity_str)
                                
                                # Create the corrected item
                                corrected_item = {
                                    'name': correct_name,
                                    'type': correct_type,
                                    'quantity': quantity
                                }
                                
                                # Update the database
                                db.add_items([corrected_item], overwrite=True)
                                print("Database updated successfully.")
                    else:
                        print(f"Failed to add correction for Cell ID {cell_id}.")
                    
                    break
            
            if not cell_found:
                print(f"Cell ID {cell_id} not found. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid Cell ID or 'q' to quit.")

def view_correction_stats(image_processor):
    """
    View statistics about the corrections database.
    
    Args:
        image_processor (ImageProcessor): The image processor instance
    """
    print("\n===== Correction Statistics =====")
    
    # Get the correction statistics
    stats = image_processor.corrections_manager.get_correction_stats()
    
    print(f"Text corrections: {stats['text_corrections_count']}")
    print(f"Visual corrections: {stats['visual_corrections_count']}")
    
    if stats['top_corrections']:
        print("\nTop corrections (by usage count):")
        for i, (text, correction) in enumerate(stats['top_corrections'], 1):
            print(f"{i}. '{text}' → '{correction['name']}' ({correction['type']}) - Used {correction.get('count', 0)} times")
    else:
        print("\nNo corrections have been added yet.")
