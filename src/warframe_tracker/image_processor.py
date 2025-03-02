#!/usr/bin/env python3
"""
Image processor module for extracting inventory information from Warframe screenshots.
"""

import os
import cv2
import numpy as np
import pytesseract
import re
import matplotlib.pyplot as plt
from datetime import datetime
from .corrections_manager import CorrectionsManager

class Item:
    """
    Class representing an inventory item in Warframe.
    
    Contains item details such as name, type, and quantity.
    """
    
    def __init__(self, name, item_type, quantity=1):
        """
        Initialize a new item.
        
        Args:
            name (str): The name of the item
            item_type (str): The type of item (Blueprint, Component, Resource)
            quantity (int): The quantity of the item, defaults to 1
        """
        self.name = name
        self.type = item_type
        self.quantity = quantity
        self.added = datetime.now().isoformat()
        self.last_updated = self.added
    
    def to_dict(self):
        """
        Convert the item to a dictionary.
        
        Returns:
            dict: Item as a dictionary for storage in database
        """
        return {
            'name': self.name,
            'type': self.type,
            'quantity': self.quantity,
            'added': self.added,
            'last_updated': self.last_updated
        }
    
    @staticmethod
    def from_dict(data):
        """
        Create an Item instance from a dictionary.
        
        Args:
            data (dict): Dictionary containing item data
            
        Returns:
            Item: A new Item instance
        """
        item = Item(data['name'], data['type'], data['quantity'])
        if 'added' in data:
            item.added = data['added']
        if 'last_updated' in data:
            item.last_updated = data['last_updated']
        return item


class ImageProcessor:
    """Class to handle all image processing and OCR operations."""
    
    # Common Warframe item categories
    RESOURCE_KEYWORDS = [
        'alloy', 'crystal', 'plate', 'extract', 'essence', 
        'orb', 'sludge', 'core', 'nodule'
    ]
    BLUEPRINT_KEYWORDS = ['blueprint', 'systems', 'chassis', 'neuroptics']
    COMPONENT_KEYWORDS = [
        'barrel', 'receiver', 'stock', 'casing', 
        'fragment', 'tag', 'kuaka', 'kavat', 'pod'
    ]
    
    # Constants for image processing
    TOP_LEFT_HEIGHT_RATIO = 0.2  # Percentage of cell height to use for top-left region
    TOP_LEFT_WIDTH_RATIO = 0.3   # Percentage of cell width to use for top-left region
    DEFAULT_QUANTITY = 1         # Default quantity if none detected
    
    # OCR Configuration presets
    OCR_CONFIG_DEFAULT = r'--oem 3 --psm 11 -c preserve_interword_spaces=1'
    OCR_CONFIG_NUMBERS_ONLY = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    
    # Regular expression patterns
    REGEX_QUANTITY_PATTERN = r'(?:x|X)\s*(\d+)'
    REGEX_NUMBER_AT_START = r'^\s*(\d+)\s'
    REGEX_NUMBER_AT_END = r'\s(\d+)\s*$'
    REGEX_NUMBER_WITH_COMMA = r'(\d+)[,\s]+(\d+)'  # Match numbers like "162,109" or "162 109"
    REGEX_CLEAN_TEXT = r'[^\w\s\dxX,]'  # Now also preserving commas for better number parsing
    REGEX_NORMALIZE_SPACES = r'\s+'
    REGEX_WORD_BOUNDARY = r'(?i)\b{}\b'
    REGEX_ALL_DIGITS = r'\d'  # Match all individual digits
    REGEX_DIGIT_GROUPS = r'(\d+)'  # Match groups of consecutive digits
    
    def __init__(self, tesseract_path=None):
        """Initialize the image processor with the path to Tesseract."""
        # Set path to Tesseract executable if provided, otherwise assume it's in PATH
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Try to check Tesseract version to ensure it's installed
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR detected successfully.")
        except Exception as e:
            print(f"Warning: Could not detect Tesseract OCR: {e}")
            print("Please install Tesseract OCR and make sure it's in your PATH")
            print("Download from: https://github.com/tesseract-ocr/tesseract/releases")
        
        # Configure OCR parameters for better Warframe inventory recognition
        self.debug = False
        
        # Initialize the corrections manager
        self.corrections_manager = CorrectionsManager()
        
        # Store cell images and OCR results for potential correction
        self.last_processed_cells = []
    
    def _perform_ocr(self, image, config_type='default', lang='eng'):
        """
        Perform OCR on an image with specified configuration.
        
        Args:
            image: The image to perform OCR on
            config_type: Type of OCR configuration to use ('default' or 'numbers_only')
            lang: Language for OCR
            
        Returns:
            str: The extracted text
        """
        if config_type == 'numbers_only':
            config = self.OCR_CONFIG_NUMBERS_ONLY
        else:
            config = self.OCR_CONFIG_DEFAULT
            
        return pytesseract.image_to_string(image, lang=lang, config=config).strip()
    
    def process_screenshot(self, image_path):
        """
        Process a screenshot to extract item information.
        
        Args:
            image_path (str): Path to the screenshot image
            
        Returns:
            list: List of dictionaries containing item information
        """
        try:
            # Load the image
            print(f"Processing the image...")
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return []
                
            # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocess the image
            preprocessed = self._preprocess_image(img_rgb)
            
            # Segment the inventory grid
            cells = self.segment_inventory_grid(preprocessed)
            print(f"Detected {len(cells)} inventory cells in the image.")
            
            # Extract item information from each cell
            items = []
            self.last_processed_cells = []  # Clear previous cells
            
            for i, cell in enumerate(cells):
                if self.debug:
                    plt.imshow(cell)
                    plt.title("Cell Image")
                    plt.show()
                
                # Try to detect quantity from the cell image itself (top left corner)
                quantity_from_image = self._detect_quantity_from_cell_image(cell)
                
                # Extract text with OCR
                text = self._perform_ocr(cell)
                
                # Store the cell and OCR result for potential correction later
                cell_data = {
                    'cell_id': i,
                    'image': cell.copy(),
                    'ocr_text': text,
                    'item': None
                }
                
                # Check if we have a correction for this OCR text
                correction = self.corrections_manager.get_correction_for_text(text)
                if correction:
                    print(f"Applied text correction for: '{text}'")
                    self._create_item_from_detection(
                        cell_data, 
                        correction['name'], 
                        correction['type'], 
                        text, 
                        quantity_from_image, 
                        items
                    )
                    continue
                
                # Check if we have a visual correction
                visual_correction = self.corrections_manager.get_most_similar_visual_correction(cell, text)
                if visual_correction:
                    print(f"Applied visual correction with similarity score {visual_correction['score']:.2f}")
                    self._create_item_from_detection(
                        cell_data, 
                        visual_correction['name'], 
                        visual_correction['type'], 
                        text, 
                        quantity_from_image, 
                        items
                    )
                    continue
                    
                # If no correction worked, use OCR and parsing
                item = self._parse_cell_text(text)
                if item:
                    # Update quantity if we detected it from the image
                    if quantity_from_image > 1:
                        item.quantity = quantity_from_image
                    cell_data['item'] = item
                    items.append(item.to_dict())
                
                self.last_processed_cells.append(cell_data)
            
            print(f"Extracted {len(items)} items from the image.")
            return items
            
        except Exception as e:
            print(f"Error processing screenshot: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_item_from_detection(self, cell_data, name, item_type, text, quantity_from_image, items):
        """
        Create an item from detection results and update the data structures.
        
        Args:
            cell_data (dict): Dictionary with cell metadata
            name (str): Detected item name
            item_type (str): Detected item type
            text (str): OCR text from the cell
            quantity_from_image (int): Quantity detected from image analysis
            items (list): List of items to append to
            
        Returns:
            Item: The created item object
        """
        quantity = quantity_from_image or self._extract_quantity(text)
        item = Item(name, item_type, quantity)
        
        cell_data['item'] = item
        items.append(item.to_dict())
        self.last_processed_cells.append(cell_data)
        return item
    
    def segment_inventory_grid(self, image):
        """
        Segment the inventory grid into individual cells using knowledge
        of Warframe's inventory grid structure.
        
        Args:
            image: The preprocessed image containing the inventory grid
            
        Returns:
            list: A list of cell images
        """
        print("Detecting inventory grid cells...")
        height, width = image.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        # Warframe inventory knowledge: always 7 columns (left-to-right)
        # Rows can be 1-4, depending on the screenshot
        
        # Estimate number of rows from image height
        # Using the knowledge that cells are roughly square in Warframe
        estimated_cell_width = width / 7  # 7 columns is fixed in Warframe
        estimated_cell_height = estimated_cell_width  # Roughly square
        
        estimated_rows = max(1, min(4, round(height / estimated_cell_height)))
        print(f"Estimated rows based on image proportions: {estimated_rows}")
        
        # Use multiple detection methods and choose the most reliable result
        
        # Method 1: Grid line detection for full grids
        horizontal_lines, vertical_lines = self._detect_grid_lines(image)
        
        # Check if grid lines make sense
        if (len(horizontal_lines) >= 2 and len(vertical_lines) >= 2 and
            len(horizontal_lines) - 1 <= 5 and  # No more than 5 rows
            len(vertical_lines) - 1 <= 8):      # No more than 8 columns
            
            print(f"Grid line detection found {len(horizontal_lines)-1} rows and {len(vertical_lines)-1} columns")
            cells = self._extract_cells_from_grid_lines(image, horizontal_lines, vertical_lines)
            
            # Verify we have a reasonable number of cells
            if 5 <= len(cells) <= 35:
                print(f"Using grid line detection: {len(cells)} cells")
                return cells
        
        # Method 2: Fixed grid based on estimated rows (reliable for standard layouts)
        if estimated_rows in [1, 2, 4]:  # Most common layouts
            cells = self._segment_grid_by_fixed_size(image, estimated_rows, 7)
            
            # If we get the expected number of cells, return them
            if len(cells) == estimated_rows * 7:
                print(f"Using fixed grid ({estimated_rows}x7): {len(cells)} cells")
                return cells
        
        # Method 3: Try contour detection (useful for partial inventories)
        cells = self._segment_grid_by_contours(image)
        
        # Adjust for common inventory sizes based on number of detected cells
        if len(cells) > 3:
            if 10 <= len(cells) <= 15:
                # Likely a 2-row layout (expecting 14 cells)
                print(f"Detected cells ({len(cells)}) suggest 2-row layout")
                return self._segment_grid_by_fixed_size(image, 2, 7)
            
            elif len(cells) <= 9:
                # Likely a 1-row layout (expecting 7 cells)
                print(f"Detected cells ({len(cells)}) suggest 1-row layout")
                return self._segment_grid_by_fixed_size(image, 1, 7)
            
            elif len(cells) >= 20:
                # Likely a 4-row layout (expecting 28 cells)
                print(f"Detected cells ({len(cells)}) suggest 4-row layout")
                return self._segment_grid_by_fixed_size(image, 4, 7)
            
            print(f"Using contour detection: {len(cells)} cells")
            return cells
        
        # Fallback: Use fixed grid with estimated rows
        print(f"Using fallback fixed grid ({estimated_rows}x7)")
        return self._segment_grid_by_fixed_size(image, estimated_rows, 7)
    
    def _preprocess_image(self, image):
        """
        Preprocess the image to improve OCR accuracy.
        
        Args:
            image: The input image
            
        Returns:
            numpy.ndarray: The preprocessed image
        """
        # Convert to RGB if not already
        if len(image.shape) == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img = image.copy()
        
        # Apply bilateral filter to smooth the image while preserving edges
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply contrast stretching
        min_val, max_val = np.percentile(gray, (2, 98))
        stretched = np.clip((gray - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            stretched, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to RGB for compatibility
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _extract_cells_from_grid_lines(self, image, horizontal_lines, vertical_lines):
        """
        Extract cells from detected grid lines.
        
        Args:
            image: The preprocessed image
            horizontal_lines: List of y-coordinates for horizontal lines
            vertical_lines: List of x-coordinates for vertical lines
            
        Returns:
            list: A list of cell images
        """
        height, width = image.shape[:2]
        
        # Add image boundaries if needed
        if horizontal_lines[0] > height * 0.1:
            horizontal_lines.insert(0, 0)
        if horizontal_lines[-1] < height * 0.9:
            horizontal_lines.append(height)
        if vertical_lines[0] > width * 0.1:
            vertical_lines.insert(0, 0)
        if vertical_lines[-1] < width * 0.9:
            vertical_lines.append(width)
        
        # Extract cells
        cells = []
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                y1 = horizontal_lines[i]
                y2 = horizontal_lines[i + 1]
                x1 = vertical_lines[j]
                x2 = vertical_lines[j + 1]
                
                # Skip cells that are too small
                cell_height = y2 - y1
                cell_width = x2 - x1
                if cell_height < 10 or cell_width < 10:
                    continue
                
                # Add padding
                padding = 2
                y1_pad = max(0, y1 - padding)
                y2_pad = min(height, y2 + padding)
                x1_pad = max(0, x1 - padding)
                x2_pad = min(width, x2 + padding)
                
                cell = image[y1_pad:y2_pad, x1_pad:x2_pad]
                
                # Only add cells with some content
                if cell.size > 0:
                    cells.append(cell)
        
        return cells
    
    def _detect_grid_lines(self, image):
        """
        Detect horizontal and vertical grid lines in the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            tuple: (horizontal_lines, vertical_lines) where each is a list of line positions
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect vertical lines - use specialized vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
        vertical_detect = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Detect horizontal lines - use specialized horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
        horizontal_detect = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Combine the two images to get all lines
        combined = cv2.addWeighted(vertical_detect, 0.5, horizontal_detect, 0.5, 0)
        
        # Threshold the image
        _, thresh = cv2.threshold(combined, 25, 255, cv2.THRESH_BINARY_INV)
        
        # Apply edge detection for better line detection
        edges = cv2.Canny(thresh, 30, 100)
        
        # Dilate edges to enhance line detection
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(
            dilated, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50, 
            minLineLength=min(width, height) * 0.15,  # At least 15% of min dimension
            maxLineGap=20
        )
        
        if lines is None:
            # Try alternative approach - direct projection 
            horizontal_lines = self._detect_lines_by_projection(gray, axis=0)  # horizontal lines
            vertical_lines = self._detect_lines_by_projection(gray, axis=1)  # vertical lines
            return horizontal_lines, vertical_lines
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle to determine if line is horizontal or vertical
            dx, dy = x2 - x1, y2 - y1
            
            # Horizontal lines: angle close to 0 or 180 degrees
            if abs(dx) > abs(dy) * 3:  # Horizontal if x change >> y change
                y_avg = (y1 + y2) // 2
                horizontal_lines.append(y_avg)
            # Vertical lines: angle close to 90 degrees
            elif abs(dy) > abs(dx) * 3:  # Vertical if y change >> x change
                x_avg = (x1 + x2) // 2
                vertical_lines.append(x_avg)
        
        # If we didn't find lines, try direct projection
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            h_projection = self._detect_lines_by_projection(gray, axis=0)
            v_projection = self._detect_lines_by_projection(gray, axis=1) 
            
            # Use projection results if they found more lines
            if len(h_projection) > len(horizontal_lines):
                horizontal_lines = h_projection
            if len(v_projection) > len(vertical_lines):
                vertical_lines = v_projection
        
        # Remove duplicate lines by clustering them
        horizontal_lines = self._cluster_line_positions(horizontal_lines, height * 0.02)
        vertical_lines = self._cluster_line_positions(vertical_lines, width * 0.02)
        
        # Sort lines by position
        horizontal_lines.sort()
        vertical_lines.sort()
        
        return horizontal_lines, vertical_lines
    
    def _segment_grid_by_contours(self, image):
        """
        Segment the inventory grid by detecting contours of individual cells.
        
        Args:
            image: The preprocessed image
            
        Returns:
            list: A list of cell images
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply thresholding and morphological operations to enhance cell boundaries
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Dilate to enhance cell boundaries
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Calculate statistics to adapt to the image
        areas = [cv2.contourArea(c) for c in contours]
        if not areas:
            return []
        
        median_area = np.median(areas)
        
        # Estimate cell dimensions
        avg_cell_height = height / min(4, max(1, round(height / (width / 7))))
        avg_cell_width = width / 7
        
        # Filter contours to find cells
        filtered_contours = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Aspect ratio for cells (should be roughly square)
            aspect_ratio = w / h if h > 0 else 0
            
            # Adaptive area thresholds
            min_area = min(avg_cell_width * avg_cell_height * 0.3, median_area * 0.5)
            max_area = max(avg_cell_width * avg_cell_height * 3.0, median_area * 2.0)
            
            # Filter by area, aspect ratio, and minimum size
            if (min_area <= area <= max_area and 
                0.5 <= aspect_ratio <= 2.0 and 
                w >= 20 and h >= 20):
                
                # Add padding
                padding = 2
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(width - x_pad, w + padding * 2)
                h_pad = min(height - y_pad, h + padding * 2)
                
                # Extract cell with padding
                cell = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                
                # Add to filtered contours
                filtered_contours.append((x, y, w, h, cell))
        
        # Sort contours by position
        # Group cells into rows based on y-coordinate, then sort by x-coordinate
        row_threshold = avg_cell_height * 0.5
        
        # Group into rows
        rows = []
        filtered_contours.sort(key=lambda c: c[1])  # Sort by y-coordinate
        
        current_row = []
        current_y = -1000  # Initial value outside image
        
        for contour in filtered_contours:
            x, y, w, h, cell = contour
            
            if current_y < 0 or abs(y - current_y) <= row_threshold:
                # Add to current row
                current_row.append(contour)
                current_y = (current_y * len(current_row) + y) / (len(current_row))  # Update average y
            else:
                # Start a new row
                if current_row:
                    # Sort current row by x-coordinate
                    current_row.sort(key=lambda c: c[0])
                    rows.append(current_row)
                
                current_row = [contour]
                current_y = y
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)
        
        # Flatten rows to get all cells in correct order
        sorted_contours = []
        for row in rows:
            sorted_contours.extend(row)
        
        # Extract cell images
        cells = [c[4] for c in sorted_contours]
        
        return cells
    
    def _parse_cell_text(self, text):
        """
        Parse text from a single inventory cell.
        
        Args:
            text (str): The text extracted from an inventory cell
            
        Returns:
            Item or None: Item information if a valid item is found, None otherwise
        """
        # Clean up the text - remove special characters and normalize spaces
        text = re.sub(self.REGEX_CLEAN_TEXT, ' ', text)  # Keep letters, numbers, x/X and spaces
        text = re.sub(self.REGEX_NORMALIZE_SPACES, ' ', text)  # Normalize spaces
        text = text.strip()
        
        if not text:
            return None
            
        # Try to find quantity pattern
        quantity = self._extract_quantity(text)
        
        # Clean text for item name extraction
        name_text = text
        quantity_match = re.search(self.REGEX_QUANTITY_PATTERN, text)
        if quantity_match:
            name_text = text.replace(quantity_match.group(0), '').strip()
        
        # Look for item type indicators
        item_type = 'Blueprint'  # Default type
        
        # Check for specific type keywords
        if 'blueprint' in text.lower():
            item_type = 'Blueprint'
        elif any(keyword in text.lower() for keyword in self.COMPONENT_KEYWORDS):
            item_type = 'Component'
        elif any(keyword in text.lower() for keyword in self.RESOURCE_KEYWORDS):
            item_type = 'Resource'
        
        # Extract name - use the whole text if we can't identify parts
        name = name_text
        
        # Remove type words from name if present
        type_words = self.BLUEPRINT_KEYWORDS + self.COMPONENT_KEYWORDS
        for word in type_words:
            if word.lower() in name.lower():
                name = re.sub(self.REGEX_WORD_BOUNDARY.format(word), '', name).strip()
        
        # If name is too short or empty, return None
        if len(name) < 2:
            return None
            
        # Create and return an Item object
        return Item(name, item_type, quantity)
    
    def _extract_quantity(self, text):
        """
        Extract quantity from OCR text.
        
        Args:
            text (str): The OCR text
            
        Returns:
            int: The quantity, defaults to 1 if not found
        """
        # Debug information
        if self.debug:
            print(f"Extracting quantity from: '{text}'")

        # Special handling for heavily separated digit sequences
        # Try this first to catch patterns like "1 6 2 1 0 9 Alloy Plate" or "9 9 9 9 9 Polymer Bundle"
        clean_text = text.strip()
        words = clean_text.split()
        
        # If we have multiple single digits as separate words, it might be a large quantity with OCR errors
        single_digit_words = [word for word in words if word.isdigit() and len(word) == 1]
        if len(single_digit_words) >= 3:  # At least 3 separate digits suggests a number
            try:
                quantity = int(''.join(single_digit_words))
                if self.debug:
                    print(f"Found widely separated digits: {quantity}")
                return quantity
            except ValueError:
                pass
            
        # Try to find a quantity pattern like "x10" or "X 5"
        quantity_match = re.search(self.REGEX_QUANTITY_PATTERN, text)
        if quantity_match:
            quantity_part = text[quantity_match.start():]
            # Check if there are more digits following with spaces in between
            # e.g. "x 1 2 3 4 5" should be interpreted as "x12345"
            all_digits = re.findall(r'\d', quantity_part)
            if len(all_digits) > 1:
                try:
                    quantity = int(''.join(all_digits))
                    if self.debug:
                        print(f"Found extended quantity pattern: {quantity}")
                    return quantity
                except ValueError:
                    pass
            
            # If the extended approach fails, fall back to the original pattern
            quantity = int(quantity_match.group(1))
            if self.debug:
                print(f"Found quantity pattern: {quantity}")
            return quantity
        
        # Look for numbers with commas or spaces between digits (like "162,109" or "162 109")
        comma_match = re.search(self.REGEX_NUMBER_WITH_COMMA, text)
        if comma_match:
            try:
                # Join the number parts without comma/space and convert to int
                full_number = comma_match.group(1) + comma_match.group(2)
                quantity = int(full_number)
                if self.debug:
                    print(f"Found number with comma: {quantity}")
                return quantity
            except ValueError:
                # If conversion fails, continue with other patterns
                if self.debug:
                    print(f"Failed to convert comma-separated number: {comma_match.group(0)}")
                pass
        
        # Special handling for Warframe-typical resource quantities which are often high numbers
        # Check for any sequence of consecutive single digits separated by spaces
        # This helps with OCR outputs like "1 6 2 1 0 9 Alloy Plate"
        consecutive_digit_words = []
        for i, word in enumerate(words):
            if word.isdigit() and len(word) == 1:
                consecutive_digit_words.append((i, word))
            else:
                # Reset if we find a non-digit word
                if consecutive_digit_words and (i - consecutive_digit_words[-1][0] > 1):
                    break
                    
        # If we have at least 3 consecutive single digits
        if len(consecutive_digit_words) >= 3:
            try:
                digits = ''.join([digit for _, digit in consecutive_digit_words])
                quantity = int(digits)
                if self.debug:
                    print(f"Found sequence of consecutive single digits: {quantity}")
                return quantity
            except ValueError:
                if self.debug:
                    print(f"Failed to convert consecutive digits: {''.join([digit for _, digit in consecutive_digit_words])}")
                pass
        
        # Check if the first few words are all single digits, which is common in OCR 
        # for large numbers in Warframe inventory
        digit_sequence = []
        for word in words:
            if word.isdigit() and len(word) == 1:
                digit_sequence.append(word)
            else:
                break
                
        if len(digit_sequence) >= 3:  # At least 3 consecutive single-digit numbers
            try:
                quantity = int(''.join(digit_sequence))
                if self.debug:
                    print(f"Found sequence of digit-words at start: {quantity}")
                return quantity
            except ValueError:
                if self.debug:
                    print(f"Failed to convert digit sequence: {''.join(digit_sequence)}")
                pass
        
        # Try to extract all digit groups (for OCR that might combine some digits)
        # e.g., "162 109" or "16 21 09"
        digit_groups = re.findall(self.REGEX_DIGIT_GROUPS, clean_text)
        if len(digit_groups) >= 2 and all(len(dg) <= 3 for dg in digit_groups[:2]):
            # If we have multiple digit groups and they're reasonably sized
            # (to avoid capturing item IDs or other numeric identifiers)
            try:
                # Try combining the first two groups
                quantity = int(''.join(digit_groups[:2]))
                if self.debug:
                    print(f"Found and combined digit groups: {quantity}")
                return quantity
            except ValueError:
                if self.debug:
                    print(f"Failed to convert digit groups: {''.join(digit_groups[:2])}")
                pass
        
        # Try to find standalone numbers at the beginning or end that might be quantities
        # For example: "43 Spinal Core" or "Star Amarast 6"
        number_at_start = re.search(self.REGEX_NUMBER_AT_START, text)
        if number_at_start:
            quantity = int(number_at_start.group(1))
            if self.debug:
                print(f"Found number at start: {quantity}")
            return quantity
            
        number_at_end = re.search(self.REGEX_NUMBER_AT_END, text)
        if number_at_end:
            quantity = int(number_at_end.group(1))
            if self.debug:
                print(f"Found number at end: {quantity}")
            return quantity
            
        # As a last resort, try to extract all digits and combine them
        # This is a more aggressive approach that might work for some OCR errors
        all_digits = re.findall(self.REGEX_ALL_DIGITS, text)
        if len(all_digits) >= 3:  # Only consider if we have at least 3 digits
            try:
                # Check if all digits are at the beginning of the text
                first_non_digit_pos = min([text.find(c) for c in text if not c.isdigit() and c != ' ' and c != ','] + [len(text)])
                if first_non_digit_pos > 0:
                    digit_part = text[:first_non_digit_pos]
                    digit_only = ''.join([c for c in digit_part if c.isdigit()])
                    if len(digit_only) >= 3:
                        quantity = int(digit_only)
                        if self.debug:
                            print(f"Found all digits in prefix: {quantity}")
                        return quantity
            except (ValueError, IndexError):
                pass
        
        if self.debug:
            print(f"No quantity found, using default: {self.DEFAULT_QUANTITY}")
        return self.DEFAULT_QUANTITY
    
    def _detect_quantity_from_cell_image(self, cell):
        """
        Detect quantity from cell image by analyzing the top-left corner for icons.
        
        In Warframe inventory:
        - Checkbox in circle + number: quantity of the item
        - Microscope icon + number: number of blueprints
        - No icon in top left: quantity is 1 (for non-blueprints)
        
        Args:
            cell (numpy.ndarray): The cell image
            
        Returns:
            int: The detected quantity
        """
        # Get the top left corner of the cell (approximately 20% of cell dimensions)
        height, width = cell.shape[:2]
        top_left = cell[0:int(height * self.TOP_LEFT_HEIGHT_RATIO), 
                        0:int(width * self.TOP_LEFT_WIDTH_RATIO)]
        
        # Convert to grayscale and threshold
        if len(top_left.shape) == 3:
            gray = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
        else:
            gray = top_left
        
        # Apply OCR specifically to the top left corner to extract any numbers
        number_text = self._perform_ocr(gray, 'numbers_only')
        
        # Try to extract a number
        quantity = self.DEFAULT_QUANTITY
        if number_text and number_text.isdigit():
            quantity = int(number_text)
        
        # TODO: Future enhancement - use template matching to detect specific icons
        # (checkbox, microscope, infinity symbol)
        
        return quantity

    def process_all_screenshots(self, screenshot_dir='screenshots'):
        """
        Process all screenshots in the specified directory.
        
        Args:
            screenshot_dir (str): Path to the directory containing screenshots
            
        Returns:
            list: List of dictionaries containing item information from all screenshots
        """
        all_items = []
        
        # Check if the directory exists
        if not os.path.exists(screenshot_dir):
            print(f"Screenshot directory not found: {screenshot_dir}")
            return all_items
            
        # Get all image files from the directory
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = [f for f in os.listdir(screenshot_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not image_files:
            print(f"No images found in directory: {screenshot_dir}")
            return all_items
            
        # Process each image file
        for image_file in image_files:
            image_path = os.path.join(screenshot_dir, image_file)
            print(f"Processing screenshot: {image_file}")
            
            items = self.process_screenshot(image_path)
            all_items.extend(items)
            
        return all_items
    
    def add_correction(self, cell_id, correct_name, correct_type):
        """
        Add a correction for a processed cell.
        
        Args:
            cell_id (int): The ID of the cell to correct
            correct_name (str): The correct item name
            correct_type (str): The correct item type
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Find the cell with the given ID
        for cell_data in self.last_processed_cells:
            if cell_data['cell_id'] == cell_id:
                # Add a text correction
                self.corrections_manager.add_text_correction(
                    cell_data['ocr_text'], 
                    correct_name, 
                    correct_type
                )
                
                # Add a visual correction
                self.corrections_manager.add_visual_correction(
                    cell_data['image'],
                    cell_data['ocr_text'],
                    correct_name,
                    correct_type
                )
                
                print(f"Added correction for cell {cell_id}: '{correct_name}' ({correct_type})")
                return True
                
        print(f"Cell with ID {cell_id} not found.")
        return False
    
    def get_last_processed_cells(self):
        """
        Get the last processed cells for potential correction.
        
        Returns:
            list: The last processed cells
        """
        return self.last_processed_cells
    
    def _detect_lines_by_projection(self, image, axis=0):
        """
        Detect lines by projecting the image onto an axis.
        
        Args:
            image: Grayscale image
            axis: 0 for horizontal lines, 1 for vertical lines
            
        Returns:
            list: Line positions
        """
        # Make sure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Invert the image so lines are white
        _, inv_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Project onto the axis
        projection = np.sum(inv_img, axis=axis)
        
        # Normalize
        projection = (projection - np.min(projection)) / (np.max(projection) - np.min(projection) + 1e-6)
        
        # Find peaks
        lines = []
        threshold = 0.2
        peak_width_min = 3
        
        # Smooth projection
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(projection, kernel, mode='same')
        
        # Find local maxima
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > threshold and smoothed[i] > smoothed[i-1] and smoothed[i] >= smoothed[i+1]:
                # Check if this is a wide enough peak
                width = 1
                j = i + 1
                while j < len(smoothed) and smoothed[j] > threshold * 0.7:
                    width += 1
                    j += 1
                
                j = i - 1
                while j >= 0 and smoothed[j] > threshold * 0.7:
                    width += 1
                    j -= 1
                
                if width >= peak_width_min:
                    lines.append(i)
        
        # Try to find expected number of lines
        if len(lines) < 2:  # Not enough lines detected
            # For horizontal lines (axis=0), expect 2-5 lines for Warframe inventory grid
            if axis == 0 and image.shape[0] > 100:
                # Manually add lines based on image height
                height = image.shape[0]
                lines = [int(height * i / 5) for i in range(6)]  # 5 rows, 6 boundary lines
            
            # For vertical lines (axis=1), expect 7-8 lines for Warframe inventory grid
            elif axis == 1 and image.shape[1] > 100:
                # Manually add lines based on image width
                width = image.shape[1]
                lines = [int(width * i / 7) for i in range(8)]  # 7 columns, 8 boundary lines
        
        return lines
    
    def _cluster_line_positions(self, line_positions, threshold):
        """
        Cluster line positions that are close to each other.
        
        Args:
            line_positions: List of line positions (x or y coordinates)
            threshold: Distance threshold for clustering
            
        Returns:
            list: Clustered line positions
        """
        if not line_positions:
            return []
            
        # Sort positions
        positions = sorted(line_positions)
        
        # Group positions into clusters
        clusters = []
        current_cluster = [positions[0]]
        
        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] < threshold:
                # Positions are close, add to current cluster
                current_cluster.append(positions[i])
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [positions[i]]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        # Calculate the average position for each cluster
        return [sum(cluster) // len(cluster) for cluster in clusters]
    
    def _segment_grid_by_fixed_size(self, image, rows=4, cols=7):
        """
        Segment the inventory grid by dividing it into a fixed number of rows and columns.
        
        Args:
            image: The preprocessed image
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            
        Returns:
            list: A list of cell images
        """
        height, width = image.shape[:2]
        
        # Fixed cell size based on image dimensions and grid size
        cell_height = height // rows
        cell_width = width // cols
        
        print(f"Fixed grid segmentation: {rows*cols} cells with {rows} rows, {cols} columns")
        
        cells = []
        for row in range(rows):
            for col in range(cols):
                y_start = row * cell_height
                y_end = (row + 1) * cell_height
                x_start = col * cell_width
                x_end = (col + 1) * cell_width
                
                # Add some padding
                padding = 2
                y_start_pad = max(0, y_start - padding)
                y_end_pad = min(height, y_end + padding)
                x_start_pad = max(0, x_start - padding)
                x_end_pad = min(width, x_end + padding)
                
                cell = image[y_start_pad:y_end_pad, x_start_pad:x_end_pad]
                cells.append(cell)
        
        return cells
