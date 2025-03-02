#!/usr/bin/env python3
"""
Corrections manager module for handling user corrections to improve OCR accuracy over time.
"""

import os
import json
import cv2
import numpy as np
import base64
from datetime import datetime

class CorrectionsManager:
    """Class to handle storage and retrieval of user corrections for OCR improvement."""
    
    def __init__(self, corrections_dir='temp_db', corrections_file='corrections.json'):
        """Initialize the corrections manager with the specified directory and file."""
        self.corrections_dir = corrections_dir
        self.corrections_file = os.path.join(corrections_dir, corrections_file)
        self.text_corrections = {}  # Map from incorrect OCR text to correct item names
        self.visual_corrections = []  # List of visual corrections with original images
        
        # Create corrections directory if it doesn't exist
        if not os.path.exists(corrections_dir):
            os.makedirs(corrections_dir)
        
        # Initialize or load the corrections database
        if os.path.exists(self.corrections_file):
            self.load_corrections()
        else:
            self.save_corrections()
    
    def load_corrections(self):
        """Load the corrections database from the JSON file."""
        try:
            with open(self.corrections_file, 'r') as f:
                data = json.load(f)
                self.text_corrections = data.get('text_corrections', {})
                
                # Load visual corrections
                self.visual_corrections = []
                for corr in data.get('visual_corrections', []):
                    # Convert base64 string back to image if it exists
                    if 'image_data' in corr:
                        try:
                            img_data = base64.b64decode(corr['image_data'])
                            nparr = np.frombuffer(img_data, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            corr['image'] = img
                        except Exception as e:
                            print(f"Error decoding image for correction: {e}")
                            corr['image'] = None
                    
                    self.visual_corrections.append(corr)
                
            print(f"Loaded {len(self.text_corrections)} text corrections and {len(self.visual_corrections)} visual corrections.")
        except Exception as e:
            print(f"Error loading corrections database: {e}")
            self.text_corrections = {}
            self.visual_corrections = []
    
    def save_corrections(self):
        """Save the corrections database to the JSON file."""
        try:
            # Prepare data for serialization
            data = {
                'text_corrections': self.text_corrections,
                'visual_corrections': []
            }
            
            # Convert images to base64 strings for storage
            for corr in self.visual_corrections:
                corr_copy = corr.copy()
                
                # Convert image to base64 if it exists
                if 'image' in corr and corr['image'] is not None:
                    _, img_encoded = cv2.imencode('.png', corr['image'])
                    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                    corr_copy['image_data'] = img_base64
                
                # Remove the actual image object as it's not JSON serializable
                if 'image' in corr_copy:
                    del corr_copy['image']
                
                data['visual_corrections'].append(corr_copy)
            
            with open(self.corrections_file, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"Saved {len(self.text_corrections)} text corrections and {len(self.visual_corrections)} visual corrections.")
        except Exception as e:
            print(f"Error saving corrections database: {e}")
    
    def add_text_correction(self, incorrect_text, correct_item_name, correct_item_type):
        """
        Add a text-based correction.
        
        Args:
            incorrect_text (str): The incorrect OCR text
            correct_item_name (str): The correct item name
            correct_item_type (str): The correct item type
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Normalize text for better matching
            incorrect_text = self._normalize_text(incorrect_text)
            
            self.text_corrections[incorrect_text] = {
                'name': correct_item_name,
                'type': correct_item_type,
                'count': 1,
                'last_updated': datetime.now().isoformat()
            }
            
            self.save_corrections()
            return True
        except Exception as e:
            print(f"Error adding text correction: {e}")
            return False
    
    def add_visual_correction(self, image, ocr_text, correct_item_name, correct_item_type):
        """
        Add a visual correction with the original image.
        
        Args:
            image: The original image
            ocr_text (str): The OCR text that was extracted
            correct_item_name (str): The correct item name
            correct_item_type (str): The correct item type
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            correction = {
                'image': image,
                'ocr_text': ocr_text,
                'name': correct_item_name,
                'type': correct_item_type,
                'added': datetime.now().isoformat()
            }
            
            self.visual_corrections.append(correction)
            self.save_corrections()
            return True
        except Exception as e:
            print(f"Error adding visual correction: {e}")
            return False
    
    def get_correction_for_text(self, ocr_text):
        """
        Get the correction for the given OCR text.
        
        Args:
            ocr_text (str): The OCR text to check
        
        Returns:
            dict or None: The correction if found, None otherwise
        """
        # Normalize text for better matching
        normalized_text = self._normalize_text(ocr_text)
        
        # Check if there's an exact match
        if normalized_text in self.text_corrections:
            correction = self.text_corrections[normalized_text]
            
            # Update the count
            correction['count'] = correction.get('count', 0) + 1
            correction['last_updated'] = datetime.now().isoformat()
            self.text_corrections[normalized_text] = correction
            
            return {
                'name': correction['name'],
                'type': correction['type']
            }
        
        # If no exact match, check for partial matches
        for text, correction in self.text_corrections.items():
            # Check if the normalized OCR text contains the known incorrect text
            if text in normalized_text or normalized_text in text:
                # Calculate similarity to avoid false positives
                similarity = self._calculate_similarity(normalized_text, text)
                if similarity > 0.7:  # Threshold for considering it a match
                    # Update the count
                    correction['count'] = correction.get('count', 0) + 1
                    correction['last_updated'] = datetime.now().isoformat()
                    self.text_corrections[text] = correction
                    
                    return {
                        'name': correction['name'],
                        'type': correction['type']
                    }
        
        return None
    
    def get_most_similar_visual_correction(self, image, ocr_text):
        """
        Get the most similar visual correction for the given image.
        
        Args:
            image: The image to check
            ocr_text (str): The OCR text to help with matching
        
        Returns:
            dict or None: The most similar correction if found, None otherwise
        """
        best_match = None
        best_score = 0
        
        # Normalize the input OCR text
        normalized_ocr = self._normalize_text(ocr_text)
        
        for correction in self.visual_corrections:
            # Skip entries without images
            if 'image' not in correction or correction['image'] is None:
                continue
            
            # Calculate visual similarity
            visual_score = self._calculate_image_similarity(image, correction['image'])
            
            # Calculate text similarity
            text_score = self._calculate_similarity(normalized_ocr, correction['ocr_text'])
            
            # Combined score (weighted more towards visual similarity)
            combined_score = (visual_score * 0.7) + (text_score * 0.3)
            
            if combined_score > best_score and combined_score > 0.6:  # Threshold
                best_score = combined_score
                best_match = correction
        
        if best_match:
            return {
                'name': best_match['name'],
                'type': best_match['type'],
                'score': best_score
            }
        
        return None
    
    def _normalize_text(self, text):
        """
        Normalize text by removing extra spaces, converting to lowercase, etc.
        
        Args:
            text (str): The text to normalize
        
        Returns:
            str: The normalized text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        import re
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _calculate_similarity(self, text1, text2):
        """
        Calculate the similarity between two text strings.
        Uses a simple Levenshtein distance based similarity.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
        
        Returns:
            float: Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0
            
        # Normalize texts
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        # Use Levenshtein distance
        import Levenshtein
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 0
            
        return 1 - (distance / max_len)
    
    def _calculate_image_similarity(self, img1, img2):
        """
        Calculate the similarity between two images.
        
        Args:
            img1: First image
            img2: Second image
        
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Resize images to the same size
            img1_resized = cv2.resize(img1, (100, 100))
            img2_resized = cv2.resize(img2, (100, 100))
            
            # Convert to grayscale
            if len(img1_resized.shape) == 3:
                img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1_resized
                
            if len(img2_resized.shape) == 3:
                img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2_resized
            
            # Use histogram comparison
            hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Calculate correlation
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Return a value between 0 and 1
            return max(0, min(1, (similarity + 1) / 2))
        except Exception as e:
            print(f"Error calculating image similarity: {e}")
            return 0
    
    def get_top_corrections(self, limit=10):
        """
        Get the top corrections by usage count.
        
        Args:
            limit (int): Maximum number of corrections to return
        
        Returns:
            list: List of top corrections
        """
        # Sort text corrections by count
        sorted_corrections = sorted(
            self.text_corrections.items(),
            key=lambda x: x[1].get('count', 0),
            reverse=True
        )
        
        # Return the top corrections
        return sorted_corrections[:limit]
    
    def get_correction_stats(self):
        """
        Get statistics about the corrections database.
        
        Returns:
            dict: Statistics about the corrections
        """
        return {
            'text_corrections_count': len(self.text_corrections),
            'visual_corrections_count': len(self.visual_corrections),
            'top_corrections': self.get_top_corrections(5)
        }
