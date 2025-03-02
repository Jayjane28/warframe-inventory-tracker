# Warframe Inventory Tracker

A Python application that processes Warframe inventory screenshots to track items, types, and quantities.

## Setup

1. Install Python 3.8+ if not already installed
2. Install Tesseract OCR:
   - Windows: Download from https://github.com/tesseract-ocr/tesseract/releases
   - Default installation path: `C:\Program Files\Tesseract-OCR\`
   - Make sure to add Tesseract to your PATH during installation
3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Take screenshots of your Warframe inventory
2. Place screenshots in the `screenshots` folder
3. Run the main application:
   ```
   python run_app.py
   ```
4. Use the menu options to:
   - Process screenshots
   - Search for specific items
   - View your complete inventory
   - Export inventory to CSV
   - Correct misidentified items and help the system learn
   - View correction statistics

## Testing

Several test scripts are available to verify functionality:

1. Basic functionality test:
   ```
   python test_sample.py
   ```

2. Test screenshot processing:
   ```
   python test_processing.py
   ```

3. Non-interactive test (runs all features without user input):
   ```
   python test_noninteractive.py
   ```

4. Debug image processing steps:
   ```
   python debug_image.py
   ```

## Features

- Processes Warframe inventory screenshots using OCR
- Advanced image preprocessing for better text recognition
- Extracts item names, types, and quantities
- Recognizes common Warframe resources and components
- Stores inventory data in a JSON database
- Search and filtering capabilities
- Export to CSV for spreadsheet analysis
- **Learning system** that improves over time through user corrections

## Learning System

The application includes a user-assisted correction feature that:

1. **Remembers Corrections**: When you correct a misidentified item, the system stores both:
   - The OCR text that was incorrectly read
   - The visual appearance of the item

2. **Applies Corrections Automatically**: When processing new screenshots, the system:
   - Checks if the OCR text matches any previous corrections
   - Uses visual pattern matching to identify similar items

3. **Improves Over Time**: The more you use the system and correct misidentifications:
   - The more accurate it becomes
   - Common error patterns are learned and fixed automatically

To use the learning feature:
1. Process your screenshots
2. Select "Correct misidentified items" from the menu
3. Enter the Cell ID of any item that needs correction
4. Provide the correct name and type

You can view statistics about your corrections and how the system is learning by selecting "View correction statistics" from the menu.

## Troubleshooting

If OCR is not recognizing items correctly:

1. Use the debug script to see the processing steps:
   ```
   python debug_image.py your_screenshot.png
   ```

2. Check the `debug_output` folder to see how the image is processed

3. Make sure your screenshots are clear and well-lit

4. Try adjusting in-game UI settings for better contrast

5. Use the correction feature to teach the system about frequently misidentified items
