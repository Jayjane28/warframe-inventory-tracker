# Warframe Inventory Tracker Memories

## System Components

### Core Architecture
- **Project Structure**: The Warframe Inventory Tracker has key components in:
  - Image Processing (`utils/image_processor.py`) - Handles screenshot processing using OpenCV, Tesseract OCR for text extraction, identifies inventory grid cells
  - Corrections System (`utils/corrections_manager.py`) - Stores text and visual corrections in JSON format, applies corrections to improve OCR accuracy
  - User Interface (`utils/user_interface.py`) - Provides CLI interface for user interaction, handles item correction workflow
  - Database Management - Stores inventory in JSON format, maintains persistent state across sessions

### User-Assisted Correction System
- Learns from user corrections over time
- Stores both text-based and visual corrections in a database
- Compares OCR results with previously seen patterns
- Applies corrections automatically
- Improves accuracy with continued use

### Inventory Grid Detection
- Enhanced algorithm specific to Warframe's inventory structure
- Leverages knowledge of Warframe's fixed 7-column inventory layout
- Prioritizes detection methods based on estimated rows (1-4)
- Uses multiple fallback strategies including grid line detection and contour-based approaches
- Handles partial grids and empty cells while preserving grid structure

## OCR Quantity Detection

### Supported Patterns
1. Standard patterns:
   - "x5", "X 10", "x 15"
   - "5x", "10X", "15 x"

2. Standalone numbers:
   - At beginning: "43 Spinal Core"
   - At end: "Star Amarast 6"
   - With item type: "Credits 150000"

3. Numbers with separators:
   - Comma-separated: "150,000 Credits"
   - Space-separated: "162 109 Alloy Plate"

4. Sequences of separated digits:
   - "1 6 2 1 0 9 Alloy Plate" → 162109
   - "Resource x 1 2 3 4 5" → 12345

### Algorithm Implementation
- Robust algorithm for detecting quantities in OCR text
- Prioritizes checking for widely separated digits first
- Moves to more specific patterns
- Addresses common OCR errors with large numbers getting broken apart

## Testing Framework

### Test Organization
1. Directory Structure:
   - `tests` directory at project root
   - Test files mirror main project structure (e.g., `tests/utils/test_image_processor.py`)
   - Separate directories for unit vs. integration tests

2. Test Data:
   - `test_data` directory containing:
     - Sample screenshots with known items and quantities
     - Expected OCR outputs
     - Fixtures for different correction scenarios

3. Unit Test Focus Areas:
   - OCR quantity extraction with various patterns
   - Correction application (text and visual)
   - Database operations (storage and retrieval)

4. Best Practices:
   - Descriptive test method names
   - Proper setup and teardown
   - Mock external dependencies
   - Include edge cases and failure scenarios

### Testing Strategy for OCR
- Create test cases for each pattern
- Include edge cases with ambiguous patterns
- Test transitions between patterns
- Maintain suite with examples of previously failed detections

### Test-Driven Development Approach
1. Write tests first defining expected behavior
2. Run and verify tests fail
3. Implement minimum code to make tests pass
4. Refactor while ensuring tests continue to pass
5. Repeat cycle for new features/bug fixes

### Testing Limitations
- Main application testing (`run_app.py`) requires manual testing
- Interactive console input not supported in automated environments
- Non-interactive tests work well with scripts like `test_corrections.py`