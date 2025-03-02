# Warframe Inventory Tracker Memories

## Memory 1: User-Assisted Correction System
I've implemented a User-Assisted Correction system for the Warframe Inventory Tracker that allows it to learn from user corrections over time. The system works by storing both text-based and visual corrections in a database. When processing new images, it compares OCR results with previously seen patterns and applies corrections automatically, improving accuracy with continued use.

## Memory 2: Quantity Detection Algorithm
I've implemented a robust algorithm for detecting quantities in OCR text from Warframe inventory screenshots. The system can now accurately extract quantities in various formats, including:

1. Standard patterns like "x5" or "X 10"
2. Standalone numbers at beginning or end (e.g., "43 Spinal Core" or "Star Amarast 6")
3. Numbers with commas or spaces (e.g., "150,000 Credits" or "162 109")
4. Sequences of separated single digits (e.g., "1 6 2 1 0 9 Alloy Plate")
5. Extended quantity patterns following an "x" marker (e.g., "Resource x 1 2 3 4 5")

The algorithm prioritizes checking for widely separated digits first, then moves to more specific patterns. This approach addresses the common OCR errors where large numbers get broken apart into individual digits.

## Memory 3: Test Organization
Test Organization for Warframe Inventory Tracker:

1. Test Directory Structure:
   - Create a 'tests' directory at the project root
   - Organize test files to mirror the main project structure (e.g., tests/utils/test_image_processor.py)
   - Use separate directories for unit tests vs. integration tests

2. Test Data:
   - Create a 'test_data' directory with:
     - Sample screenshots with known items and quantities
     - Expected OCR outputs
     - Fixtures representing different correction scenarios

3. Unit Test Focus Areas:
   - OCR quantity extraction: Test with various quantity patterns (e.g., "x5", standalone numbers, comma-separated numbers)
   - Correction application: Verify both text and visual corrections
   - Database operations: Test storage and retrieval of items and corrections

4. Best Practices:
   - Use descriptive test method names (test_should_extract_quantity_from_pattern_x5)
   - Set up and tear down test state properly
   - Mock external dependencies (e.g., Tesseract OCR) for unit tests
   - Include edge cases and failure scenarios

## Memory 4: Quantity Detection Patterns
Quantity Detection Patterns for OCR:

The system should detect quantities in these formats:

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

Testing strategy:
- Create test cases for each pattern
- Include edge cases with ambiguous patterns
- Test transitions between patterns (incremental improvements)
- Maintain a test suite with examples of previously failed detections

## Memory 5: Key Components
The Warframe Inventory Tracker has the following key components:

1. Image Processing (utils/image_processor.py):
   - Handles screenshot processing using OpenCV
   - Uses Tesseract OCR for text extraction
   - Identifies inventory grid cells and extracts items

2. Corrections System (utils/corrections_manager.py):
   - Stores text and visual corrections in JSON format
   - Applies corrections to improve OCR accuracy over time
   - Handles both exact matches and similarity-based corrections

3. User Interface (utils/user_interface.py):
   - Provides CLI interface for user interaction
   - Handles item correction workflow

4. Database Management:
   - Stores inventory in JSON format
   - Maintains persistent state across sessions

When testing, focus on isolated unit tests for each component before integration testing.

## Memory 6: Test-Driven Development Approach
When working on the Warframe Inventory Tracker, we should follow a Test-Driven Development (TDD) approach:

1. Write tests first that define the expected behavior
2. Run the tests and verify they fail (to confirm they're testing the right things)
3. Implement the minimum code necessary to make the tests pass
4. Refactor the code while ensuring tests continue to pass
5. Repeat the cycle for each new feature or bug fix

This approach will be particularly valuable for the OCR quantity detection and corrections system, where we've encountered issues that would benefit from more methodical testing.

## Memory 7: Manual Testing Note
When testing functionality in the main Warframe Inventory Tracker application (run_app.py), I should ask the user to perform manual testing rather than attempting to run it programmatically. The main application requires interactive console input which isn't supported in this environment. Automated test scripts like test_corrections.py work well for non-interactive testing.

## Memory 8: Enhanced Grid Detection
Enhanced the inventory grid detection algorithm to be more specific to Warframe's inventory structure. The improved system leverages knowledge of Warframe's fixed 7-column inventory layout, prioritizes detection methods based on estimated rows (1-4), and uses multiple fallback strategies including grid line detection and contour-based approaches. This approach handles partial grids and empty cells while preserving the entire grid structure.