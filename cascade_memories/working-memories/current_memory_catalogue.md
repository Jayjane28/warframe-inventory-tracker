# Warframe Inventory Tracker Memories

## Memory 1: Key Components
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

## Memory 2: Test-Driven Development Approach
When working on the Warframe Inventory Tracker, we should follow a Test-Driven Development (TDD) approach:

1. Write tests first that define the expected behavior
2. Run the tests and verify they fail (to confirm they're testing the right things)
3. Implement the minimum code necessary to make the tests pass
4. Refactor the code while ensuring tests continue to pass
5. Repeat the cycle for each new feature or bug fix

This approach will be particularly valuable for the OCR quantity detection and corrections system, where we've encountered issues that would benefit from more methodical testing.

-----------------------------

# User Background & Communication Preferences

## Technical Background
- Senior Frontend Software Engineer specializing in Angular
- Strong understanding of general programming concepts across languages
- May need implementation help with non-frontend languages (converting pseudo-code to actual code)

## Communication Style
- Keep responses concise and focused (ADHD-friendly)
- Use a casual conversational tone with appropriate humor
- Break complex information into digestible chunks
- Get to the point quickly, then expand if needed
- Use analogies and metaphors when explaining technical concepts

## Teaching Approach
- Provide context for why certain approaches are recommended
- Offer frontend-to-backend analogies when applicable
- Focus on practical implementation rather than theoretical concepts
- Assume competence with programming fundamentals

---------------------

# Warframe Inventory System

## Inventory Structure
- Inventory is organized in a grid-based system (7 columns)

## Item Types
- **Resources**: Basic crafting materials (Rubedo, Nano Spores, etc.)
- **Components**: Parts used to craft weapons/Warframes (Systems, Chassis, etc.)
- **Weapons**: Primary, Secondary, and Melee weapons
- **Warframes**: The playable "suits" with unique abilities
- **Blueprints**: Blueprints for making a new item
- **Relics**: Orb-like items with a random item inside
- **Misc**: Items that don't fit into the other categories

## Inventory Display
- Items show quantity in top left corner, always positioned consistently
- Different icons appear next to quantities depending on item type:
  * Most items: Yellow/gold checkmark inside a circle
  * Blueprints: Microscope icon
- Reusable blueprints show an infinity symbol (∞) instead of a numeric quantity
- Item names appear at the bottom of each inventory cell in gold/yellow text 
- Some resources display large quantities (100k+) which may appear with commas
- Quantity is displayed as a number only (no "x" prefix)
- Items appear on a dark background with distinct cell borders

## Visual Characteristics
- Items have distinct visual appearances (colors, shapes)
- Background is predominantly dark, making light-colored items and text stand out
- Each inventory slot is a bordered rectangle containing the item image, name, and quantity
- Some cells contain decorative lines within the main cell border
- Some items may not display any quantity indicator at all

## Relevant OCR Challenges
- Text often overlays on item images with varying contrast
- Quantity text can be small and potentially misread
- Similar-looking items may need contextual distinction
- Long item names may be difficult to extract accurately
- Text can overlay on visually busy backgrounds with similar colors
- Items with detailed/complex backgrounds create low-contrast situations
- Some item names contain multiple words that might be detected separately
- Special symbols like infinity (∞) need to be recognized and handled differently
- Different icons next to quantities may need to be ignored during extraction
- The system needs to differentiate between numbers and special quantity indicators

## Grid Detection Challenges
- Need to distinguish between actual cell borders and decorative internal lines
- Grid detection should focus on consistent outer borders rather than internal designs
- Cell content varies significantly (items with/without quantities, different icons)
- Consistent 7-column layout provides structural constraint that can aid detection