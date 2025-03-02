#!/usr/bin/env python3
"""
Install the package in development mode.
This allows changes to the source code to be immediately reflected without reinstalling.
"""

import subprocess
import sys

def main():
    """Install the package in development mode."""
    print("Installing Warframe Inventory Tracker in development mode...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("Installation successful!")
        print("\nYou can now run the application with:")
        print("  python run_app_new.py")
        print("\nOr run the tests with:")
        print("  python run_tests.py")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
