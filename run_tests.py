#!/usr/bin/env python3
"""
Test runner for Warframe Inventory Tracker.
"""

import unittest
import sys
import os

def run_tests():
    """Run all tests for the Warframe Inventory Tracker."""
    print("===== Running Warframe Inventory Tracker Tests =====")
    
    # Add appropriate paths to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    
    # Discover and run all tests
    test_suite = unittest.defaultTestLoader.discover('tests')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
