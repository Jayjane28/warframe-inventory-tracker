#!/usr/bin/env python3
"""
Setup script for the Warframe Inventory Tracker package.
"""

from setuptools import setup, find_packages

setup(
    name="warframe-tracker",
    version="0.1.0",
    description="A tool for tracking your Warframe inventory using OCR",
    author="Warframe Inventory Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "opencv-python",
        "numpy",
        "pytesseract",
    ],
    entry_points={
        "console_scripts": [
            "warframe-tracker=warframe_tracker.main:main",
        ],
    },
    python_requires=">=3.8",
)
