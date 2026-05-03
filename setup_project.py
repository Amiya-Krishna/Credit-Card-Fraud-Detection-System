#!/usr/bin/env python
"""
Setup script to create project structure for Credit Card Fraud Detection
"""
import os
import sys

# Define the base directory structure
structure = {
    'data': ['README.md'],
    'notebooks': [],
    'src': ['__init__.py'],
    'models': [],
    'serving': [],
    'dashboard/app': [],
    'outputs': [],
    'images': [],
}

base_path = os.path.dirname(os.path.abspath(__file__))

def create_structure():
    """Create directory structure and placeholder files"""
    print("🚀 Creating project structure...\n")
    
    for folder, files in structure.items():
        full_path = os.path.join(base_path, folder)
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ {folder}/")
        
        for file in files:
            file_path = os.path.join(full_path, file)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write("")
                print(f"   ✅ {file}")
    
    print("\n✅ Project structure created successfully!")

if __name__ == "__main__":
    create_structure()
