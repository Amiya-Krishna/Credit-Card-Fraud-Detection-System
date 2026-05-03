import os
from pathlib import Path

# Base directory
base = Path(r"C:\Users\amiya\Desktop\python bible\MEGA PROJECT 1 JARVIS\jarvis")

# Create all directories
folders = [
    base / "data",
    base / "notebooks",
    base / "src",
    base / "models",
    base / "serving",
    base / "dashboard" / "app",
    base / "outputs",
    base / "images",
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {folder}")

print("\n✅ All directories created successfully!")
