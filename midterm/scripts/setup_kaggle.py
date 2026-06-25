"""
Setup script for Kaggle API and Brain-to-Text competition data download
"""

import os
import subprocess
import sys

def setup_kaggle_credentials():
    """
    Guide user through setting up Kaggle API credentials
    """
    print("Setting up Kaggle API credentials...")
    print("\nTo use the Kaggle API, you need to:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download a file called 'kaggle.json'")
    print("5. Place this file in ~/.kaggle/ directory")
    print("\nAlternatively, you can manually download the competition data from:")
    print("https://www.kaggle.com/competitions/brain-to-text-25/data")
    
    # Create .kaggle directory if it doesn't exist
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    print(f"\nCreated directory: {kaggle_dir}")
    print("Please place your kaggle.json file in this directory.")
    
    return kaggle_dir

def download_competition_data():
    """
    Download the Brain-to-Text competition data
    """
    try:
        print("\nAttempting to download competition data...")
        result = subprocess.run([
            "kaggle", "competitions", "download", "-c", "brain-to-text-25"
        ], capture_output=True, text=True, cwd="/Users/meg/Desktop/1141datamining")
        
        if result.returncode == 0:
            print("Data downloaded successfully!")
            print(result.stdout)
        else:
            print("Error downloading data:")
            print(result.stderr)
            print("\nPlease ensure you have:")
            print("1. Set up your Kaggle API credentials")
            print("2. Accepted the competition rules")
            print("3. Have internet connection")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please install kaggle CLI and set up credentials manually.")

if __name__ == "__main__":
    setup_kaggle_credentials()
    download_competition_data()

