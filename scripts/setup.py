"""
Setup script for Plant Disease Recognition System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Try main requirements first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error with main requirements: {e}")
        print("Trying minimal requirements...")
        
        # Try minimal requirements
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-minimal.txt"])
            print("✅ Minimal packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Error installing minimal packages: {e2}")
            print("\nManual installation required:")
            print("pip install tensorflow opencv-python pillow numpy pandas scikit-learn matplotlib seaborn flask flask-cors requests")
            return False

def check_dataset():
    """Check if dataset is available"""
    dataset_path = "plantvillage dataset/color"
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found at {dataset_path}")
        return True
    else:
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the PlantVillage dataset is in the correct location.")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static', 'models', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def main():
    """Main setup function"""
    print("Plant Disease Recognition System - Setup")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        print("❌ Setup failed at package installation")
        return
    
    # Check dataset
    print("\n3. Checking dataset...")
    if not check_dataset():
        print("❌ Setup failed - dataset not found")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python test_system.py' to test the system")
    print("2. Run 'python train_model.py' to train the model")
    print("3. Run 'python web_interface.py' to start the web application")

if __name__ == "__main__":
    main()
