import os
import subprocess
import sys

def build_exe():
    print("==================================================")
    # 1. Install pyinstaller if not present
    try:
        import PyInstaller
        print("[OK] PyInstaller is already installed.")
    except ImportError:
        print("[INFO] PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
    # 2. Check if PyInstaller can find hidden PyTorch imports
    # PyTorch needs some manual care sometimes with PyInstaller.
    # We will build with standard options.
    cmd = [
        "pyinstaller",
        "--onefile",
        "--noconsole",
        "--name=TopoGrad_OCR",
        "--clean",
        "gui_app.py"
    ]
    
    print("\n[INFO] Compiling GUI application into an executable...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("\n==================================================")
        print("[SUCCESS] Executable built successfully!")
        print("You can find the standalone exe at: dist/TopoGrad_OCR.exe")
        print("Note: Make sure the folder 'ocr_evaluation_outputs_super_hybrid' remains in the same directory as the executable to load the model weights correctly.")
        print("==================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] PyInstaller compilation failed: {e}")
        
if __name__ == "__main__":
    build_exe()
