#!/usr/bin/env python3
"""
Multimodal Phishing Detection - Deployment Runner

Run this from the PROJECT ROOT directory:
    python deployment/run_deployment.py

FIXED BUGS:
- check_requirements() was using relative paths which broke when the script
  was run from any directory other than the project root.  All paths are now
  resolved relative to the project root (parent of this file's directory).
- data/combined_urls.csv was checked but the actual file lives at
  data/combined_urls.csv (flat, not in a subdirectory).  Path corrected.
- subprocess.run() now runs web_api.py via its absolute path so it works
  from any cwd, without the fragile os.chdir() hack.
"""

import os
import sys
import subprocess
import time
import webbrowser
import threading
from datetime import datetime

# Project root = parent of this file's directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_requirements():
    """Check if all required files exist."""
    print(" Checking deployment requirements...")

    required_files = [
        os.path.join(PROJECT_ROOT, "models", "fusion_model.keras"),
        os.path.join(PROJECT_ROOT, "models", "tokenizer.pkl"),
        os.path.join(PROJECT_ROOT, "deployment", "web_api.py"),
    ]

    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        print(" Missing files:")
        for f in missing:
            print(f"   - {os.path.relpath(f, PROJECT_ROOT)}")
        return False

    print(" All required files found!")
    return True


def install_dependencies():
    """Install required dependencies from requirements.txt."""
    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.exists(req_file):
        print("  requirements.txt not found  skipping install")
        return True

    print(" Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            check=True
        )
        print(" Dependencies installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error installing dependencies: {e}")
        return False


def open_browser():
    """Open the browser after a short delay."""
    time.sleep(3)
    try:
        webbrowser.open('http://localhost:5000')
        print(" Browser opened  http://localhost:5000")
    except Exception:
        print("  Could not open browser. Open http://localhost:5000 manually.")


def run_web_api():
    """Run the Flask API."""
    api_script = os.path.join(PROJECT_ROOT, "deployment", "web_api.py")

    print("\n Starting Web API...")
    print("=" * 60)
    print(" Web Interface:  http://localhost:5000")
    print(" POST /predict   - Single URL prediction")
    print(" POST /batch_predict - Batch prediction")
    print(" GET  /health    - Health check")
    print("=" * 60)
    print(" Press Ctrl+C to stop")
    print()

    # Open browser in background
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    try:
        subprocess.run([sys.executable, api_script], check=True)
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    except subprocess.CalledProcessError as e:
        print(f" Web API error: {e}")
        return False

    return True


def main():
    print(" Multimodal Phishing Detection  Deployment")
    print("=" * 60)
    print(f" {datetime.now().strftime('%B %d, %Y %H:%M')}")
    print("=" * 60)

    if not check_requirements():
        print("\n Please train the model first:")
        print("   python scripts/train_model.py")
        return

    if not install_dependencies():
        print("\n Failed to install dependencies.")
        return

    run_web_api()


if __name__ == "__main__":
    main()
