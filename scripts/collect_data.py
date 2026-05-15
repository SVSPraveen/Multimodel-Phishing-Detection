"""
collect_data.py
---------------
Data collection script for multimodal phishing detection.
Collects phishing URLs from PhishTank and legitimate URLs from Tranco.
"""

import os
import sys
import pandas as pd
import requests
import logging
from pathlib import Path
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    PHISHTANK_API_URL, URLHAUS_API_URL, TRANCO_URL,
    DATASET_SIZE, LEGITIMATE_SAMPLES, PHISHING_SAMPLES,
    RAW_DATA_DIR, DATASETS_DIR, PHISHING_DATA_PATH, 
    LEGITIMATE_DATA_PATH, COMBINED_DATA_PATH, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_phishing_urls():
    """Collect phishing URLs from PhishTank and URLhaus."""
    logger.info("Collecting phishing URLs...")
    
    phishing_urls = []
    
    # Collect from PhishTank
    try:
        logger.info(f"Fetching from PhishTank: {PHISHTANK_API_URL}")
        response = requests.get(PHISHTANK_API_URL, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if isinstance(data, list):
            for item in data[:PHISHING_SAMPLES]:
                if 'url' in item:
                    phishing_urls.append({
                        'url': item['url'],
                        'source': 'phishtank',
                        'verified': item.get('verified', 'yes')
                    })
        
        logger.info(f"Collected {len(phishing_urls)} URLs from PhishTank")
        
    except Exception as e:
        logger.error(f"Error fetching from PhishTank: {e}")
    
    # If we need more URLs, try URLhaus
    if len(phishing_urls) < PHISHING_SAMPLES:
        try:
            logger.info(f"Fetching from URLhaus: {URLHAUS_API_URL}")
            response = requests.get(URLHAUS_API_URL, timeout=30)
            response.raise_for_status()
            
            lines = response.text.split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip() and len(phishing_urls) < PHISHING_SAMPLES:
                    parts = line.split(',')
                    if len(parts) > 2:
                        phishing_urls.append({
                            'url': parts[2],
                            'source': 'urlhaus',
                            'verified': 'yes'
                        })
            
            logger.info(f"Total phishing URLs collected: {len(phishing_urls)}")
            
        except Exception as e:
            logger.error(f"Error fetching from URLhaus: {e}")
    
    return phishing_urls[:PHISHING_SAMPLES]

def collect_legitimate_urls():
    """Collect legitimate URLs from Tranco top list."""
    logger.info("Collecting legitimate URLs...")
    
    legitimate_urls = []
    
    try:
        logger.info(f"Fetching from Tranco: {TRANCO_URL}")
        response = requests.get(TRANCO_URL, timeout=30)
        response.raise_for_status()
        
        lines = response.text.split('\n')
        for line in lines[:LEGITIMATE_SAMPLES + 1]:  # +1 for header
            if line.strip() and ',' in line:
                parts = line.split(',')
                if len(parts) >= 2 and parts[1].replace('.', '').isalnum():
                    legitimate_urls.append({
                        'url': parts[1],
                        'source': 'tranco',
                        'rank': int(parts[0])
                    })
        
        logger.info(f"Collected {len(legitimate_urls)} legitimate URLs")
        
    except Exception as e:
        logger.error(f"Error fetching from Tranco: {e}")
        
        # Fallback to common legitimate domains
        fallback_domains = [
            'google.com', 'facebook.com', 'youtube.com', 'amazon.com',
            'microsoft.com', 'apple.com', 'netflix.com', 'instagram.com',
            'twitter.com', 'linkedin.com', 'github.com', 'stackoverflow.com',
            'wikipedia.org', 'reddit.com', 'medium.com', 'quora.com'
        ]
        
        legitimate_urls = [
            {'url': domain, 'source': 'fallback', 'rank': i+1}
            for i, domain in enumerate(fallback_domains[:LEGITIMATE_SAMPLES])
        ]
        
        logger.info(f"Used fallback domains: {len(legitimate_urls)}")
    
    return legitimate_urls[:LEGITIMATE_SAMPLES]

def save_datasets(phishing_urls, legitimate_urls):
    """Save collected datasets to CSV files."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    # Save phishing URLs
    phishing_df = pd.DataFrame(phishing_urls)
    phishing_df['label'] = 1  # Phishing = 1
    phishing_df.to_csv(PHISHING_DATA_PATH, index=False)
    logger.info(f"Saved {len(phishing_df)} phishing URLs to {PHISHING_DATA_PATH}")
    
    # Save legitimate URLs
    legitimate_df = pd.DataFrame(legitimate_urls)
    legitimate_df['label'] = 0  # Legitimate = 0
    legitimate_df.to_csv(LEGITIMATE_DATA_PATH, index=False)
    logger.info(f"Saved {len(legitimate_df)} legitimate URLs to {LEGITIMATE_DATA_PATH}")
    
    # Combine datasets
    combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    combined_df.to_csv(COMBINED_DATA_PATH, index=False)
    
    logger.info(f"Combined dataset: {len(combined_df)} total URLs")
    logger.info(f"  - Phishing: {len(phishing_df)}")
    logger.info(f"  - Legitimate: {len(legitimate_df)}")
    logger.info(f"  - Saved to: {COMBINED_DATA_PATH}")
    
    return combined_df

def main():
    """Main data collection function."""
    logger.info("=" * 60)
    logger.info("DATA COLLECTION FOR MULTIMODAL PHISHING DETECTION")
    logger.info("=" * 60)
    
    # Collect phishing URLs
    phishing_urls = collect_phishing_urls()
    
    # Collect legitimate URLs
    legitimate_urls = collect_legitimate_urls()
    
    # Save datasets
    combined_df = save_datasets(phishing_urls, legitimate_urls)
    
    logger.info("=" * 60)
    logger.info("DATA COLLECTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Dataset ready for preprocessing and model training")
    logger.info(f"Next steps:")
    logger.info(f"  1. Run preprocessing/text_processor.py")
    logger.info(f"  2. Run preprocessing/visual_processor.py")
    logger.info(f"  3. Run scripts/train_model.py")

if __name__ == "__main__":
    main()
