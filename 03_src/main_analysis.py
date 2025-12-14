"""
Project: Menu vs. Delivery Experience Analysis using Food App Reviews

Execution:
1. Place CSV in 01_data/raw/zomato_swiggy_reviews.csv
2. Run from project root:
   python 03_src/main_analysis.py

This version uses BULLETPROOF CSV parsing for real-world noisy data.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, '01_data', 'raw', 'zomato_swiggy_reviews.csv')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, '01_data', 'processed', 'reviews_with_scores.csv')
FIGURES_PATH = os.path.join(PROJECT_ROOT, '04_reports', 'figures')

os.makedirs(FIGURES_PATH, exist_ok=True)

# -----------------------------------------------------------
# NLTK SETUP
# -----------------------------------------------------------
try:
    STOPWORDS_SET = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS_SET = set(stopwords.words('english'))

# -----------------------------------------------------------
# STEP 1: DATA ACQUISITION (ROBUST LINE PARSING)
# -----------------------------------------------------------
print("--- STEP 1: Data Acquisition ---")

rows = []

try:
    with open(RAW_DATA_PATH, encoding='utf-8', errors='ignore') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            # Remove quotes
            line = line.replace('"', '').replace("'", '')

            # Must contain at least one comma
            if ',' not in line:
                continue

            # Split ONLY at last comma
            review, rating = line.rsplit(',', 1)
            rating = rating.strip()

            if rating.isdigit():
                rows.append({
                    'review_text': review.strip(),
                    'score': int(rating)
                })

except FileNotFoundError:
    print(f"ERROR: File not found -> {RAW_DATA_PATH}")
    sys.exit(1)

if not rows:
    print("FATAL ERROR: No valid rows extracted. Check CSV content.")
    sys.exit(1)

df = pd.DataFrame(rows)
print(f"Valid rows loaded: {len(df)}")

# -----------------------------------------------------------
# STEP 2: TEXT CLEANING
# -----------------------------------------------------------
print("\n--- STEP 2: Text Preprocessing ---")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(w for w in text.split() if w not in STOPWORDS_SET and len(w) > 1)
    return text

df['cleaned_review'] = df['review_text'].apply(clean_text)
df = df[df['cleaned_review'].str.len() > 10]

print(f"Rows after cleaning: {len(df)}")

# -----------------------------------------------------------
# STEP 3: NLP FEATURE ENGINEERING
# -----------------------------------------------------------
print("\n--- STEP 3: NLP Feature Engineering ---")

MENU_KEYWORDS = [
    'taste','flavor','spicy','fresh','portion','dish','ingredients','texture',
    'delicious','raw','salty','menu','food','quality','rotten','cooked',
    'chicken','paneer','pizza','burger','curry'
]

DELIVERY_KEYWORDS = [
    'late','delay','cold','packaging','spilled','time','wrong','cancelled',
    'delivery','wait','damaged','courier','rude','tracking','slow','fast'
]

def calculate_scores(text):
    words = text.split()
    menu_score = sum(1 for w in words if w in MENU_KEYWORDS)
    delivery_score = sum(1 for w in words if w in DELIVERY_KEYWORDS)
    return menu_score, delivery_score

df[['menu_score', 'delivery_score']] = df['cleaned_review'].apply(
    lambda x: pd.Series(calculate_scores(x))
)

def get_dominant(row):
    if row['menu_score'] > row['delivery_score'] and row['menu_score'] > 0:
        return 'MENU_DOMINATED'
    elif row['delivery_score'] > row['menu_score'] and row['delivery_score'] > 0:
        return 'DELIVERY_DOMINATED'
    else:
        return 'NEUTRAL'

df['dominant_complaint'] = df.apply(get_dominant, axis=1)

print(df[['menu_score','delivery_score','dominant_complaint']].head())

# Save processed data
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Processed data saved to {PROCESSED_DATA_PATH}")

# -----------------------------------------------------------
# STEP 4: ANALYSIS & VISUALIZATION
# -----------------------------------------------------------
print("\n--- STEP 4: Analysis and Visualization ---")

analysis_df = df[df['dominant_complaint'] != 'NEUTRAL']

menu_mean = analysis_df[analysis_df['dominant_complaint']=='MENU_DOMINATED']['score'].mean()
delivery_mean = analysis_df[analysis_df['dominant_complaint']=='DELIVERY_DOMINATED']['score'].mean()

plt.figure(figsize=(9,6))
sns.boxplot(
    x='dominant_complaint',
    y='score',
    data=analysis_df,
    palette={'MENU_DOMINATED':'#1f77b4','DELIVERY_DOMINATED':'#ff7f0e'}
)

plt.title('Impact of Complaint Type on Ratings', fontsize=14)
plt.xlabel('Complaint Type')
plt.ylabel('Star Rating')
plt.grid(axis='y', alpha=0.4)

if not np.isnan(menu_mean):
    plt.axhline(menu_mean, linestyle='--', label=f'Menu Mean: {menu_mean:.2f}')
if not np.isnan(delivery_mean):
    plt.axhline(delivery_mean, linestyle='--', label=f'Delivery Mean: {delivery_mean:.2f}')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'boxplot_ratings_impact.png'))
plt.close()

print(f"Figures saved to {FIGURES_PATH}")

if not np.isnan(menu_mean) and not np.isnan(delivery_mean):
    print(f"Final Insight: Delivery issues reduce ratings by {(menu_mean - delivery_mean):.2f} points")
else:
    print("Final Insight: Insufficient classified data")
