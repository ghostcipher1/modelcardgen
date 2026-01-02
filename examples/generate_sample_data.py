#!/usr/bin/env python
"""
Generate sample email evaluation dataset with 100,000 rows.

This script creates a realistic evaluation dataset for email spam classification,
saved as both CSV and Parquet formats. The data includes email metadata, features,
and ground truth labels for model evaluation.

Usage:
    python generate_sample_data.py

Output:
    - sample_email_evaluation_100k.csv (CSV format)
    - sample_email_evaluation_100k.parquet (Parquet format for S3 upload)
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_sample_email_data(num_samples: int = 100000) -> pd.DataFrame:
    """
    Generate synthetic email evaluation dataset.
    
    Args:
        num_samples: Number of email samples to generate (default: 100,000)
        
    Returns:
        DataFrame with email features and labels
    """
    random.seed(42)
    np.random.seed(42)
    
    print(f"Generating {num_samples:,} sample emails...")
    
    data = {
        "email_id": [f"email_{i:06d}" for i in range(num_samples)],
        "timestamp": [
            (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
            for _ in range(num_samples)
        ],
        "sender_domain": np.random.choice(
            [
                "gmail.com", "outlook.com", "yahoo.com", "company.com",
                "university.edu", "suspicious-domain.ru", "phishing-site.tk",
                "noreply@promo.com", "alerts@bank.com", "newsletter.example.com"
            ],
            num_samples
        ),
        "recipient_count": np.random.choice([1, 2, 5, 10, 50, 100], num_samples, p=[0.4, 0.3, 0.15, 0.1, 0.04, 0.01]),
        "attachment_count": np.random.choice([0, 1, 2, 5], num_samples, p=[0.7, 0.2, 0.08, 0.02]),
        "subject_length": np.random.gamma(shape=2, scale=15, size=num_samples).astype(int),
        "body_length": np.random.gamma(shape=2, scale=200, size=num_samples).astype(int),
        "html_content": np.random.choice([True, False], num_samples, p=[0.3, 0.7]),
        "contains_links": np.random.choice([True, False], num_samples, p=[0.4, 0.6]),
        "contains_suspicious_keywords": np.random.choice(
            [True, False], 
            num_samples, 
            p=[0.15, 0.85]
        ),
        "sender_reputation_score": np.random.beta(a=2, b=5, size=num_samples),
        "domain_age_days": np.random.exponential(scale=365, size=num_samples).astype(int),
        "spf_valid": np.random.choice([True, False], num_samples, p=[0.85, 0.15]),
        "dkim_valid": np.random.choice([True, False], num_samples, p=[0.8, 0.2]),
        "dmarc_policy": np.random.choice(
            ["none", "quarantine", "reject", "missing"],
            num_samples,
            p=[0.3, 0.2, 0.3, 0.2]
        ),
        "from_header_matches_spf": np.random.choice([True, False], num_samples, p=[0.9, 0.1]),
        "reply_to_count": np.random.choice([0, 1, 2], num_samples, p=[0.6, 0.35, 0.05]),
        "previous_contact": np.random.choice([True, False], num_samples, p=[0.7, 0.3]),
        "account_age_days": np.random.exponential(scale=1000, size=num_samples).astype(int),
        "model_confidence": np.random.beta(a=3, b=2, size=num_samples),
    }
    
    df = pd.DataFrame(data)
    
    spam_probability = (
        0.1 * (df["contains_suspicious_keywords"].astype(int)) +
        0.15 * (1 - df["sender_reputation_score"]) +
        0.1 * (df["recipient_count"] > 50).astype(int) +
        0.1 * (~df["spf_valid"]).astype(int) +
        0.1 * (~df["dkim_valid"]).astype(int) +
        0.1 * (~df["from_header_matches_spf"]).astype(int) +
        0.2 * (~df["previous_contact"]).astype(int) +
        0.05 * (df["dmarc_policy"] == "missing").astype(int)
    )
    
    df["spam_label"] = (np.random.random(num_samples) < spam_probability).astype(int)
    
    df["predicted_label"] = np.where(
        df["model_confidence"] > 0.5,
        (np.random.random(num_samples) < df["model_confidence"]).astype(int),
        np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    )
    
    df["prediction_correct"] = (df["spam_label"] == df["predicted_label"]).astype(int)
    
    print(f"Generated dataset shape: {df.shape}")
    print(f"Spam ratio: {df['spam_label'].mean():.2%}")
    print(f"Model accuracy: {df['prediction_correct'].mean():.2%}")
    
    return df


def main():
    """Generate and save sample data in CSV and Parquet formats."""
    df = generate_sample_email_data(num_samples=100000)
    
    csv_path = "sample_email_evaluation_100k.csv"
    parquet_path = "sample_email_evaluation_100k.parquet"
    
    print(f"\nSaving to CSV: {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV file saved ({csv_path})")
    
    print(f"\nSaving to Parquet: {parquet_path}")
    df.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"[OK] Parquet file saved ({parquet_path})")
    
    csv_size_mb = __import__("os").path.getsize(csv_path) / (1024 ** 2)
    parquet_size_mb = __import__("os").path.getsize(parquet_path) / (1024 ** 2)
    
    print(f"\nFile sizes:")
    print(f"  CSV:     {csv_size_mb:.2f} MB")
    print(f"  Parquet: {parquet_size_mb:.2f} MB (compression: {parquet_size_mb/csv_size_mb:.1%})")
    
    print(f"\nSample data preview:")
    print(df.head(10))
    
    print(f"\nDataset summary:")
    print(df.describe())


if __name__ == "__main__":
    main()
