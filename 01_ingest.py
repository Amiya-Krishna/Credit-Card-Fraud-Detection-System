"""
PHASE 3: DATA INGESTION & VALIDATION
=====================================
This script:
1. Generates synthetic transaction dataset (or loads public data)
2. Validates schema & types
3. Handles data quality issues
4. Saves to Parquet for efficient storage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 1. DEFINE SCHEMA
# ============================================================================

SCHEMA = {
    'tx_id': 'string',                      # Unique transaction ID
    'ts': 'string',                         # Timestamp (will be converted to datetime)
    'amount': 'float64',                    # Transaction amount
    'merchant_cat': 'category',             # Merchant category (e.g., 'grocery', 'jewelry')
    'merchant_id_hash': 'string',           # Hashed merchant ID
    'card_id_hash': 'string',               # Hashed card ID
    'city': 'category',                     # Transaction city
    'country': 'category',                  # Transaction country
    'device_type': 'category',              # Device (e.g., 'mobile', 'web', 'chip')
    'channel': 'category',                  # Channel (e.g., 'online', 'atm', 'pos')
    'hour': 'int64',                        # Hour of transaction (0-23)
    'dayofweek': 'int64',                   # Day of week (0-6, 0=Monday)
    'prev_24h_tx_count_card': 'float64',    # Num transactions in last 24h for this card
    'prev_24h_amt_card': 'float64',         # Total amount in last 24h for this card
    'prev_1h_tx_count_card': 'float64',     # Num transactions in last 1h for this card
    'velocity_amt_1h': 'float64',           # Total amount in last 1h for this card
    'is_international': 'bool',             # Is transaction from foreign country?
    'is_night': 'bool',                     # Is transaction between 10 PM - 6 AM?
    'is_fraud': 'int64',                    # Target: 0=Normal, 1=Fraud
}

MERCHANTS = ['grocery', 'clothing', 'jewelry', 'electronics', 'travel', 'gas', 
             'restaurant', 'hotel', 'utility', 'entertainment']
CITIES = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'pune', 'kolkata', 'dubai', 
          'london', 'new_york', 'singapore']
COUNTRIES = ['IN', 'US', 'UK', 'AE', 'SG', 'JP', 'CA']
DEVICES = ['mobile', 'web', 'chip', 'atm']
CHANNELS = ['online', 'pos', 'atm']


# ============================================================================
# 2. GENERATE SYNTHETIC DATA
# ============================================================================

def generate_synthetic_data(n_rows: int = 100000, fraud_rate: float = 0.015) -> pd.DataFrame:
    """
    Generate synthetic transaction dataset.
    
    Args:
        n_rows: Number of transactions to generate
        fraud_rate: Fraction of fraudulent transactions (typically 0.1% - 2%)
    
    Returns:
        pd.DataFrame with transaction data
    """
    logger.info(f"🔄 Generating {n_rows:,} synthetic transactions with {fraud_rate:.1%} fraud rate...")
    
    np.random.seed(42)
    n_fraud = int(n_rows * fraud_rate)
    n_normal = n_rows - n_fraud
    
    data = []
    base_time = datetime.now() - timedelta(days=30)  # 30 days of history
    
    # ---- Generate NORMAL transactions ----
    for i in range(n_normal):
        tx = {
            'tx_id': f'TX_{i:08d}',
            'ts': base_time + timedelta(seconds=int(np.random.exponential(600))),
            'amount': np.random.lognormal(4, 2),  # Log-normal distribution
            'merchant_cat': np.random.choice(MERCHANTS),
            'merchant_id_hash': f"M_{np.random.randint(1000, 9999)}",
            'card_id_hash': f"C_{np.random.randint(1000, 9999)}",
            'city': np.random.choice(CITIES),
            'country': np.random.choice(COUNTRIES, p=[0.7, 0.15, 0.05, 0.05, 0.03, 0.01, 0.01]),
            'device_type': np.random.choice(DEVICES, p=[0.4, 0.35, 0.15, 0.1]),
            'channel': np.random.choice(CHANNELS, p=[0.5, 0.35, 0.15]),
            'hour': np.random.randint(0, 24),
            'dayofweek': np.random.randint(0, 7),
            'prev_24h_tx_count_card': max(0, np.random.poisson(3)),
            'prev_24h_amt_card': np.random.gamma(2, 500),
            'prev_1h_tx_count_card': max(0, np.random.poisson(0.5)),
            'velocity_amt_1h': np.random.gamma(1, 200),
            'is_international': np.random.choice([True, False], p=[0.15, 0.85]),
            'is_night': np.random.choice([True, False], p=[0.20, 0.80]),
            'is_fraud': 0,
        }
        data.append(tx)
    
    # ---- Generate FRAUDULENT transactions ----
    for i in range(n_normal, n_rows):
        tx = {
            'tx_id': f'TX_{i:08d}',
            'ts': base_time + timedelta(seconds=int(np.random.exponential(600))),
            'amount': np.random.lognormal(5, 1.5),  # Higher amounts for fraud
            'merchant_cat': np.random.choice(['jewelry', 'electronics', 'travel'], p=[0.4, 0.4, 0.2]),
            'merchant_id_hash': f"M_{np.random.randint(1000, 9999)}",
            'card_id_hash': f"C_{np.random.randint(1000, 9999)}",
            'city': np.random.choice(CITIES),
            'country': np.random.choice(COUNTRIES, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]),
            'device_type': np.random.choice(DEVICES, p=[0.2, 0.3, 0.3, 0.2]),
            'channel': np.random.choice(['online', 'pos'], p=[0.7, 0.3]),
            'hour': np.random.choice([2, 3, 4, 5] + list(range(24))),  # More night time
            'dayofweek': np.random.randint(0, 7),
            'prev_24h_tx_count_card': np.random.poisson(8),  # High velocity
            'prev_24h_amt_card': np.random.gamma(3, 800),
            'prev_1h_tx_count_card': np.random.poisson(3),  # Multiple in short time
            'velocity_amt_1h': np.random.gamma(2, 500),
            'is_international': np.random.choice([True, False], p=[0.5, 0.5]),
            'is_night': np.random.choice([True, False], p=[0.6, 0.4]),
            'is_fraud': 1,
        }
        data.append(tx)
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    logger.info(f"✅ Generated {len(df):,} transactions | Fraud: {df['is_fraud'].sum():,} | Normal: {(1-df['is_fraud']).sum().astype(int):,}")
    return df


# ============================================================================
# 3. LOAD & VALIDATE DATA
# ============================================================================

def validate_schema(df: pd.DataFrame, schema: dict) -> bool:
    """
    Validate DataFrame columns and types against schema.
    """
    # Check columns exist
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        logger.warning(f"⚠️  Missing columns: {missing_cols}")
        return False
    
    # Check types (will cast if needed)
    for col, dtype in schema.items():
        if dtype == 'string' or dtype == 'category':
            continue  # Skip strings
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            logger.error(f"❌ Type conversion error for {col}: {e}")
            return False
    
    logger.info(f"✅ Schema validation passed | {len(df.columns)} columns")
    return True


def ingest_data(path: str = None, generate: bool = True) -> pd.DataFrame:
    """
    Load or generate transaction data with validation.
    
    Args:
        path: Path to CSV file (if loading existing)
        generate: Whether to generate synthetic data
    
    Returns:
        Validated DataFrame
    """
    if generate:
        df = generate_synthetic_data(n_rows=100000, fraud_rate=0.015)
    else:
        if path is None:
            raise ValueError("Path required if not generating data")
        logger.info(f"📂 Loading data from {path}...")
        df = pd.read_csv(path)
    
    # Validate schema
    if not validate_schema(df, SCHEMA):
        raise ValueError("Schema validation failed")
    
    # Convert timestamp
    df['ts'] = pd.to_datetime(df['ts'])
    
    # Sort by time (important for chronological splits)
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Check for duplicates
    n_dupes = df.duplicated(subset=['tx_id']).sum()
    if n_dupes > 0:
        logger.warning(f"⚠️  Removing {n_dupes} duplicate transactions")
        df = df.drop_duplicates(subset=['tx_id'], keep='first')
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"⚠️  Missing values:\n{missing[missing > 0]}")
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
    
    logger.info(f"✅ Data ingestion complete | Shape: {df.shape}")
    logger.info(f"   Fraud Rate: {df['is_fraud'].mean():.2%}")
    logger.info(f"   Date Range: {df['ts'].min()} to {df['ts'].max()}")
    
    return df


# ============================================================================
# 4. SAVE DATA
# ============================================================================

def save_data(df: pd.DataFrame, output_path: str = 'data/transactions.parquet'):
    """
    Save DataFrame to Parquet (efficient columnar format).
    """
    df.to_parquet(output_path, index=False, compression='snappy')
    logger.info(f"✅ Data saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Generate or load data
    df = ingest_data(generate=True)
    
    # Display info
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nFraud Distribution:\n{df['is_fraud'].value_counts()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    # Save
    save_data(df)
    print("\n✅ Phase 3 Complete: Data ingestion finished!")
