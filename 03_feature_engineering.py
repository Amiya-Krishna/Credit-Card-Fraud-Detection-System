"""
PHASE 4: FEATURE ENGINEERING & PREPROCESSING
==============================================
This script:
1. Creates velocity and behavioral features
2. Builds preprocessing pipeline (OneHotEncoder, StandardScaler)
3. Performs chronological train/validation split
4. Prepares data for model training
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 1. FEATURE ENGINEERING
# ============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from raw data.
    
    Features created:
    - log_amount: Log transformation of amount (reduces skewness)
    - avg_tx_amt_24h: Average transaction amount in last 24h
    - velocity_ratio: Ratio of 1h velocity to 24h average
    - is_weekend: Binary flag for weekend
    - merchant_cat_rare: Flag for rare merchant categories
    """
    logger.info("🔧 Creating engineered features...")
    df = df.copy()
    
    # ---- Numeric Transformations ----
    # Log transformation (handle zeros with log1p = log(1+x))
    df['log_amount'] = np.log1p(df['amount'])
    logger.info("   ✓ log_amount")
    
    # Average transaction amount in last 24h
    df['avg_tx_amt_24h'] = (
        df['prev_24h_amt_card'] / (df['prev_24h_tx_count_card'] + 1e-8)
    )
    df['avg_tx_amt_24h'].fillna(0, inplace=True)
    logger.info("   ✓ avg_tx_amt_24h")
    
    # Velocity ratio: How much did we spend in last 1h vs typical daily average
    df['velocity_ratio'] = (
        df['velocity_amt_1h'] / (df['avg_tx_amt_24h'] + 1e-8)
    )
    df['velocity_ratio'].fillna(0, inplace=True)
    logger.info("   ✓ velocity_ratio")
    
    # ---- Categorical Features ----
    # Weekend flag (5=Friday, 6=Saturday)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    logger.info("   ✓ is_weekend")
    
    # Rare merchant category flag (merchant categories with <50 occurrences)
    cat_counts = df['merchant_cat'].value_counts()
    df['merchant_cat_rare'] = (
        df['merchant_cat'].map(lambda x: int(cat_counts.get(x, 0) < 50))
    )
    logger.info("   ✓ merchant_cat_rare")
    
    # ---- Time-based Ratio Features ----
    # High activity in last hour indicator
    df['high_1h_activity'] = (df['prev_1h_tx_count_card'] > 2).astype(int)
    logger.info("   ✓ high_1h_activity")
    
    logger.info(f"✅ Feature engineering complete! Created {len(df.columns)} total columns")
    return df


# ============================================================================
# 2. DEFINE PREPROCESSING PIPELINE
# ============================================================================

def build_preprocessing_pipeline():
    """
    Build a sklearn ColumnTransformer that:
    - Scales numeric features (StandardScaler)
    - One-hot encodes categorical features
    """
    logger.info("\n🔨 Building preprocessing pipeline...")
    
    # ---- Numeric Columns ----
    NUM_FEATURES = [
        'amount',                      # Original amount
        'log_amount',                  # Log-transformed amount
        'prev_24h_tx_count_card',      # Transactions in 24h
        'prev_24h_amt_card',           # Amount in 24h
        'prev_1h_tx_count_card',       # Transactions in 1h
        'velocity_amt_1h',             # Spending velocity
        'avg_tx_amt_24h',              # Average transaction
        'velocity_ratio',              # Velocity ratio
        'hour',                        # Hour of day (0-23)
        'dayofweek',                   # Day of week (0-6)
    ]
    
    # ---- Categorical Columns ----
    CAT_FEATURES = [
        'merchant_cat',                # Merchant category
        'city',                        # Transaction city
        'country',                     # Country
        'device_type',                 # Device type
        'channel',                     # Transaction channel
        'is_international',            # International flag
        'is_night',                    # Night flag
        'is_weekend',                  # Weekend flag
        'merchant_cat_rare',           # Rare merchant flag
        'high_1h_activity',            # High activity flag
    ]
    
    # ---- Numeric Pipeline ----
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing with median
        ('scaler', StandardScaler()),                   # Standardize (mean=0, std=1)
    ])
    
    # ---- Categorical Pipeline ----
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with mode
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',              # Ignore unknown categories
            sparse_output=False,                  # Dense output (not sparse)
            drop='if_binary'                      # Drop one level for binary features
        )),
    ])
    
    # ---- Combine Both ----
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_FEATURES),
            ('cat', categorical_transformer, CAT_FEATURES),
        ],
        n_jobs=-1,  # Use all available cores
    )
    
    logger.info(f"   ✓ Numeric features: {len(NUM_FEATURES)}")
    logger.info(f"   ✓ Categorical features: {len(CAT_FEATURES)}")
    logger.info("✅ Pipeline built successfully!")
    
    return preprocessor, NUM_FEATURES, CAT_FEATURES


# ============================================================================
# 3. TIME-AWARE TRAIN/VALIDATION SPLIT
# ============================================================================

def chronological_train_val_split(
    df: pd.DataFrame, 
    test_size: float = 0.2,
    drop_target_cols: bool = True
) -> dict:
    """
    Split data chronologically (no future data leakage).
    
    Training set: First 80% (past)
    Validation set: Last 20% (recent)
    
    This prevents data leakage because we always predict the future
    based on historical patterns.
    """
    logger.info("\n⏰ Performing chronological train/val split...")
    
    # Ensure data is sorted by time
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    logger.info(f"   Train: {len(train_df):,} rows ({100*(1-test_size):.0f}%)")
    logger.info(f"   Val:   {len(val_df):,} rows ({100*test_size:.0f}%)")
    logger.info(f"   Train period: {train_df['ts'].min()} → {train_df['ts'].max()}")
    logger.info(f"   Val period:   {val_df['ts'].min()} → {val_df['ts'].max()}")
    
    # Check fraud distribution
    train_fraud_rate = train_df['is_fraud'].mean()
    val_fraud_rate = val_df['is_fraud'].mean()
    logger.info(f"   Train fraud rate: {train_fraud_rate:.2%}")
    logger.info(f"   Val fraud rate:   {val_fraud_rate:.2%}")
    
    # Extract features and target
    drop_cols = ['is_fraud', 'ts', 'tx_id', 'merchant_id_hash', 'card_id_hash']
    
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['is_fraud']
    
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df['is_fraud']
    
    logger.info(f"✅ Split complete! X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'train_df': train_df,
        'val_df': val_df,
    }


# ============================================================================
# 4. APPLY PREPROCESSING
# ============================================================================

def preprocess_data(preprocessor, X_train, X_val):
    """
    Fit preprocessor on training data, transform both train and val.
    """
    logger.info("\n⚙️  Applying preprocessing...")
    
    # Fit on train, transform both
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    
    logger.info(f"   X_train shape: {X_train.shape} → {X_train_scaled.shape}")
    logger.info(f"   X_val shape:   {X_val.shape} → {X_val_scaled.shape}")
    logger.info("✅ Preprocessing complete!")
    
    # Convert to DataFrame for easier handling (optional)
    feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
    
    return X_train_scaled, X_val_scaled


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # ---- Load Data ----
    logger.info("📂 Loading data...")
    df = pd.read_parquet('data/transactions.parquet')
    logger.info(f"✅ Loaded {len(df):,} rows")
    
    # ---- Add Features ----
    df = add_features(df)
    
    # ---- Build Pipeline ----
    preprocessor, num_features, cat_features = build_preprocessing_pipeline()
    
    # ---- Train/Val Split ----
    split_data = chronological_train_val_split(df, test_size=0.2)
    
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_val = split_data['X_val']
    y_val = split_data['y_val']
    
    # ---- Preprocess ----
    X_train_scaled, X_val_scaled = preprocess_data(preprocessor, X_train, X_val)
    
    # ---- Save for Next Phase ----
    import joblib
    
    joblib.dump(X_train_scaled, 'data/X_train_scaled.joblib')
    joblib.dump(X_val_scaled, 'data/X_val_scaled.joblib')
    joblib.dump(y_train, 'data/y_train.joblib')
    joblib.dump(y_val, 'data/y_val.joblib')
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    logger.info("\n✅ All data prepared and saved!")
    logger.info("   X_train_scaled, X_val_scaled, y_train, y_val")
    logger.info("   preprocessor")
    
    print("\n" + "="*70)
    print("✅ PHASE 4 COMPLETE: Feature engineering finished!")
    print("="*70)
    print(f"\nData Summary:")
    print(f"  Training set: {X_train_scaled.shape}")
    print(f"  Validation set: {X_val_scaled.shape}")
    print(f"  Total features: {X_train_scaled.shape[1]}")
    print(f"  Train fraud rate: {y_train.mean():.2%}")
    print(f"  Val fraud rate: {y_val.mean():.2%}")
