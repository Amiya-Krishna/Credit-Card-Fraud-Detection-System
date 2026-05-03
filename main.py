"""
MAIN ORCHESTRATION SCRIPT
=========================
Run this script to execute the entire fraud detection pipeline end-to-end.

Phases:
  1. Data ingestion & validation
  2. EDA (Exploratory Data Analysis)
  3. Feature engineering
  4. Model training & hyperparameter tuning
  5. Model evaluation
  6. Generate predictions
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(phase_num, phase_name):
    """Print formatted phase header."""
    print("\n" + "="*70)
    print(f"PHASE {phase_num}: {phase_name}")
    print("="*70 + "\n")


def run_phase(script_name, phase_num, phase_name):
    """Run a Python script as a phase."""
    print_header(phase_num, phase_name)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        logger.info(f"✅ Phase {phase_num} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Phase {phase_num} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error(f"❌ Script not found: {script_name}")
        return False


def create_directories():
    """Create necessary directories."""
    dirs = ['data', 'models', 'outputs', 'notebooks']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        logger.info(f"✓ {d}/")


def main():
    """Execute full pipeline."""
    print("\n" + "#"*70)
    print("# CREDIT CARD FRAUD DETECTION SYSTEM - FULL PIPELINE")
    print("#"*70 + "\n")
    
    # Create directories
    logger.info("📁 Creating directories...")
    create_directories()
    
    # Phase 1: Data Ingestion
    if not run_phase('01_ingest.py', 1, 'DATA INGESTION'):
        logger.error("Pipeline stopped at Phase 1")
        return False
    
    # Phase 2: EDA
    if not run_phase('02_eda.py', 2, 'EXPLORATORY DATA ANALYSIS'):
        logger.error("Pipeline stopped at Phase 2")
        return False
    
    # Phase 3: Feature Engineering
    if not run_phase('03_feature_engineering.py', 3, 'FEATURE ENGINEERING'):
        logger.error("Pipeline stopped at Phase 3")
        return False
    
    # Phase 4: Model Training
    if not run_phase('04_model_training.py', 4, 'MODEL TRAINING & TUNING'):
        logger.error("Pipeline stopped at Phase 4")
        return False
    
    # Phase 5: Evaluation
    if not run_phase('05_evaluation.py', 5, 'MODEL EVALUATION'):
        logger.error("Pipeline stopped at Phase 5")
        return False
    
    # Success!
    print_header("✅", "PIPELINE COMPLETE!")
    print("""
🎉 All phases completed successfully!

📊 Generated Outputs:
   ✓ data/transactions.parquet
   ✓ data/X_train_scaled.joblib
   ✓ data/X_val_scaled.joblib
   ✓ models/fraud_model_bundle.joblib
   ✓ outputs/*.png (visualizations)

🚀 Next Steps:
   1. Review evaluation metrics in outputs/
   2. Start API server:
      $ python -m uvicorn app:app --reload --port 8000
   
   3. Test predictions at: http://localhost:8000/docs
   
   4. Deploy to production!

📚 Documentation:
   See README.md for full details
""")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
