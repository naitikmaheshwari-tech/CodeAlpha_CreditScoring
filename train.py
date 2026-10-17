"""
Credit Scoring Model - Training Script
======================================
Automated pipeline to:
1. Synthesize realistic credit scoring dataset (or load existing CSV)
2. Handle missing values, scaling, and feature preprocessing
3. Train an optimized Random Forest Classifier
4. Benchmark against Gradient Boosting and Logistic Regression
5. Evaluate Precision, Recall, F1-Score, and ROC-AUC
6. Persist model artifacts, preprocessor, and metrics for deployment

Usage:
    python train.py
    python train.py --samples 8000 --seed 42
    python train.py --data-path data/custom_data.csv

Author: Senior ML Engineer & Python Developer (CodeAlpha Internship Project)
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_generator import generate_synthetic_credit_data
from src.model_trainer import train_and_evaluate_model, FEATURE_COLUMNS, TARGET_COLUMN


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate Credit Scoring Model")
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic records to generate (default: 5000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test set fraction (default: 0.20)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to existing CSV dataset; if not provided, synthetic data will be generated"
    )
    parser.add_argument(
        "--save-data-path",
        type=str,
        default="data/credit_scoring_dataset.csv",
        help="Path where generated dataset is saved (default: data/credit_scoring_dataset.csv)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts (default: models)"
    )
    return parser.parse_args()


def print_banner():
    banner = """
========================================================================
       CODEALPHA MACHINE LEARNING INTERNSHIP - CREDIT SCORING MODEL     
               End-to-End ML Training & Evaluation Pipeline             
========================================================================
"""
    print(banner)


def print_metrics_table(metrics: dict):
    print("\n" + "=" * 60)
    print("      PRIMARY MODEL EVALUATION: RANDOM FOREST CLASSIFIER        ")
    print("=" * 60)
    print(f"  Accuracy           : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision          : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"  Recall             : {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score           : {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)")
    print(f"  ROC-AUC Score      : {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
    print(f"  PR-AUC Score       : {metrics['pr_auc']:.4f}")
    print("-" * 60)

    cm = metrics["confusion_matrix"]
    print("  Confusion Matrix:")
    print(f"    True Negative (High Risk):  {cm[0][0]:<6} | False Positive: {cm[0][1]}")
    print(f"    False Negative (Denied):    {cm[1][0]:<6} | True Positive (Approved): {cm[1][1]}")
    print("=" * 60)


def print_comparison_table(comparisons: list):
    print("\n" + "=" * 70)
    print(f"{'Model':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'ROC-AUC':<8}")
    print("-" * 70)
    for c in comparisons:
        print(f"{c['Model']:<28} | {c['Accuracy']:<9.4f} | {c['Precision']:<9.4f} | {c['Recall']:<7.4f} | {c['F1-Score']:<7.4f} | {c['ROC-AUC']:<8.4f}")
    print("=" * 70)


def print_feature_importance_table(feat_imp: pd.DataFrame):
    print("\n" + "=" * 55)
    print("               TOP FEATURE IMPORTANCES                  ")
    print("=" * 55)
    for idx, row in feat_imp.iterrows():
        bar = "#" * int(row['Importance'] * 50)
        print(f"  {row['Feature']:<26} | {row['Importance']:.4f} | {bar}")
    print("=" * 55)


def main():
    args = parse_args()
    print_banner()

    # Step 1: Load or Generate Dataset
    if args.data_path and os.path.exists(args.data_path):
        print(f"[*] Loading existing dataset from: {args.data_path}")
        df = pd.read_csv(args.data_path)
    else:
        print(f"[*] Generating {args.samples:,} synthetic consumer credit records (seed={args.seed})...")
        df = generate_synthetic_credit_data(
            n_samples=args.samples,
            random_state=args.seed,
            introduce_missing=True,
            missing_rate=0.015
        )
        os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)
        df.to_csv(args.save_data_path, index=False)
        print(f"[OK] Synthetic dataset successfully saved to: {args.save_data_path}")

    # Display dataset info
    print(f"[*] Total Records: {len(df):,} | Features: {len(FEATURE_COLUMNS)}")
    dist = df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    print(f"[*] Target Distribution: 0 (High Risk) = {dist.get(0, 0)*100:.1f}%, 1 (Creditworthy) = {dist.get(1, 0)*100:.1f}%")
    print(f"[*] Missing Values Handled: {df.isnull().sum().sum()} missing entries across features")

    # Step 2: Model Training & Comprehensive Evaluation
    print("\n[*] Starting Machine Learning Training Pipeline...")
    results = train_and_evaluate_model(
        df=df,
        test_size=args.test_size,
        random_state=args.seed,
        save_artifacts=True,
        artifacts_dir=args.artifacts_dir
    )

    # Step 3: Present Formatted Evaluation Results
    print_metrics_table(results["metrics"])
    print_comparison_table(results["comparison"])
    print_feature_importance_table(results["feature_importances"])

    print("\n[SUCCESS] Pipeline execution complete! Ready for Streamlit Web Application.")
    print("Run the dashboard using: streamlit run app.py\n")


if __name__ == "__main__":
    main()
