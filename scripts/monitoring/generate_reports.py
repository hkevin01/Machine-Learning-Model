#!/usr/bin/env python3
"""
Generate monitoring reports using Evidently for data drift detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
except ImportError:
    print("❌ Evidently not installed. Install with: pip install evidently")
    sys.exit(1)


def generate_sample_data():
    """Generate sample reference and current datasets for demonstration."""
    np.random.seed(42)
    
    # Reference data (training period)
    n_samples = 1000
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(2, 1.5, n_samples),
        'feature_3': np.random.exponential(1, n_samples),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    })
    
    # Current data (production period with some drift)
    current_data = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, n_samples),  # Mean shift
        'feature_2': np.random.normal(2, 2.0, n_samples),     # Variance increase
        'feature_3': np.random.exponential(1.3, n_samples),   # Distribution change
        'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.3, 0.3, 0.3, 0.1]),  # New category
        'target': np.random.binomial(1, 0.4, n_samples)       # Target drift
    })
    
    return reference_data, current_data


def generate_drift_report(reference_data, current_data, output_dir):
    """Generate data drift report using Evidently."""
    
    # Define column mapping
    column_mapping = ColumnMapping(
        target='target',
        numerical_features=['feature_1', 'feature_2', 'feature_3'],
        categorical_features=['categorical_feature']
    )
    
    # Create data drift report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])
    
    # Run the report
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = output_dir / f"data_drift_report_{timestamp}.html"
    data_drift_report.save_html(str(html_path))
    
    # Save JSON report for programmatic access
    json_path = output_dir / f"data_drift_report_{timestamp}.json"
    data_drift_report.save_json(str(json_path))
    
    print(f"✅ Reports saved:")
    print(f"   HTML: {html_path}")
    print(f"   JSON: {json_path}")
    
    return html_path, json_path


def main():
    """Main function to generate monitoring reports."""
    parser = argparse.ArgumentParser(description="Generate ML monitoring reports")
    parser.add_argument(
        "--reference-data",
        type=str,
        help="Path to reference dataset CSV"
    )
    parser.add_argument(
        "--current-data", 
        type=str,
        help="Path to current dataset CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use generated sample data for demonstration"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Generating ML monitoring reports...")
    
    if args.use_sample_data or (not args.reference_data or not args.current_data):
        print("📊 Using sample data for demonstration")
        reference_data, current_data = generate_sample_data()
    else:
        print(f"📂 Loading reference data from: {args.reference_data}")
        print(f"📂 Loading current data from: {args.current_data}")
        
        try:
            reference_data = pd.read_csv(args.reference_data)
            current_data = pd.read_csv(args.current_data)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    # Generate reports
    try:
        html_path, json_path = generate_drift_report(
            reference_data, current_data, output_dir
        )
        
        print(f"\n📋 Report Summary:")
        print(f"   Reference data shape: {reference_data.shape}")
        print(f"   Current data shape: {current_data.shape}")
        print(f"   Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
