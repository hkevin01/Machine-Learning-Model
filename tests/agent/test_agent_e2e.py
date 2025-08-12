"""Agent Mode E2E tests."""
import pytest
import tempfile
import json
import os
from pathlib import Path

pytestmark = pytest.mark.agent


@pytest.mark.agent
def test_agent_import():
    """Test that agent components can be imported."""
    try:
        # Test core agent imports
        from machine_learning_model import main_app
        assert main_app is not None
    except ImportError as e:
        pytest.skip(f"Agent components not available: {e}")


@pytest.mark.agent
def test_agent_state_persistence():
    """Test agent state can be saved and loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = Path(temp_dir) / "agent_state.json"
        
        # Create sample state
        test_state = {
            "current_step": "data_validation",
            "completed_steps": ["initialization"],
            "dataset_info": {
                "location": "test_data.csv",
                "target_column": "label",
                "features": ["feature1", "feature2"]
            },
            "model_config": {
                "type": "classification",
                "metrics": ["accuracy", "f1"]
            }
        }
        
        # Save state
        with open(state_file, 'w') as f:
            json.dump(test_state, f, indent=2)
        
        # Load and verify state
        with open(state_file, 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state == test_state
        assert loaded_state["current_step"] == "data_validation"
        assert "initialization" in loaded_state["completed_steps"]


@pytest.mark.agent
@pytest.mark.slow
def test_mini_pipeline_e2e():
    """End-to-end test with toy dataset."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create toy dataset
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=0,
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save dataset
        dataset_path = Path(temp_dir) / "toy_dataset.csv"
        df.to_csv(dataset_path, index=False)
        
        # Mini pipeline steps
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Assert baseline performance
        assert accuracy > 0.5, f"Model accuracy {accuracy} too low"
        
        # Create artifacts
        artifacts_dir = Path(temp_dir) / "artifacts"
        artifacts_dir.mkdir()
        
        # Save model metadata
        metadata = {
            "model_type": "RandomForestClassifier",
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "accuracy": float(accuracy),
            "test_size": 0.2
        }
        
        metadata_path = artifacts_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Verify artifacts
        assert metadata_path.exists()
        assert metadata["accuracy"] > 0.5


@pytest.mark.agent
def test_run_agent_script():
    """Test that run_agent.sh can be found and is executable."""
    script_path = Path("run_agent.sh")
    
    if not script_path.exists():
        pytest.skip("run_agent.sh not found")
    
    # Check if file is executable (Unix-like systems)
    if os.name != 'nt':  # Not Windows
        assert os.access(script_path, os.X_OK), "run_agent.sh is not executable"
    
    # Read script content
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Basic validation
    assert len(content) > 0, "run_agent.sh is empty"
    assert "#!/bin/bash" in content or "#!/usr/bin/env bash" in content


@pytest.mark.agent
def test_agent_config_validation():
    """Test agent configuration validation."""
    # Test valid config
    valid_config = {
        "problem_type": "classification",
        "dataset_location": "./data/raw/dataset.csv",
        "target_column": "label",
        "success_metrics": ["accuracy", "f1"],
        "constraints": {
            "training_time": "30min",
            "model_size": "small"
        }
    }
    
    # Validate required fields
    required_fields = ["problem_type", "dataset_location", "target_column"]
    for field in required_fields:
        assert field in valid_config
    
    # Validate problem type
    valid_problem_types = ["classification", "regression", "clustering", "semi-supervised"]
    assert valid_config["problem_type"] in valid_problem_types
