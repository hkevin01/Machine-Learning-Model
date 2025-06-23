"""
ML Agent - Intelligent Machine Learning Workflow Assistant

This module provides an AI agent that guides users through the complete
machine learning pipeline, providing recommendations and automating repetitive tasks.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStepStatus(Enum):
    """Status of a workflow step."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    DATA_COLLECTION = "data_collection"
    DATA_PREPROCESSING = "data_preprocessing"
    EXPLORATORY_DATA_ANALYSIS = "exploratory_data_analysis"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_SPLITTING = "data_splitting"
    ALGORITHM_SELECTION = "algorithm_selection"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_DEPLOYMENT = "model_deployment"
    MONITORING = "monitoring"


@dataclass
class WorkflowStep:
    """Represents a single step in the ML workflow."""
    
    name: str
    step_type: WorkflowStepType
    description: str
    status: WorkflowStepStatus = WorkflowStepStatus.NOT_STARTED
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 10
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def start(self):
        """Mark the step as started."""
        self.status = WorkflowStepStatus.IN_PROGRESS
        self.started_at = datetime.now()
        logger.info(f"Started workflow step: {self.name}")
    
    def complete(self):
        """Mark the step as completed."""
        self.status = WorkflowStepStatus.COMPLETED
        self.progress = 1.0
        self.completed_at = datetime.now()
        logger.info(f"Completed workflow step: {self.name}")
    
    def fail(self, error_message: str):
        """Mark the step as failed."""
        self.status = WorkflowStepStatus.FAILED
        self.error_message = error_message
        logger.error(f"Failed workflow step: {self.name} - {error_message}")
    
    def skip(self, reason: str = ""):
        """Mark the step as skipped."""
        self.status = WorkflowStepStatus.SKIPPED
        self.metadata["skip_reason"] = reason
        logger.info(f"Skipped workflow step: {self.name} - {reason}")


class MLAgent:
    """
    Intelligent ML Agent that guides users through the machine learning workflow.
    
    The agent provides:
    - Step-by-step guidance through the ML pipeline
    - Intelligent recommendations based on data characteristics
    - Automated execution of routine tasks
    - Progress tracking and state management
    - Error handling and recovery suggestions
    """
    
    def __init__(self, project_name: str = "ml_project", workspace_dir: str = ".", auto_save: bool = True):
        """
        Initialize the ML Agent.
        
        Args:
            project_name: Name of the ML project
            workspace_dir: Directory to store project files
            auto_save: Whether to automatically save progress
        """
        self.project_name = project_name
        self.workspace_dir = workspace_dir
        self.auto_save = auto_save
        
        # Workflow state
        self.current_step_index = 0
        self.steps: List[WorkflowStep] = []
        self.project_metadata: Dict[str, Any] = {}
        self.recommendations: List[str] = []
        
        # Initialize default workflow
        self._initialize_default_workflow()
        
        # Load existing state if available
        self._load_state()
    
    def _initialize_default_workflow(self):
        """Initialize the default ML workflow steps."""
        self.steps = [
            WorkflowStep(
                name="Data Collection",
                step_type=WorkflowStepType.DATA_COLLECTION,
                description="Gather and organize raw data for the ML project",
                estimated_time_minutes=30
            ),
            WorkflowStep(
                name="Data Preprocessing", 
                step_type=WorkflowStepType.DATA_PREPROCESSING,
                description="Clean, validate, and prepare data for analysis",
                dependencies=["Data Collection"],
                estimated_time_minutes=45
            ),
            WorkflowStep(
                name="Exploratory Data Analysis",
                step_type=WorkflowStepType.EXPLORATORY_DATA_ANALYSIS,
                description="Analyze data patterns, distributions, and relationships",
                dependencies=["Data Preprocessing"],
                estimated_time_minutes=60
            ),
            WorkflowStep(
                name="Feature Engineering",
                step_type=WorkflowStepType.FEATURE_ENGINEERING,
                description="Create, select, and transform features for modeling",
                dependencies=["Exploratory Data Analysis"],
                estimated_time_minutes=40
            ),
            WorkflowStep(
                name="Data Splitting",
                step_type=WorkflowStepType.DATA_SPLITTING,
                description="Split data into training, validation, and test sets",
                dependencies=["Feature Engineering"],
                estimated_time_minutes=15
            ),
            WorkflowStep(
                name="Algorithm Selection",
                step_type=WorkflowStepType.ALGORITHM_SELECTION,
                description="Choose appropriate ML algorithms based on problem type and data",
                dependencies=["Data Splitting"],
                estimated_time_minutes=30
            ),
            WorkflowStep(
                name="Model Training",
                step_type=WorkflowStepType.MODEL_TRAINING,
                description="Train selected models on the training data",
                dependencies=["Algorithm Selection"],
                estimated_time_minutes=90
            ),
            WorkflowStep(
                name="Model Evaluation",
                step_type=WorkflowStepType.MODEL_EVALUATION,
                description="Evaluate model performance using validation and test data",
                dependencies=["Model Training"],
                estimated_time_minutes=30
            ),
            WorkflowStep(
                name="Hyperparameter Tuning",
                step_type=WorkflowStepType.HYPERPARAMETER_TUNING,
                description="Optimize model hyperparameters for better performance",
                dependencies=["Model Evaluation"],
                estimated_time_minutes=120
            ),
            WorkflowStep(
                name="Model Deployment",
                step_type=WorkflowStepType.MODEL_DEPLOYMENT,
                description="Deploy the final model for production use",
                dependencies=["Hyperparameter Tuning"],
                estimated_time_minutes=60
            ),
            WorkflowStep(
                name="Monitoring",
                step_type=WorkflowStepType.MONITORING,
                description="Monitor model performance and data drift in production",
                dependencies=["Model Deployment"],
                estimated_time_minutes=0  # Ongoing
            )
        ]
    
    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get the current workflow step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def get_next_step(self) -> Optional[WorkflowStep]:
        """Get the next workflow step."""
        next_index = self.current_step_index + 1
        if next_index < len(self.steps):
            return self.steps[next_index]
        return None
    
    def get_progress(self) -> Tuple[int, int, float]:
        """
        Get overall workflow progress.
        
        Returns:
            Tuple of (completed_steps, total_steps, progress_percentage)
        """
        completed_steps = sum(1 for step in self.steps if step.status == WorkflowStepStatus.COMPLETED)
        total_steps = len(self.steps)
        progress_percentage = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
        return completed_steps, total_steps, progress_percentage
    
    def get_recommendations(self) -> List[str]:
        """Get AI-generated recommendations for the current step."""
        current_step = self.get_current_step()
        if not current_step:
            return ["Workflow completed! Consider starting a new project."]
        
        # Generate context-aware recommendations
        recommendations = []
        
        if current_step.step_type == WorkflowStepType.DATA_COLLECTION:
            recommendations.extend([
                "Ensure data quality by checking for completeness and accuracy",
                "Document data sources and collection methodology",
                "Consider data privacy and ethical implications",
                "Plan for data versioning and backup strategies"
            ])
        elif current_step.step_type == WorkflowStepType.DATA_PREPROCESSING:
            recommendations.extend([
                "Handle missing values appropriately for your use case",
                "Identify and address outliers based on domain knowledge",
                "Ensure consistent data formats and encodings",
                "Create a preprocessing pipeline for reproducibility"
            ])
        elif current_step.step_type == WorkflowStepType.EXPLORATORY_DATA_ANALYSIS:
            recommendations.extend([
                "Visualize data distributions to understand patterns",
                "Analyze feature correlations and relationships",
                "Identify potential biases in the dataset",
                "Document insights and hypotheses for modeling"
            ])
        elif current_step.step_type == WorkflowStepType.ALGORITHM_SELECTION:
            recommendations.extend([
                "Consider problem type: classification, regression, or clustering",
                "Match algorithm complexity to dataset size",
                "Start with simple baseline models",
                "Consider interpretability requirements"
            ])
        else:
            recommendations.append(f"Focus on completing {current_step.name} step")
        
        return recommendations
    
    def advance_to_next_step(self) -> bool:
        """
        Advance to the next workflow step.
        
        Returns:
            True if advanced successfully, False if at end or prerequisites not met
        """
        current_step = self.get_current_step()
        if current_step and current_step.status != WorkflowStepStatus.COMPLETED:
            logger.warning(f"Cannot advance: current step '{current_step.name}' not completed")
            return False
        
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self._save_state()
            return True
        
        return False
    
    def start_current_step(self):
        """Start the current workflow step."""
        current_step = self.get_current_step()
        if current_step:
            current_step.start()
            self._save_state()
    
    def complete_current_step(self, metadata: Optional[Dict[str, Any]] = None):
        """Complete the current workflow step."""
        current_step = self.get_current_step()
        if current_step:
            if metadata:
                current_step.metadata.update(metadata)
            current_step.complete()
            self._save_state()
    
    def fail_current_step(self, error_message: str):
        """Mark the current step as failed."""
        current_step = self.get_current_step()
        if current_step:
            current_step.fail(error_message)
            self._save_state()
    
    def _save_state(self):
        """Save the current workflow state to disk."""
        if not self.auto_save:
            return
        
        try:
            state_file = os.path.join(self.workspace_dir, f"{self.project_name}_workflow_state.json")
            
            # Prepare serializable state
            state = {
                "project_name": self.project_name,
                "current_step_index": self.current_step_index,
                "project_metadata": self.project_metadata,
                "steps": []
            }
            
            for step in self.steps:
                step_data = {
                    "name": step.name,
                    "step_type": step.step_type.value,
                    "description": step.description,
                    "status": step.status.value,
                    "progress": step.progress,
                    "metadata": step.metadata,
                    "dependencies": step.dependencies,
                    "estimated_time_minutes": step.estimated_time_minutes,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error_message": step.error_message
                }
                state["steps"].append(step_data)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}")
    
    def _load_state(self):
        """Load workflow state from disk if it exists."""
        try:
            state_file = os.path.join(self.workspace_dir, f"{self.project_name}_workflow_state.json")
            
            if not os.path.exists(state_file):
                return
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.current_step_index = state.get("current_step_index", 0)
            self.project_metadata = state.get("project_metadata", {})
            
            # Restore steps if they exist in saved state
            if "steps" in state:
                self.steps = []
                for step_data in state["steps"]:
                    step = WorkflowStep(
                        name=step_data["name"],
                        step_type=WorkflowStepType(step_data["step_type"]),
                        description=step_data["description"],
                        status=WorkflowStepStatus(step_data["status"]),
                        progress=step_data["progress"],
                        metadata=step_data["metadata"],
                        dependencies=step_data["dependencies"],
                        estimated_time_minutes=step_data["estimated_time_minutes"],
                        error_message=step_data.get("error_message")
                    )
                    
                    # Parse datetime strings back to datetime objects
                    if step_data.get("started_at"):
                        step.started_at = datetime.fromisoformat(step_data["started_at"])
                    if step_data.get("completed_at"):
                        step.completed_at = datetime.fromisoformat(step_data["completed_at"])
                    
                    self.steps.append(step)
                    
        except Exception as e:
            logger.error(f"Failed to load workflow state: {e}")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire workflow state."""
        completed, total, progress = self.get_progress()
        current_step = self.get_current_step()
        
        return {
            "project_name": self.project_name,
            "progress": {
                "completed_steps": completed,
                "total_steps": total,
                "percentage": progress
            },
            "current_step": {
                "name": current_step.name if current_step else "Completed",
                "description": current_step.description if current_step else "",
                "status": current_step.status.value if current_step else "completed"
            },
            "recommendations": self.get_recommendations(),
            "estimated_remaining_time": self._get_estimated_remaining_time()
        }
    
    def _get_estimated_remaining_time(self) -> int:
        """Get estimated remaining time in minutes."""
        remaining_time = 0
        for i in range(self.current_step_index, len(self.steps)):
            step = self.steps[i]
            if step.status != WorkflowStepStatus.COMPLETED:
                remaining_time += step.estimated_time_minutes
        return remaining_time
