"""
Complete Workflow Example - End-to-End ML Pipeline

This example demonstrates the complete ML workflow using the agent-based system,
from data collection through deployment and monitoring.
"""

import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from machine_learning_model.workflow import MLAgent, MLWorkflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_workflow_example():
    """Run a complete workflow example from start to finish."""
    print("=" * 80)
    print("COMPLETE ML WORKFLOW EXAMPLE - AGENT MODE")
    print("=" * 80)
    
    # Create workspace directory
    workspace_dir = "example_workflow_output"
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Initialize the ML Agent
    print("\n1. Initializing ML Agent...")
    agent = MLAgent(
        project_name="complete_workflow_example",
        workspace_dir=workspace_dir,
        auto_save=True
    )
    
    # Create the workflow
    print("2. Creating ML Workflow...")
    workflow = MLWorkflow(agent)
    
    # Print initial workflow status
    print("\n3. Initial Workflow Status:")
    summary = agent.get_workflow_summary()
    print(f"   Project: {summary['project_name']}")
    print(f"   Progress: {summary['progress']['completed_steps']}/{summary['progress']['total_steps']} steps")
    print(f"   Current Step: {summary['current_step']['name']}")
    print(f"   Estimated Time: {summary['estimated_remaining_time']} minutes")
    
    # Show initial recommendations
    print("\n4. AI Recommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Execute the complete workflow
    print("\n5. Executing Complete Workflow:")
    print("-" * 50)
    
    # Workflow parameters
    workflow_params = {
        'dataset': 'iris',  # Use Iris dataset for demo
        'missing_strategy': 'auto',
        'remove_outliers': False,
        'scale_features': True,
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42
    }
    
    success = workflow.run_complete_workflow(**workflow_params)
    
    if success:
        print("\n6. Workflow Completed Successfully! ✅")
        
        # Generate final report
        print("\n7. Generating Final Report...")
        report = workflow.get_workflow_report()
        
        print("\n" + "=" * 80)
        print("FINAL WORKFLOW REPORT")
        print("=" * 80)
        
        print(f"\nProject: {report['project_name']}")
        print(f"Progress: {report['workflow_progress']['progress_percentage']:.1f}% complete")
        
        if 'data_summary' in report and report['data_summary']:
            print(f"\nData Summary:")
            print(f"  Shape: {report['data_summary']['shape']}")
            print(f"  Features: {len(report['data_summary']['features'])}")
            print(f"  Target: {report['data_summary']['target_column']}")
        
        if 'model_performance' in report and report['model_performance']:
            print(f"\nModel Performance:")
            for model_name, metrics in report['model_performance'].items():
                if 'accuracy' in metrics:
                    print(f"  {model_name}: {metrics['accuracy']:.4f} accuracy")
                elif 'r2_score' in metrics:
                    print(f"  {model_name}: {metrics['r2_score']:.4f} R²")
        
        print(f"\nStep Details:")
        for step in report['step_details']:
            status_icon = "✅" if step['status'] == 'completed' else "❌" if step['status'] == 'failed' else "⏳"
            print(f"  {status_icon} {step['name']}: {step['status']}")
        
        # Save detailed report
        report_file = os.path.join(workspace_dir, "complete_workflow_report.json")
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Workspace directory: {os.path.abspath(workspace_dir)}")
        
        # Show generated files
        print(f"\nGenerated Files:")
        for root, dirs, files in os.walk(workspace_dir):
            level = root.replace(workspace_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        return True
        
    else:
        print("\n6. Workflow Failed ❌")
        return False


def run_step_by_step_example():
    """Run a step-by-step workflow example with user interaction."""
    print("=" * 80)
    print("STEP-BY-STEP ML WORKFLOW EXAMPLE")
    print("=" * 80)
    
    workspace_dir = "step_by_step_output"
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Initialize agent
    agent = MLAgent(
        project_name="step_by_step_example",
        workspace_dir=workspace_dir,
        auto_save=True
    )
    workflow = MLWorkflow(agent)
    
    print("\nThis example will guide you through each step of the ML workflow.")
    print("Press Enter to continue to each step...\n")
    
    step_count = 0
    while agent.get_current_step() is not None:
        step_count += 1
        current_step = agent.get_current_step()
        
        print(f"\n{'='*60}")
        print(f"STEP {step_count}: {current_step.name.upper()}")
        print(f"{'='*60}")
        print(f"Description: {current_step.description}")
        print(f"Estimated time: {current_step.estimated_time_minutes} minutes")
        
        # Show recommendations
        recommendations = agent.get_recommendations()
        if recommendations:
            print(f"\nAI Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Wait for user input
        input(f"\nPress Enter to execute '{current_step.name}'...")
        
        # Execute step
        print(f"Executing {current_step.name}...")
        success = workflow.execute_current_step(dataset='iris')
        
        if success:
            print(f"✅ {current_step.name} completed successfully!")
            
            # Show step results if available
            if current_step.metadata:
                print(f"Step Results:")
                for key, value in current_step.metadata.items():
                    if isinstance(value, dict):
                        print(f"  {key}: {len(value)} items")
                    elif isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    else:
                        print(f"  {key}: {value}")
            
            agent.advance_to_next_step()
        else:
            print(f"❌ {current_step.name} failed!")
            break
        
        # Show progress
        completed, total, progress = agent.get_progress()
        print(f"\nProgress: {completed}/{total} steps completed ({progress:.1f}%)")
    
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETE!")
    print(f"{'='*60}")
    
    # Final report
    report = workflow.get_workflow_report()
    print(f"\nFinal Progress: {report['workflow_progress']['progress_percentage']:.1f}%")
    print(f"Workspace: {os.path.abspath(workspace_dir)}")


def demonstrate_agent_features():
    """Demonstrate specific agent features and capabilities."""
    print("=" * 80)
    print("ML AGENT FEATURES DEMONSTRATION")
    print("=" * 80)
    
    workspace_dir = "agent_demo_output"
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Initialize agent
    agent = MLAgent(
        project_name="agent_features_demo",
        workspace_dir=workspace_dir,
        auto_save=True
    )
    
    print("\n1. WORKFLOW STATE MANAGEMENT")
    print("-" * 40)
    print(f"Project: {agent.project_name}")
    print(f"Workspace: {agent.workspace_dir}")
    print(f"Auto-save: {agent.auto_save}")
    print(f"Current step: {agent.current_step_index + 1}/{len(agent.steps)}")
    
    print("\n2. STEP INFORMATION")
    print("-" * 40)
    for i, step in enumerate(agent.steps):
        icon = "📍" if i == agent.current_step_index else "⭕"
        print(f"{icon} {i+1}. {step.name} ({step.estimated_time_minutes}min)")
        print(f"   {step.description}")
    
    print("\n3. AI RECOMMENDATIONS")
    print("-" * 40)
    recommendations = agent.get_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n4. WORKFLOW SUMMARY")
    print("-" * 40)
    summary = agent.get_workflow_summary()
    import json
    print(json.dumps(summary, indent=2, default=str))
    
    print("\n5. STATE PERSISTENCE")
    print("-" * 40)
    print("The agent automatically saves its state to disk.")
    state_file = os.path.join(workspace_dir, f"{agent.project_name}_workflow_state.json")
    print(f"State file: {state_file}")
    
    if os.path.exists(state_file):
        print("✅ State file exists")
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        print(f"   Saved steps: {len(state_data.get('steps', []))}")
        print(f"   Current step index: {state_data.get('current_step_index', 0)}")
    else:
        print("❌ State file not found")


if __name__ == "__main__":
    print("ML Workflow Examples")
    print("=" * 50)
    print("1. Complete Workflow (automated)")
    print("2. Step-by-Step Workflow (interactive)")
    print("3. Agent Features Demo")
    print("4. All Examples")
    
    choice = input("\nSelect example (1-4, or Enter for all): ").strip()
    
    if choice == "1":
        run_complete_workflow_example()
    elif choice == "2":
        run_step_by_step_example()
    elif choice == "3":
        demonstrate_agent_features()
    else:
        # Run all examples
        print("\n" + "🚀" * 20)
        print("RUNNING ALL EXAMPLES")
        print("🚀" * 20)
        
        print("\n>>> Running Complete Workflow Example...")
        run_complete_workflow_example()
        
        print("\n>>> Running Agent Features Demo...")
        demonstrate_agent_features()
        
        print("\n>>> Step-by-Step Example available separately")
        print("    (Run with choice '2' for interactive mode)")
        
        print("\n✅ All automated examples completed!")
