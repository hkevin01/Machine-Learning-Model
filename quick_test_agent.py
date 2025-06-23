#!/usr/bin/env python3
"""
Quick test of the ML Agent Workflow System
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_agent():
    """Test the ML Agent basic functionality."""
    try:
        print("🧪 Testing ML Agent Workflow System...")
        print("=" * 50)
        
        # Test imports
        from machine_learning_model.workflow.ml_agent import MLAgent
        print("✅ MLAgent imported successfully")
        
        # Test agent creation
        agent = MLAgent('test_project', '.', auto_save=False)
        print(f"✅ Agent created: {agent.project_name}")
        print(f"✅ Agent has {len(agent.steps)} workflow steps")
        
        # Test recommendations
        recommendations = agent.get_recommendations()
        print(f"✅ Agent provides {len(recommendations)} recommendations")
        
        # Test progress tracking
        completed, total, progress = agent.get_progress()
        print(f"✅ Progress tracking: {completed}/{total} ({progress:.1f}%)")
        
        # Test workflow summary
        summary = agent.get_workflow_summary()
        print(f"✅ Workflow summary generated: {summary['project_name']}")
        
        print("\n🎉 ALL TESTS PASSED! Agent Mode is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent()
    sys.exit(0 if success else 1)
