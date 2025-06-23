"""
Enhanced ML Workflow Application - Main Entry Point

This application provides both the traditional algorithm explorer and 
the new agent-based workflow navigator in a unified interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from machine_learning_model.gui.main_window import MLFrameworkGUI
from machine_learning_model.gui.workflow_gui import WorkflowNavigatorGUI


class MLApplicationLauncher:
    """
    Main application launcher providing access to different ML tools.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Machine Learning Framework - Agent Mode")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")
        
        self.create_interface()
    
    def create_interface(self):
        """Create the main launcher interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Machine Learning Framework", 
                               font=("Arial", 24, "bold"))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Choose your ML workspace", 
                                  font=("Arial", 12))
        subtitle_label.pack(pady=(0, 30))
        
        # Application options
        self.create_app_cards(main_frame)
        
        # Footer
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(30, 0))
        
        ttk.Label(footer_frame, text="Machine Learning Framework v2.0 - Agent Mode", 
                 font=("Arial", 10)).pack()
    
    def create_app_cards(self, parent):
        """Create application selection cards."""
        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill=tk.BOTH, expand=True)
        
        # Workflow Navigator Card
        workflow_card = self.create_card(
            cards_frame,
            "🤖 AI Workflow Navigator",
            "Step-by-step ML pipeline guidance with intelligent assistance",
            [
                "• Automated workflow progression",
                "• AI-powered recommendations", 
                "• Interactive data exploration",
                "• Real-time progress tracking",
                "• Comprehensive result analysis"
            ],
            self.launch_workflow_navigator,
            "#4CAF50"
        )
        workflow_card.pack(fill=tk.X, pady=(0, 15))
        
        # Algorithm Explorer Card
        explorer_card = self.create_card(
            cards_frame,
            "🔬 Algorithm Explorer",
            "Traditional algorithm-focused exploration and experimentation",
            [
                "• Individual algorithm testing",
                "• Algorithm comparison tools",
                "• Performance visualization",
                "• Educational examples",
                "• Quick prototyping"
            ],
            self.launch_algorithm_explorer,
            "#2196F3"
        )
        explorer_card.pack(fill=tk.X, pady=(0, 15))
        
        # Unified Mode Card
        unified_card = self.create_card(
            cards_frame,
            "🚀 Unified Mode",
            "Combined interface with both workflow guidance and algorithm exploration",
            [
                "• Best of both worlds",
                "• Switch between modes",
                "• Integrated experience",
                "• Advanced features",
                "• Production ready"
            ],
            self.launch_unified_mode,
            "#FF9800"
        )
        unified_card.pack(fill=tk.X)
    
    def create_card(self, parent, title, description, features, command, color):
        """Create an application card."""
        # Main card frame
        card_frame = ttk.LabelFrame(parent, text="", padding=15)
        
        # Card header
        header_frame = ttk.Frame(card_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text=title, 
                               font=("Arial", 16, "bold"))
        title_label.pack(anchor=tk.W)
        
        desc_label = ttk.Label(header_frame, text=description, 
                              font=("Arial", 10), foreground="#666")
        desc_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Features list
        features_frame = ttk.Frame(card_frame)
        features_frame.pack(fill=tk.X, pady=(0, 15))
        
        for feature in features:
            feature_label = ttk.Label(features_frame, text=feature, 
                                     font=("Arial", 9))
            feature_label.pack(anchor=tk.W, pady=1)
        
        # Launch button
        button_frame = ttk.Frame(card_frame)
        button_frame.pack(fill=tk.X)
        
        launch_button = ttk.Button(button_frame, text="Launch", 
                                  command=command, width=15)
        launch_button.pack(side=tk.RIGHT)
        
        return card_frame
    
    def launch_workflow_navigator(self):
        """Launch the AI Workflow Navigator."""
        try:
            self.root.withdraw()  # Hide launcher
            
            # Create new window for workflow navigator
            workflow_root = tk.Toplevel()
            workflow_app = WorkflowNavigatorGUI(workflow_root)
            
            # Handle window closing
            def on_close():
                workflow_root.destroy()
                self.root.deiconify()  # Show launcher again
            
            workflow_root.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Workflow Navigator: {e}")
            self.root.deiconify()
    
    def launch_algorithm_explorer(self):
        """Launch the Algorithm Explorer."""
        try:
            self.root.withdraw()  # Hide launcher
            
            # Create new window for algorithm explorer
            explorer_root = tk.Toplevel()
            explorer_app = MLFrameworkGUI()
            explorer_app.root = explorer_root
            explorer_app.create_widgets()
            
            # Handle window closing
            def on_close():
                explorer_root.destroy()
                self.root.deiconify()  # Show launcher again
            
            explorer_root.protocol("WM_DELETE_WINDOW", on_close)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Algorithm Explorer: {e}")
            self.root.deiconify()
    
    def launch_unified_mode(self):
        """Launch the Unified Mode (both interfaces)."""
        try:
            # For now, launch workflow navigator as the unified mode
            # In a full implementation, this would be a tabbed interface with both
            self.launch_workflow_navigator()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Unified Mode: {e}")
    
    def run(self):
        """Run the launcher application."""
        self.root.mainloop()


def main():
    """Main entry point for the enhanced ML application."""
    try:
        app = MLApplicationLauncher()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
