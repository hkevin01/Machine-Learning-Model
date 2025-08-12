"""
Workflow GUI - Interactive Machine Learning Workflow Interface

This module provides a comprehensive GUI for the ML Agent workflow system,
allowing users to navigate through the ML pipeline step-by-step with visual guidance.
"""

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..workflow import MLAgent, MLWorkflow
from ..workflow.ml_agent import WorkflowStepStatus
from .icon_utils import icon_for_status


class WorkflowNavigatorGUI:
    """
    Comprehensive ML Workflow Navigator GUI.
    
    Provides an interactive interface for:
    - Step-by-step workflow guidance
    - Progress tracking and visualization
    - AI recommendations and assistance
    - Real-time workflow execution
    - Results visualization and reporting
    """
    
    def __init__(self, root: tk.Tk):
        """Initialize the Workflow Navigator GUI."""
        self.root = root
        self.root.title("ML Workflow Navigator - Agent Mode")
        self.root.geometry("1400x900")
        
        # Workflow components
        self.agent: Optional[MLAgent] = None
        self.workflow: Optional[MLWorkflow] = None
        self.project_name = "ml_project"
        self.workspace_dir = "."
        
        # GUI state
        self.current_step_frame = None
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready to start workflow")
        
        # Create the main interface
        self.create_main_interface()
        
        # Initialize with default project
        self.initialize_default_project()
    
    def create_main_interface(self):
        """Create the main GUI interface."""
        # Create main container with sidebar and content area
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left sidebar for navigation and project info
        self.create_sidebar(main_container)
        
        # Main content area
        self.create_content_area(main_container)
        
        # Bottom status bar
        self.create_status_bar()
    
    def create_sidebar(self, parent):
        """Create the sidebar with project info and step navigation."""
        sidebar_frame = ttk.Frame(parent, width=350)
        sidebar_frame.pack_propagate(False)
        parent.add(sidebar_frame)
        
        # Project section
        project_section = ttk.LabelFrame(sidebar_frame, text="Project Information", padding=10)
        project_section.pack(fill=tk.X, padx=5, pady=5)
        
        # Project name
        ttk.Label(project_section, text="Project Name:").pack(anchor=tk.W)
        self.project_name_var = tk.StringVar(value=self.project_name)
        ttk.Entry(project_section, textvariable=self.project_name_var, width=30).pack(fill=tk.X, pady=(0, 5))
        
        # Workspace directory
        ttk.Label(project_section, text="Workspace Directory:").pack(anchor=tk.W)
        workspace_frame = ttk.Frame(project_section)
        workspace_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.workspace_var = tk.StringVar(value=self.workspace_dir)
        ttk.Entry(workspace_frame, textvariable=self.workspace_var, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(workspace_frame, text="Browse", command=self.browse_workspace, width=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Project controls
        controls_frame = ttk.Frame(project_section)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="New Project", command=self.new_project).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Load Project", command=self.load_project).pack(side=tk.LEFT)
        
        # Progress section
        progress_section = ttk.LabelFrame(sidebar_frame, text="Workflow Progress", padding=10)
        progress_section.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_section, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Progress text
        self.progress_label = ttk.Label(progress_section, text="0/0 steps completed (0%)")
        self.progress_label.pack(anchor=tk.W)
        
        # Current step
        self.current_step_label = ttk.Label(progress_section, text="Current: Not started", 
                                          font=("Arial", 10, "bold"))
        self.current_step_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Step navigation
        nav_section = ttk.LabelFrame(sidebar_frame, text="Step Navigation", padding=10)
        nav_section.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Steps list
        self.steps_tree = ttk.Treeview(nav_section, columns=("status", "progress"), show="tree headings", height=10)
        self.steps_tree.heading("#0", text="Step")
        self.steps_tree.heading("status", text="Status")
        self.steps_tree.heading("progress", text="Progress")
        self.steps_tree.column("#0", width=200)
        self.steps_tree.column("status", width=80)
        self.steps_tree.column("progress", width=60)
        
        # Scrollbar for steps
        steps_scrollbar = ttk.Scrollbar(nav_section, orient=tk.VERTICAL, command=self.steps_tree.yview)
        self.steps_tree.configure(yscrollcommand=steps_scrollbar.set)
        
        steps_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.steps_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind step selection
        self.steps_tree.bind("<<TreeviewSelect>>", self.on_step_select)
        
        # Navigation buttons
        nav_buttons = ttk.Frame(nav_section)
        nav_buttons.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_buttons, text="Previous", command=self.previous_step).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_buttons, text="Next", command=self.next_step).pack(side=tk.LEFT)
        ttk.Button(nav_buttons, text="Execute", command=self.execute_current_step).pack(side=tk.RIGHT)
    
    def create_content_area(self, parent):
        """Create the main content area."""
        content_frame = ttk.Frame(parent)
        parent.add(content_frame)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Step Details tab
        self.create_step_details_tab()
        
        # AI Assistant tab
        self.create_ai_assistant_tab()
        
        # Data Viewer tab
        self.create_data_viewer_tab()
        
        # Results tab
        self.create_results_tab()
        
        # Visualization tab
        self.create_visualization_tab()
    
    def create_step_details_tab(self):
        """Create the step details tab."""
        step_frame = ttk.Frame(self.notebook)
        self.notebook.add(step_frame, text="Step Details")
        
        # Step title and description
        self.step_title = ttk.Label(step_frame, text="Welcome to ML Workflow Navigator", 
                                   font=("Arial", 16, "bold"))
        self.step_title.pack(pady=(10, 5))
        
        self.step_description = ttk.Label(step_frame, text="Select a project to begin the ML workflow", 
                                        font=("Arial", 11), wraplength=800)
        self.step_description.pack(pady=(0, 10))
        
        # Step content area (will be populated dynamically)
        self.step_content_frame = ttk.Frame(step_frame)
        self.step_content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Step controls
        controls_frame = ttk.Frame(step_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.execute_button = ttk.Button(controls_frame, text="Execute Step", 
                                       command=self.execute_current_step, state=tk.DISABLED)
        self.execute_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.skip_button = ttk.Button(controls_frame, text="Skip Step", 
                                    command=self.skip_current_step, state=tk.DISABLED)
        self.skip_button.pack(side=tk.RIGHT)
    
    def create_ai_assistant_tab(self):
        """Create the AI assistant tab."""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="AI Assistant")
        
        # Assistant title
        ttk.Label(ai_frame, text="AI Workflow Assistant", 
                 font=("Arial", 14, "bold")).pack(pady=(10, 5))
        
        # Recommendations section
        rec_section = ttk.LabelFrame(ai_frame, text="Recommendations", padding=10)
        rec_section.pack(fill=tk.X, padx=10, pady=5)
        
        self.recommendations_text = scrolledtext.ScrolledText(rec_section, height=8, wrap=tk.WORD)
        self.recommendations_text.pack(fill=tk.X)
        
        # Workflow summary section
        summary_section = ttk.LabelFrame(ai_frame, text="Workflow Summary", padding=10)
        summary_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.summary_text = scrolledtext.ScrolledText(summary_section, height=10, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Assistant controls
        controls_frame = ttk.Frame(ai_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(controls_frame, text="Refresh Recommendations", 
                  command=self.refresh_recommendations).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Export Report", 
                  command=self.export_workflow_report).pack(side=tk.RIGHT)
    
    def create_data_viewer_tab(self):
        """Create the data viewer tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Viewer")
        
        # Data info section
        info_section = ttk.LabelFrame(data_frame, text="Dataset Information", padding=10)
        info_section.pack(fill=tk.X, padx=10, pady=5)
        
        self.data_info_text = scrolledtext.ScrolledText(info_section, height=5, wrap=tk.WORD)
        self.data_info_text.pack(fill=tk.X)
        
        # Data preview section
        preview_section = ttk.LabelFrame(data_frame, text="Data Preview", padding=10)
        preview_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create Treeview for data display
        self.data_tree = ttk.Treeview(preview_section)
        data_scrollbar_y = ttk.Scrollbar(preview_section, orient=tk.VERTICAL, command=self.data_tree.yview)
        data_scrollbar_x = ttk.Scrollbar(preview_section, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=data_scrollbar_y.set, xscrollcommand=data_scrollbar_x.set)
        
        data_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        data_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Data controls
        data_controls = ttk.Frame(data_frame)
        data_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(data_controls, text="Load Data", command=self.load_data_file).pack(side=tk.LEFT)
        ttk.Button(data_controls, text="Refresh View", command=self.refresh_data_view).pack(side=tk.LEFT, padx=(5, 0))
    
    def create_results_tab(self):
        """Create the results tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_visualization_tab(self):
        """Create the visualization tab."""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualizations")
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Visualization controls
        viz_controls = ttk.Frame(viz_frame)
        viz_controls.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(viz_controls, text="Clear Plot", command=self.clear_plot).pack(side=tk.LEFT)
        ttk.Button(viz_controls, text="Save Plot", command=self.save_plot).pack(side=tk.LEFT, padx=(5, 0))
    
    def create_status_bar(self):
        """Create the bottom status bar."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Separator(status_frame, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        status_content = ttk.Frame(status_frame)
        status_content.pack(fill=tk.X, padx=5, pady=2)
        
        # Status label
        status_label = ttk.Label(status_content, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Time estimate
        self.time_estimate_var = tk.StringVar(value="")
        time_label = ttk.Label(status_content, textvariable=self.time_estimate_var)
        time_label.pack(side=tk.RIGHT)
    
    def initialize_default_project(self):
        """Initialize a default project."""
        self.agent = MLAgent(
            project_name=self.project_name,
            workspace_dir=self.workspace_dir,
            auto_save=True
        )
        self.workflow = MLWorkflow(self.agent)
        self.update_ui()
    
    def new_project(self):
        """Create a new project."""
        project_name = self.project_name_var.get().strip()
        workspace_dir = self.workspace_var.get().strip()
        
        if not project_name:
            messagebox.showerror("Error", "Please enter a project name.")
            return
        
        if not os.path.exists(workspace_dir):
            try:
                os.makedirs(workspace_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create workspace directory: {e}")
                return
        
        self.project_name = project_name
        self.workspace_dir = workspace_dir
        
        self.agent = MLAgent(
            project_name=project_name,
            workspace_dir=workspace_dir,
            auto_save=True
        )
        self.workflow = MLWorkflow(self.agent)
        
        self.update_ui()
        self.status_var.set(f"Created new project: {project_name}")
    
    def load_project(self):
        """Load an existing project."""
        workspace_dir = filedialog.askdirectory(title="Select Project Workspace Directory")
        if not workspace_dir:
            return
        
        # Look for existing workflow state files
        state_files = [f for f in os.listdir(workspace_dir) if f.endswith('_workflow_state.json')]
        
        if not state_files:
            messagebox.showwarning("Warning", "No workflow state files found in selected directory.")
            return
        
        # Use the first state file found
        state_file = state_files[0]
        project_name = state_file.replace('_workflow_state.json', '')
        
        self.project_name = project_name
        self.workspace_dir = workspace_dir
        self.project_name_var.set(project_name)
        self.workspace_var.set(workspace_dir)
        
        self.agent = MLAgent(
            project_name=project_name,
            workspace_dir=workspace_dir,
            auto_save=True
        )
        self.workflow = MLWorkflow(self.agent)
        
        self.update_ui()
        self.status_var.set(f"Loaded project: {project_name}")
    
    def browse_workspace(self):
        """Browse for workspace directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.workspace_var.set(directory)
    
    def update_ui(self):
        """Update the entire UI with current workflow state."""
        if not self.agent:
            return
        
        self.update_progress()
        self.update_steps_tree()
        self.update_step_details()
        self.update_ai_assistant()
        self.refresh_data_view()
        self.update_time_estimate()
    
    def update_progress(self):
        """Update the progress bar and labels."""
        if not self.agent:
            return
        
        completed, total, progress = self.agent.get_progress()
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{completed}/{total} steps completed ({progress:.1f}%)")
        
        current_step = self.agent.get_current_step()
        if current_step:
            self.current_step_label.config(text=f"Current: {current_step.name}")
        else:
            self.current_step_label.config(text="Workflow Complete!")
    
    def update_steps_tree(self):
        """Update the steps tree view."""
        if not self.agent:
            return
        
        # Clear existing items
        for item in self.steps_tree.get_children():
            self.steps_tree.delete(item)
        
        # Add steps
        for i, step in enumerate(self.agent.steps):
            status_text = step.status.value.replace('_', ' ').title()
            progress_text = f"{step.progress*100:.0f}%"
            
            # Determine icon using helper for fallback safety
            icon = icon_for_status(step.status.name)
            
            item_text = f"{icon} {step.name}"
            item_id = self.steps_tree.insert("", "end", text=item_text, 
                                           values=(status_text, progress_text))
            
            # Highlight current step
            if i == self.agent.current_step_index:
                self.steps_tree.selection_set(item_id)
    
    def update_step_details(self):
        """Update the step details tab."""
        current_step = self.agent.get_current_step() if self.agent else None
        
        if current_step:
            self.step_title.config(text=current_step.name)
            self.step_description.config(text=current_step.description)
            self.execute_button.config(state=tk.NORMAL)
            self.skip_button.config(state=tk.NORMAL)
        else:
            self.step_title.config(text="Workflow Complete")
            self.step_description.config(text="All workflow steps have been completed successfully!")
            self.execute_button.config(state=tk.DISABLED)
            self.skip_button.config(state=tk.DISABLED)
        
        # Clear and update step content
        for widget in self.step_content_frame.winfo_children():
            widget.destroy()
        
        if current_step:
            self.create_step_content(current_step)
    
    def create_step_content(self, step):
        """Create step-specific content."""
        content_frame = ttk.Frame(self.step_content_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Step metadata
        if step.metadata:
            metadata_section = ttk.LabelFrame(content_frame, text="Step Information", padding=10)
            metadata_section.pack(fill=tk.X, pady=(0, 10))
            
            metadata_text = scrolledtext.ScrolledText(metadata_section, height=6, wrap=tk.WORD)
            metadata_text.pack(fill=tk.X)
            
            # Format metadata nicely
            metadata_str = json.dumps(step.metadata, indent=2, default=str)
            metadata_text.insert(tk.END, metadata_str)
            metadata_text.config(state=tk.DISABLED)
        
        # Step-specific parameters
        params_section = ttk.LabelFrame(content_frame, text="Parameters", padding=10)
        params_section.pack(fill=tk.X, pady=(0, 10))
        
        # Add step-specific parameter controls
        self.create_step_parameters(params_section, step)
    
    def create_step_parameters(self, parent, step):
        """Create step-specific parameter controls."""
        # This would be expanded to include specific parameters for each step
        if step.step_type.value == "data_collection":
            # Dataset selection
            ttk.Label(parent, text="Dataset:").pack(anchor=tk.W)
            dataset_var = tk.StringVar(value="iris")
            dataset_combo = ttk.Combobox(parent, textvariable=dataset_var, 
                                       values=["iris", "wine", "diabetes", "breast_cancer"])
            dataset_combo.pack(fill=tk.X, pady=(0, 5))
            
            # File upload option
            ttk.Label(parent, text="Or upload custom data:").pack(anchor=tk.W, pady=(10, 0))
            file_frame = ttk.Frame(parent)
            file_frame.pack(fill=tk.X, pady=(0, 5))
            
            file_var = tk.StringVar()
            ttk.Entry(file_frame, textvariable=file_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(file_frame, text="Browse", 
                      command=lambda: self.browse_data_file(file_var)).pack(side=tk.RIGHT, padx=(5, 0))
        
        elif step.step_type.value == "data_preprocessing":
            # Preprocessing options
            ttk.Checkbutton(parent, text="Remove outliers").pack(anchor=tk.W)
            ttk.Checkbutton(parent, text="Scale features").pack(anchor=tk.W)
            
            ttk.Label(parent, text="Missing value strategy:").pack(anchor=tk.W, pady=(10, 0))
            strategy_var = tk.StringVar(value="auto")
            ttk.Combobox(parent, textvariable=strategy_var, 
                        values=["auto", "mean", "median", "mode", "drop"]).pack(fill=tk.X)
        
        else:
            # Default parameter display
            ttk.Label(parent, text="No specific parameters for this step.").pack(anchor=tk.W)
    
    def browse_data_file(self, file_var):
        """Browse for data file."""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            file_var.set(filename)
    
    def update_ai_assistant(self):
        """Update the AI assistant tab."""
        if not self.agent:
            return
        
        # Update recommendations
        recommendations = self.agent.get_recommendations()
        self.recommendations_text.delete(1.0, tk.END)
        
        for i, rec in enumerate(recommendations, 1):
            self.recommendations_text.insert(tk.END, f"{i}. {rec}\n\n")
        
        # Update workflow summary
        summary = self.agent.get_workflow_summary()
        self.summary_text.delete(1.0, tk.END)
        
        summary_str = json.dumps(summary, indent=2, default=str)
        self.summary_text.insert(tk.END, summary_str)
    
    def refresh_data_view(self):
        """Refresh the data viewer tab."""
        if not self.workflow or not hasattr(self.workflow, 'data') or self.workflow.data is None:
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, "No data loaded yet.")
            
            # Clear data tree
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            return
        
        data = self.workflow.data
        
        # Update data info
        info = f"Shape: {data.shape}\n"
        info += f"Columns: {list(data.columns)}\n"
        info += f"Data types:\n{data.dtypes.to_string()}\n"
        info += f"Missing values:\n{data.isnull().sum().to_string()}"
        
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(tk.END, info)
        
        # Update data tree
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Set up columns
        columns = list(data.columns)
        self.data_tree["columns"] = columns
        self.data_tree["show"] = "headings"
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Add data rows (limit to first 100 for performance)
        for i, row in data.head(100).iterrows():
            self.data_tree.insert("", "end", values=list(row))
    
    def update_time_estimate(self):
        """Update the time estimate in status bar."""
        if not self.agent:
            return
        
        remaining_time = self.agent._get_estimated_remaining_time()
        if remaining_time > 0:
            if remaining_time >= 60:
                hours = remaining_time // 60
                minutes = remaining_time % 60
                self.time_estimate_var.set(f"Est. remaining: {hours}h {minutes}m")
            else:
                self.time_estimate_var.set(f"Est. remaining: {remaining_time}m")
        else:
            self.time_estimate_var.set("")
    
    def on_step_select(self, event):
        """Handle step selection in the tree."""
        selection = self.steps_tree.selection()
        if selection and self.agent:
            # Get the index of selected step
            selected_item = selection[0]
            all_items = self.steps_tree.get_children()
            step_index = all_items.index(selected_item)
            
            # Update step details without changing current step
            step = self.agent.steps[step_index]
            self.step_title.config(text=step.name)
            self.step_description.config(text=step.description)
    
    def execute_current_step(self):
        """Execute the current workflow step."""
        if not self.workflow:
            messagebox.showerror("Error", "No workflow initialized.")
            return
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Executing Step")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Executing workflow step...").pack(pady=20)
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        progress_bar.start()
        
        def execute_step():
            """Execute step in background thread."""
            try:
                success = self.workflow.execute_current_step()
                self.root.after(0, lambda: self.step_execution_complete(progress_window, success))
            except Exception as e:
                self.root.after(0, lambda: self.step_execution_error(progress_window, str(e)))
        
        # Start execution in background thread
        thread = threading.Thread(target=execute_step, daemon=True)
        thread.start()
    
    def step_execution_complete(self, progress_window, success):
        """Handle step execution completion."""
        progress_window.destroy()
        
        if success:
            self.status_var.set("Step executed successfully!")
            self.agent.advance_to_next_step()
            self.update_ui()
            messagebox.showinfo("Success", "Step executed successfully!")
        else:
            self.status_var.set("Step execution failed.")
            messagebox.showerror("Error", "Step execution failed. Check the logs for details.")
    
    def step_execution_error(self, progress_window, error_msg):
        """Handle step execution error."""
        progress_window.destroy()
        self.status_var.set(f"Step execution error: {error_msg}")
        messagebox.showerror("Error", f"Step execution failed:\n{error_msg}")
    
    def skip_current_step(self):
        """Skip the current workflow step."""
        if not self.agent:
            return
        
        current_step = self.agent.get_current_step()
        if current_step:
            reason = f"Skipped by user"
            current_step.skip(reason)
            self.agent.advance_to_next_step()
            self.update_ui()
            self.status_var.set(f"Skipped step: {current_step.name}")
    
    def previous_step(self):
        """Go to previous step."""
        if not self.agent or self.agent.current_step_index <= 0:
            return
        
        self.agent.current_step_index -= 1
        self.agent._save_state()
        self.update_ui()
    
    def next_step(self):
        """Go to next step."""
        if not self.agent:
            return
        
        self.agent.advance_to_next_step()
        self.update_ui()
    
    def refresh_recommendations(self):
        """Refresh AI recommendations."""
        self.update_ai_assistant()
        self.status_var.set("Recommendations refreshed.")
    
    def export_workflow_report(self):
        """Export comprehensive workflow report."""
        if not self.workflow:
            messagebox.showerror("Error", "No workflow to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Workflow Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if filename:
            try:
                report = self.workflow.get_workflow_report()
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                self.status_var.set(f"Report exported to {filename}")
                messagebox.showinfo("Success", f"Workflow report exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {e}")
    
    def load_data_file(self):
        """Load data from file."""
        filename = filedialog.askopenfilename(
            title="Load Data File",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")]
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    data = pd.read_csv(filename)
                elif filename.endswith('.json'):
                    data = pd.read_json(filename)
                else:
                    messagebox.showerror("Error", "Unsupported file format.")
                    return
                
                if self.workflow:
                    self.workflow.data = data
                    self.refresh_data_view()
                    self.status_var.set(f"Loaded data from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def clear_plot(self):
        """Clear the visualization plot."""
        self.ax.clear()
        self.canvas.draw()
    
    def save_plot(self):
        """Save the current plot."""
        filename = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Plot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the Workflow Navigator GUI."""
    root = tk.Tk()
    app = WorkflowNavigatorGUI(root)
    app.run()


if __name__ == "__main__":
    main()
