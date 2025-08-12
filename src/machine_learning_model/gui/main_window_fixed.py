"""Clean reconstructed main window (temporary fix until original file cleaned)."""
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, scrolledtext
try:  # pragma: no cover
    from .icon_utils import icon_for_status  # type: ignore
except Exception:  # pragma: no cover
    def icon_for_status(_):  # type: ignore
        return "[?]"
import numpy as np
from machine_learning_model.supervised.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from machine_learning_model.supervised.random_forest import RandomForestClassifier, RandomForestRegressor

class MainWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Machine Learning Framework Explorer (Fixed)")
        self.root.geometry("1200x800")
        self.current_algorithm: str | None = None
        self.task_var = tk.StringVar(value="classification")
        self._build()

    def _build(self) -> None:
        main = tk.Frame(self.root)
        main.pack(fill='both', expand=True)
        left = tk.Frame(main, width=320)
        left.pack(side='left', fill='y')
        right = tk.Frame(main)
        right.pack(side='right', fill='both', expand=True)

        self.listbox = tk.Listbox(left)
        self.listbox.pack(fill='both', expand=True, padx=6, pady=6)
        for name in ["Decision Trees", "Random Forest", "Linear Regression"]:
            self.listbox.insert(tk.END, name)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        self.details = scrolledtext.ScrolledText(right, wrap=tk.WORD, height=25)
        self.details.pack(fill='both', expand=True, padx=6, pady=6)

        run_frame = tk.Frame(right)
        run_frame.pack(fill='x', padx=6, pady=(0, 6))
        tk.Radiobutton(run_frame, text='Classification', variable=self.task_var, value='classification').pack(side='left')
        tk.Radiobutton(run_frame, text='Regression', variable=self.task_var, value='regression').pack(side='left')
        tk.Button(run_frame, text='Run', command=self.run_selected_algorithm).pack(side='left', padx=8)
        self.run_output = tk.Label(run_frame, text='Select algorithm → Run')
        self.run_output.pack(side='left')

        self._welcome()

    def _welcome(self):
        self.details.delete(1.0, tk.END)
        self.details.insert(1.0, "Welcome. Select an algorithm then optionally quick-run it on synthetic data.")

    def _on_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.current_algorithm = self.listbox.get(sel[0])
        self.details.delete(1.0, tk.END)
        self.details.insert(1.0, f"Selected: {self.current_algorithm}\nQuick synthetic run supported.")

    def run_selected_algorithm(self):
        if not self.current_algorithm:
            messagebox.showinfo("Run", "Select an algorithm first")
            return
        task = self.task_var.get()
        n = 80
        X = np.random.randn(n, 5)
        if task == 'classification':
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
        else:
            y = X[:, 0] * 0.5 - X[:, 1] * 0.3 + np.random.randn(n) * 0.1
        alg = self.current_algorithm
        try:
            if alg == 'Decision Trees':
                model = DecisionTreeClassifier() if task == 'classification' else DecisionTreeRegressor()
            elif alg == 'Random Forest':
                model = RandomForestClassifier(n_estimators=25) if task == 'classification' else RandomForestRegressor(n_estimators=25)
            elif alg == 'Linear Regression' and task == 'regression':
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                pred = X @ coef
                mse = float(np.mean((pred - y)**2))
                self.run_output.config(text=f"Linear Regression MSE={mse:.4f}")
                return
            else:
                self.run_output.config(text=f"Not implemented for {alg}/{task}")
                return
            model.fit(X, y)
            pred = model.predict(X)
            if task == 'classification':
                acc = float(np.mean(pred == y))
                self.run_output.config(text=f"{alg} Acc={acc:.3f}")
            else:
                mse = float(np.mean((pred - y)**2))
                self.run_output.config(text=f"{alg} MSE={mse:.4f}")
        except Exception as e:  # pragma: no cover
            self.run_output.config(text=f"Err: {e}")

    def run(self):
        self.root.mainloop()


def main():  # pragma: no cover
    app = MainWindow()
    app.run()

    
if __name__ == '__main__':  # pragma: no cover
    main()
