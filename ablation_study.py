# ablation_study.py
import subprocess
import itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm

def run_experiment(params):
    """Run a single experiment with the given parameters."""
    cmd = ["python", "assignment1.py"]
    
    for key, value in params.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Extract test results from output
    accuracy, f1_score = 0.0, 0.0
    for line in stdout.split("\n"):
        if "Accuracy: " in line:
            try:
                accuracy = float(line.split("Accuracy: ")[1])
            except:
                pass
        if "F1 Score: " in line:
            try:
                f1_score = float(line.split("F1 Score: ")[1])
            except:
                pass
    
    return {
        "params": params,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "output": stdout,
        "error": stderr
    }

def grid_search():
    """Perform grid search over hyperparameters."""
    # Define parameter grid
    param_grid = {
        "pca_components": [128],
        "hidden_layers": ["128,64,32", "256,128,64", "512,256,128,64"],
        "activation": ["gelu"],
        "optimizer": ["adam"],
        "lr": [0.01, 0.005, 0.0005],
        "dropout": [0.0, 0.5, 0.7],
        "use_batchnorm": [False, True],
        "batch_size": [128]
    }
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"ablation_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Store all parameter combinations
    all_param_combinations = []
    for k, v in param_grid.items():
        all_param_combinations.append([(k, val) for val in v])
    
    # Generate all combinations
    all_experiment_params = []
    for items in itertools.product(*all_param_combinations):
        params = {k: v for k, v in items}
        all_experiment_params.append(params)
    
    print(f"Total experiments to run: {len(all_experiment_params)}")
    
    # Run experiments
    results = []
    for i, params in enumerate(tqdm(all_experiment_params)):
        print(f"\nExperiment {i+1}/{len(all_experiment_params)}")
        result = run_experiment(params)
        results.append(result)
        
        # Save incremental results
        df = pd.DataFrame([{**r["params"], "accuracy": r["accuracy"], "f1_score": r["f1_score"]} for r in results])
        df.to_csv(f"{results_dir}/results.csv", index=False)
        
        # Save detailed results
        with open(f"{results_dir}/detailed_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    return results, results_dir

def run_focused_ablation():
    """Run a more focused ablation study by varying one parameter at a time."""
    # First find a good baseline
    base_params = {
        "pca_components": 128,
        "hidden_layers": "128,64,32",
        "activation": "gelu",
        "optimizer": "adam",
        "lr": 0.001,
        "dropout": 0.5,
        "use_batchnorm": True,
        "epochs": 50,
        "batch_size": 128,
        "w_init": "xavier"
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"focused_ablation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run baseline
    print("Running baseline model...")
    baseline_result = run_experiment(base_params)
    all_results = [baseline_result]
    
    # Parameters to ablate
    ablation_params = {
        "pca_components": [128],
        "hidden_layers": ["128,64,32", "256,128,64", "512,256,128,64"],
        "activation": ["gelu"],
        "optimizer": ["adam"],
        "lr": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        "dropout": [0.0, 0.3, 0.5, 0.7],
        "use_batchnorm": [False, True],
        "batch_size": [64, 128]
    }
    
    # For each parameter, run experiments varying just that parameter
    for param, values in ablation_params.items():
        print(f"\nAblating parameter: {param}")
        for value in values:
            if param in base_params and base_params[param] == value:
                continue  # Skip baseline case, already run
                
            test_params = base_params.copy()
            test_params[param] = value
            
            print(f"Testing {param} = {value}")
            result = run_experiment(test_params)
            all_results.append(result)
            
            # Save incremental results
            df = pd.DataFrame([{**r["params"], "accuracy": r["accuracy"], "f1_score": r["f1_score"]} for r in all_results])
            df.to_csv(f"{results_dir}/results.csv", index=False)
            
            # Save detailed results
            with open(f"{results_dir}/detailed_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
    
    return all_results, results_dir

def analyze_results(results, results_dir):
    """Analyze and visualize the results."""
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{**r["params"], "accuracy": r["accuracy"], "f1_score": r["f1_score"]} for r in results])
    
    # Find best configuration
    best_acc_idx = df["accuracy"].idxmax()
    best_f1_idx = df["f1_score"].idxmax()
    
    print("\n===== BEST CONFIGURATIONS =====")
    print("\nBest configuration by accuracy:")
    print(f"Accuracy: {df.loc[best_acc_idx, 'accuracy']:.4f}")
    for param in df.columns:
        if param not in ["accuracy", "f1_score"]:
            print(f"{param}: {df.loc[best_acc_idx, param]}")
    
    print("\nBest configuration by F1 score:")
    print(f"F1 Score: {df.loc[best_f1_idx, 'f1_score']:.4f}")
    for param in df.columns:
        if param not in ["accuracy", "f1_score"]:
            print(f"{param}: {df.loc[best_f1_idx, param]}")
    
    # Save best configurations
    with open(f"{results_dir}/best_configurations.txt", "w") as f:
        f.write("===== BEST CONFIGURATIONS =====\n\n")
        
        f.write("Best configuration by accuracy:\n")
        f.write(f"Accuracy: {df.loc[best_acc_idx, 'accuracy']:.4f}\n")
        for param in df.columns:
            if param not in ["accuracy", "f1_score"]:
                f.write(f"{param}: {df.loc[best_acc_idx, param]}\n")
        
        f.write("\nBest configuration by F1 score:\n")
        f.write(f"F1 Score: {df.loc[best_f1_idx, 'f1_score']:.4f}\n")
        for param in df.columns:
            if param not in ["accuracy", "f1_score"]:
                f.write(f"{param}: {df.loc[best_f1_idx, param]}\n")
    
    # Generate plots for each parameter
    for param in df.columns:
        if param not in ["accuracy", "f1_score"]:
            try:
                # Group by parameter value and calculate mean performance
                grouped = df.groupby(param)[["accuracy", "f1_score"]].mean().reset_index()
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(grouped[param], grouped["accuracy"], 'bo-', label='Accuracy')
                plt.plot(grouped[param], grouped["f1_score"], 'ro-', label='F1 Score')
                plt.title(f'Effect of {param} on Model Performance')
                plt.xlabel(param)
                plt.ylabel('Performance')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{results_dir}/param_{param}.png", dpi=300)
                plt.close()
            except:
                print(f"Could not create plot for parameter: {param}")
    
    return df

def create_best_model_script(best_params, filename="run_best_model.py"):
    """Create a script to run the best model configuration."""
    with open(filename, "w") as f:
        f.write("import subprocess\n\n")
        f.write("# Run the best model configuration\n")
        f.write("cmd = ['python', 'assignment1.py'")
        
        for key, value in best_params.items():
            if key not in ["accuracy", "f1_score"]:
                if isinstance(value, bool) and value:
                    f.write(f", '--{key}'")
                elif not isinstance(value, bool):
                    f.write(f", '--{key}', '{value}'")
        
        f.write("]\n\n")
        f.write("print(f\"Running: {' '.join(cmd)}\")\n")
        f.write("subprocess.run(cmd)\n")
    
    print(f"Created script to run best model: {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation study on MLP model")
    parser.add_argument("--mode", choices=["grid", "focused"], default="focused", 
                      help="Search mode: grid (exhaustive) or focused (vary one parameter at a time)")
    parser.add_argument("--epochs", type=int, default=50, 
                      help="Epochs for each experiment")
    args = parser.parse_args()
    
    if args.mode == "grid":
        results, results_dir = grid_search()
    else:
        results, results_dir = run_focused_ablation()
    
    df = analyze_results(results, results_dir)
    
    # Get best configuration by accuracy and create script
    best_params = df.loc[df["accuracy"].idxmax()].to_dict()
    create_best_model_script(best_params, f"{results_dir}/run_best_accuracy_model.py")
    
    # Get best configuration by F1 and create script
    best_f1_params = df.loc[df["f1_score"].idxmax()].to_dict()
    create_best_model_script(best_f1_params, f"{results_dir}/run_best_f1_model.py")