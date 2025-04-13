import subprocess

# Run the best model configuration
cmd = ['python', 'assignment1.py', '--pca_components', '128', '--hidden_layers', '128,64,32', '--activation', 'gelu', '--optimizer', 'adam', '--lr', '0.005', '--dropout', '0.5', '--use_batchnorm', '--epochs', '50', '--batch_size', '128', '--w_init', 'xavier']

print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd)
