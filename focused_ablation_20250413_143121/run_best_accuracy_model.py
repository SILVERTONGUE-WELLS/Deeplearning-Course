import subprocess

# Run the best model configuration
cmd = ['python', 'assignment1.py', '--pca_components', '128', '--hidden_layers', '512,256,128,64', '--activation', 'gelu', '--optimizer', 'adam', '--lr', '0.001', '--dropout', '0.5', '--use_batchnorm', '--epochs', '100', '--batch_size', '128', '--w_init', 'xavier']

print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd)
