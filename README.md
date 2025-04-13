# CS5329 Assignment 1
The project requirement is in `Assignment1.pdf`

## Project Structure
The MLP structure file is defined in `assignment1.py`, where you can define the hyper parameters as well as the MLP structure by following the parameters instruction.

Ablation study is implemented to find the best model in `ablation.py`.

## How to run the code

In local environment,to run the best model -- Under the project folder, run 

`python focused_ablation_20250413_143121/run_best_model.py`

In google colab, follow the instructions in `run.ipynb`. Run all the cells. Due to the restricted running time and the bad performance of google colab CPU, the number is decreased from 100 to 10, reducing the accuracy from 54% to 41%.
The estimated running time is **5 min**. (On my own computer the time is only 30 seconds.)
