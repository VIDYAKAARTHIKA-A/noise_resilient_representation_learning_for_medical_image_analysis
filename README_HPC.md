# Noise-Resilient Representation Learning for Medical Image Analysis - HPC Run

## Overview
This codebase contains a 5-stage training pipeline designed to prove the Noise Resilience of the Proposed Method on the ISIC 2019 dataset. The master script `main.py` handles the entire ablation study automatically.

## Steps to Run on HPC

### 1. Data Setup
You need the ISIC 2019 dataset. 
Place the data exactly in this folder structure (or modify `config.json` to point to your data):
- `data/isic2019/ISIC2019 256x256 jpeg/train.csv`
- `data/isic2019/ISIC2019 256x256 jpeg/train/` (contains all the images)

### 2. Environment & Dependencies
HPCs have specific CUDA drivers, so **do not** just run `pip install torch` blindly. 
- Use your HPC's module system (e.g., `module load pytorch`) OR 
- Install the correct PyTorch version matching your HPC's CUDA via `pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX`.

Then install the rest of the standard dependencies:
```bash
pip install -r requirements.txt
```

### 3. Modifying Epochs (Important!)
By default, the models are set to run for 15 epochs. If you have the compute power and want to train for 50 or 100 epochs for better absolute performance, you can change this!
Open the **`config.json`** file and modify these two lines:
- `"epochs": 15` (controls the supervised classifier training)
- `"simclr_epochs": 15` (controls the initial self-supervised feature learning)
Change both to 50 or 100, and save the file.

### 3. HPC Speed Hack (Important!)
Since you are on a Linux HPC, you do not have the Windows Memory Crash bug. 
Open `config.json` and change:
`"num_workers": 0` 
to 
`"num_workers": 4` (or 8). 
This will allow data loading to happen in the background and speed up training massively!

### 5. Run the Code
Run the master script:
```bash
python main.py --run_all
```
The script will output progress and will generate a new `results.json` file. When the entire ablation study finishes, please send the `results.json` file back!
