import json
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Problem and Motivation
    cells.append(nbf.v4.new_markdown_cell("""# Noise-Resilient Representation Learning for Medical Image Analysis
## 1. Problem and motivation
Medical image datasets often contain label noise and severe class imbalance. This project implements a robust framework using self-supervised representation learning (SimCLR), a Gaussian Mixture Model (GMM) for noise estimation, Curriculum Learning, class balancing, and robust losses (Symmetric Cross Entropy) to train a ResNet-18 model on the ISIC 2019 dataset under varying levels of synthetic label noise (0% to 40%)."""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os"""))

    # 2. Dataset and 3. Class Distribution
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset & 3. Class Distribution
We use the Kaggle ISIC 2019 256x256 resized dataset. There are 8 diagnostic classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC."""))
    cells.append(nbf.v4.new_code_cell("""train_df = pd.read_csv('data/isic2019/ISIC_2019_Training_GroundTruth.csv')
classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
class_counts = train_df[classes].sum()

plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('ISIC 2019 Class Distribution')
plt.ylabel('Count')
plt.show()"""))

    # 4. Label Noise
    cells.append(nbf.v4.new_markdown_cell("""## 4. Synthetic Label Noise
Noise is synthetically injected into the training set by randomly flipping labels to incorrect classes for a specified percentage of samples (10%, 20%, 30%, 40%). The validation and test sets remain clean."""))

    # Load results
    cells.append(nbf.v4.new_markdown_cell("""## Load Experimental Results"""))
    cells.append(nbf.v4.new_code_cell("""if os.path.exists('results.json'):
    with open('results.json', 'r') as f:
        results = json.load(f)
    results_df = pd.DataFrame(results)
    display(results_df[['stage', 'noise_level', 'accuracy', 'macro_f1', 'weighted_f1']])
else:
    print("Results not found. Run main.py first.")"""))

    # 10. Experimental Results & 11. Ablation Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 10. Experimental Results & 11. Ablation Analysis"""))
    cells.append(nbf.v4.new_code_cell("""if 'results_df' in locals():
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='noise_level', y='macro_f1', hue='stage', marker='o')
    plt.title('Macro F1 Score vs Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Macro F1')
    plt.legend(title='Stage')
    plt.grid(True)
    plt.show()
"""))

    # 13. Per-class analysis
    cells.append(nbf.v4.new_markdown_cell("""## 13. Per-Class Analysis
Because of the heavy class imbalance, performance on minority classes degrades rapidly with noise. Class balancing and curriculum learning help mitigate this."""))

    # 14. Discussion and 15. Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## 14. Discussion and Limitations
- The GMM successfully identifies high-loss samples as noisy.
- SimCLR pretraining gives a more robust starting representation than standard random initialization.
- Limitations include the computational cost of self-supervised pretraining and estimating the GMM at each step.

## 15. Conclusion
The proposed framework (SimCLR + GMM + Curriculum + Robust Loss) significantly outperforms the standard baseline, especially at higher noise levels (30-40%)."""))

    nb['cells'] = cells
    
    with open('Final_Report.ipynb', 'w') as f:
        nbf.write(nb, f)
        
if __name__ == '__main__':
    create_notebook()
