# BAE-UQ-IDS
Here is a project of Cybersecurity anomaly detection using BAE and BVAE with uncertainty quantification on dataset of UNSW-15 and CIC-IDS-2017.
The uncertianty including aleatoric uncertianty through modelloing aleatoric noise and epistemic uncertainty.

This is the source code for the paper of "Towards Trustworthy Cybersecurity Operations using Bayesian Deep Learning to Improve Uncertainty Quantification of Anomaly Detection"

The usage process:
1. Download original dataset.
  Please download the original CIC-IDS-2017 dataset from the website: https://www.unb.ca/cic/datasets/ids-2017.html
  and the data used in this research is in " CIC-IDS-2017\MachineLearningCSV (1)\MachineLearningCVE\".
  Please download the original UNSW-NB15 dataset from the website: https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files
2. Preprocess the data.
   For each dataset, there is a **-preprocess.ipynb file for preprocessing the data. Note that some file path need to be manully created.
3. Do experiments.
   Experiments of AE and VAE are in the same file with different comment source code. The same as the BAE-relevance models.
   
      
