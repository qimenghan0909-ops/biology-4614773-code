# HSI-SPAD-TwoTier

## Two-Tier Grid Search Optimization Framework for Hyperspectral-Based SPAD Estimation

This repository contains the code for the two-tier grid search optimization framework used in the study "Predicting Maize Leaf SPAD Values from Hyperspectral Imaging Using a Two-Tier Grid Search Framework".

## Overview

The framework systematically evaluates **576 method combinations** across four modeling stages:

1. **Sample Partitioning** - Split data into train/test sets
2. **Preprocessing** - Apply spectral preprocessing methods
3. **Feature Selection** - Select informative wavelengths
4. **Regression Modeling** - Build prediction models

**Key Design Principle**: To avoid data leakage, all preprocessing and feature selection operations are performed **only on training data**. Parameters learned from training are then applied to transform test data.

## Features

- **576 Method Combinations**: Systematic evaluation of preprocessing methods, wavelength selection algorithms, sample partitioning strategies, and regression models
- **Two-Tier Optimization**: First-tier grid search for method selection, second-tier for hyperparameter tuning
- **Data Leakage Prevention**: All preprocessing and feature selection are fitted on training data only
- **Pixel-Level SPAD Mapping**: Generate spatial distribution maps of SPAD values across maize leaves
- **GLCM Texture Analysis**: Extract texture features and leaf coverage from hyperspectral images

## Supported Methods

| Stage | Methods |
|-------|---------|
| **Preprocessing** (8) | None, MA, SG, MSC, D1, D2, DT, WAVE |
| **Feature Selection** (6) | None, CARS, LARS, UVE, SPA, GA |
| **Sample Partitioning** (3) | Random, KS, SPXY |
| **Regression Models** (4) | PLS, SVR, LGBM, RF |

**Total combinations**: 8 × 6 × 3 × 4 = 576

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/HSI-SPAD-TwoTier.git
   cd HSI-SPAD-TwoTier
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Data Preparation

Prepare your data in the following format:
- **Spectral data**: Excel file with shape (n_samples, n_wavelengths)
- **SPAD labels**: Excel file with shape (n_samples, 2) containing [SampleID, SPAD_Mean]

Place data files in the `data/` directory.

### Step 2: First-Tier Grid Search

Evaluate 576 method combinations to identify optimal pipelines:

```bash
python "tier1_algorithm combinations.py"
```

This evaluates all combinations of:
- 8 preprocessing methods
- 6 wavelength selection methods
- 3 sample partitioning methods
- 4 regression models

Each combination is run 10 times with different random seeds to ensure robustness.

### Step 3: Second-Tier Hyperparameter Optimization

Optimize hyperparameters for selected top pipelines:

```bash
python "tier2_hyperparameter optimization.py"
```

This optimizes:
- CARS iterations (2-50)
- SVR penalty parameter C (1-100)
- SVR kernel type (linear, rbf)
- Other method-specific parameters

### Step 4: Pixel-Level SPAD Prediction

Generate spatial distribution maps of SPAD values:

```bash
# 1. Train and save the model
python SPAD_predicted/save_model.py

# 2. Smooth pixel spectra
python SPAD_predicted/Smoothing_spectrum.py

# 3. Predict SPAD for each pixel
python SPAD_predicted/Predicting_spectrapy.py

# 4. Generate SPAD pseudo-color maps
python SPAD_predicted/SPAD.py
```

### Step 5: GLCM Texture Analysis

Extract texture features and calculate leaf coverage:

```bash
python glcm_texture_and_leaf_coverage.py
```

This script:
- Extracts 14 Haralick texture features from hyperspectral images
- Calculates leaf coverage (ratio of leaf pixels to total pixels)
- Saves results to Excel files

## Output

### First-Tier Results

- `all_new_models_combined_10runs.xlsx`: Complete results for all 576 combinations × 10 runs
- Contains: Train/Test R², RMSE, MAE for each combination

### Second-Tier Results

- Optimized hyperparameters for selected pipelines
- Improved performance metrics

### Visualization

- Diagonal plots comparing predicted vs. actual SPAD values
- Preprocessing effect visualizations
- Spatial SPAD distribution maps

## Data Availability

The anonymized dataset used in this study is available from the corresponding author upon reasonable request.

## Citation

If you use this code in your research, please cite our paper:


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

