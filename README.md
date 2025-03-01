# Global Horizontal Irradiance (GHI) Forecasting using Adversarial Sparse Transformer

This project implements an Adversarial Sparse Transformer (AST) architecture for Global Horizontal Irradiance (GHI) forecasting, as described in the thesis "Global Horizontal Irradiance Forecasting using Adversarial Sparse Transformer" (2025).

## Overview

The AST architecture combines sparse attention mechanisms through α-entmax transformation (α=1.5) with adversarial training procedures to capture both deterministic patterns and stochastic variations in solar radiation patterns. The model is designed to improve forecasting accuracy and temporal stability across multiple forecast horizons.

## Features

- **Sparse Attention Mechanism**: Implements α-entmax transformation to focus on relevant historical time steps
- **Adversarial Training Framework**: Uses a discriminator network for sequence-level evaluation
- **Enhanced Temporal Modeling**: Captures both short-term fluctuations and long-term patterns
- **NASA POWER Dataset Integration**: Processes solar radiation and meteorological data
- **Comprehensive Evaluation**: Reports multiple performance metrics including MAE, RMSE, MAPE, and HSI
- **Visualization Tools**: Generates plots and reports for model performance analysis

## Project Structure

```
ghi-forecasting/
├── config/
│   └── ghi_ast_config.json       # Configuration file
├── data/                         # Data directory
├── models/                       # Saved models
├── output/                       # Output files
├── visualizations/               # Visualization output
├── model.py                      # AST model implementation
├── data_processing.py            # NASA POWER data processing
├── train_evaluate.py             # Training and evaluation code
├── visualization.py              # Visualization tools
├── ghi_ast_main.py               # Main runner script
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.5+
- TensorFlow Addons
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Requests

You can install the requirements using:

```bash
pip install tensorflow tensorflow-addons numpy pandas matplotlib seaborn requests
```

## Quick Start

### 1. Configure the Model

Edit the configuration file `config/ghi_ast_config.json` or create a new one to adjust model parameters, data sources, and training settings.

### 2. Training the Model

To train the model:

```bash
python ghi_ast_main.py --config config/ghi_ast_config.json --mode train --gpu 0
```

This will:

- Download and process data from NASA POWER API
- Create and train the AST model
- Save model checkpoints
- Generate training visualizations

### 3. Evaluation

To evaluate a trained model:

```bash
python ghi_ast_main.py --config config/ghi_ast_config.json --mode evaluate --gpu 0
```

This will:

- Load the trained model
- Evaluate on the test dataset
- Generate evaluation metrics and visualizations
- Create an HTML report

### 4. Prediction

To generate predictions for a specific location and time range:

```bash
python ghi_ast_main.py --config config/ghi_ast_config.json --mode predict --gpu 0
```

## Model Architecture

The AST architecture consists of two main components:

### Generator Network

- **Encoder-Decoder Architecture**: Processes historical data and generates future GHI values
- **Sparse Multi-Head Attention**: Uses α-entmax transformation (α=1.5) for focused attention
- **Position Encodings**: Applies sinusoidal position encoding for temporal awareness
- **Normalization**: Implements layer normalization for training stability

### Discriminator Network

- **Sequence-Level Evaluation**: Assesses the realism of generated GHI forecasts
- **Hierarchical Feature Extraction**: Processes temporal patterns at multiple scales
- **Adaptive Regularization**: Provides adversarial feedback to the generator

## Data Processing

The data processing pipeline includes:

1. **Data Acquisition**: Fetches data from NASA POWER API
2. **Preprocessing**: Handles missing values and normalizes features
3. **Feature Engineering**: Creates derived features capturing atmospheric interactions
4. **Sequence Formation**: Generates input-target pairs using sliding windows

## Parameters

Key configurable parameters include:

- `lookback_history`: Length of historical data sequence (default: 168 hours)
- `forecast_horizon`: Length of forecast period (default: 24 hours)
- `num_attention_heads`: Number of attention heads (default: 8)
- `head_size`: Dimension of each attention head (default: 32)
- `hidden_size`: Hidden representation dimension (default: 256)
- `lambda_adversarial`: Weight of adversarial loss (default: 0.1)

See the full configuration template for all available parameters.

## Performance Metrics

The model is evaluated using multiple metrics:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **HSI**: Horizon-specific Stability Index, measuring forecast stability across horizons

## Visualization

Visualization tools include:

- Forecast comparison plots
- Error distribution analysis
- Temporal stability visualization
- Attention pattern visualization
- Training history plots
- HTML performance reports

## Extending the Project

### Adding New Locations

Edit the `locations` array in the configuration file to add new geographical locations for forecasting.

### Custom Feature Engineering

Extend the `engineer_features` method in `data_processing.py` to create additional derived features.

### Model Modifications

Adjust the AST architecture in `model.py` to experiment with different attention mechanisms or network configurations.

## Citation

If you use this code in your research, please cite:

```
Kashyap, S. S. (2025). Global Horizontal Irradiance Forecasting using Adversarial Sparse Transformer.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA for providing the POWER dataset
- Authors of the α-entmax transformation paper (Peters et al., 2019)
- Authors of "Adversarial Sparse Transformer for Time Series Forecasting" (Wu et al., 2020)
