# Global Horizontal Irradiance (GHI) Forecasting with Adversarial Sparse Transformer (AST)

This repository implements an advanced deep learning model for solar irradiance forecasting using an Adversarial Sparse Transformer (AST) architecture. The model is designed to predict Global Horizontal Irradiance (GHI) values with high accuracy for short-term forecasting horizons.

## Overview

The AST model combines the powerful attention mechanisms of transformers with sparse attention patterns to efficiently process long sequences of time-series data. The implementation includes:

- Custom Sparse Transformer architecture
- Adversarial training framework
- Adaptive learning components
- NASA POWER API integration for data collection

## Model Architecture

The model consists of several key components:

- **Sparse Multi-Head Attention**: Implements efficient attention patterns
- **Alpha-Entmax Activation**: Provides sparse attention weights
- **Positional Encoding**: Captures temporal relationships
- **Encoder-Decoder Structure**: Processes input sequences and generates predictions
- **Discriminator**: Improves prediction quality through adversarial training

## Requirements

```
tensorflow>=2.0.0
torch>=1.8.0
numpy
pandas
requests
scikit-learn
matplotlib
```

## Data Collection

The model uses NASA POWER API to collect hourly GHI data. To fetch data:

1. Configure your location parameters in `fetch_power_data()`
2. Specify the date range
3. The API will return hourly GHI values

Example:
```python
data = fetch_power_data(
    lat=40.7128, 
    lon=-74.0060,
    start_date="20230101",
    end_date="20231231"
)
```

## Model Training

The model can be trained using the following steps:

1. Preprocess the data:
```python
preprocessor = DataPreprocessor(window_size=24, prediction_hours=1)
normalized_data = preprocessor.normalize_data(ghi_data)
X, y = preprocessor.create_sequences(normalized_data)
```

2. Train the model:
```python
model = AST(seq_len=168, pred_len=24)
train_model(model, train_dataset, val_dataset)
```

## Performance Metrics



## Results Visualization


## Citation


## Acknowledgments

- NASA POWER Project for providing the solar irradiance data
- The authors of the original Transformer architecture
- Contributors to the PyTorch and TensorFlow frameworks

## Contact

For questions and feedback, please open an issue in the GitHub repository.
