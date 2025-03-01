import os
import sys
import json
import argparse
import logging
import tensorflow as tf
from datetime import datetime

# Import project modules
from model import Generator, Discriminator
from data_processing import NASAPowerDataProcessor
from train_evaluate import GHIForecaster
from visualization import GHIVisualizer
from enhanced_visualization import EnhancedGHIVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ghi_forecasting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ghi_ast_main')

# Check for macOS Metal support


def check_metal_support():
    try:
        # Check if tensorflow-metal is available
        import tensorflow_metal
        logger.info("tensorflow-metal package is installed")

        # Check available devices
        devices = tf.config.list_physical_devices()
        logger.info(f"Available devices: {devices}")

        # Check for MPS device
        mps_devices = tf.config.list_physical_devices('MPS')
        if mps_devices:
            logger.info(f"MPS (Metal) devices found: {mps_devices}")
            return True
        else:
            logger.warning("No MPS devices found, Metal may not be supported")
            return False
    except ImportError:
        logger.warning(
            "tensorflow-metal package not found, Metal acceleration not available")
        return False
    except Exception as e:
        logger.warning(f"Error checking Metal support: {str(e)}")
        return False


def create_directories(config):
    """Create necessary directories for the project."""
    os.makedirs(config.get('data_dir', 'data'), exist_ok=True)
    os.makedirs(config.get('model_dir', 'models/ghi_ast'), exist_ok=True)
    os.makedirs(config.get('output_dir', 'output'), exist_ok=True)
    os.makedirs(config.get('visualization_dir',
                'visualizations'), exist_ok=True)
    os.makedirs("thesis_visualizations", exist_ok=True)


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def train_model(config, args):
    """Train the GHI forecasting model."""
    logger.info("Starting model training workflow")

    try:
        # Initialize data processor
        data_processor = NASAPowerDataProcessor(
            lookback_history=config.get('lookback_history', 168),
            estimate_length=config.get('forecast_horizon', 24),
            params=config.get('parameters', None),
            locations=config.get('locations', None)
        )

        # Prepare data
        logger.info("Preparing datasets")
        datasets = data_processor.prepare_data(
            start_date=config.get('start_date', '20180101'),
            end_date=config.get('end_date', '20241031'),
            train_split=config.get('train_split', 0.8),
            val_split=config.get('val_split', 0.1),
            batch_size=config.get('batch_size', 32),
            save_dir=config.get('data_dir', 'data')
        )

        # Initialize forecaster
        logger.info("Initializing forecaster")
        use_metal = args.use_metal or check_metal_support()

        forecaster = GHIForecaster(config, use_metal=use_metal)

        # Train model
        logger.info("Starting training")
        history = forecaster.train(
            datasets['train'],
            datasets['validation'],
            epochs=config.get('epochs', 100)
        )

        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_metrics = forecaster.evaluate(datasets['test'])
        logger.info(f"Test metrics: {test_metrics}")

        # Create visualizations
        logger.info("Creating visualizations")
        visualizer = GHIVisualizer(output_dir=config.get(
            'visualization_dir', 'visualizations'))

        # Plot training history
        visualizer.plot_training_history(
            history,
            title='GHI Forecasting Training History',
            save_path='training_history.png'
        )

        # Get sample predictions for visualization
        sample_batches = next(iter(datasets['test'].take(1)))
        sample_inputs = sample_batches[0]
        sample_targets = sample_batches[1].numpy()

        sample_predictions = forecaster.predict({
            'historical': sample_inputs['historical'],
            'future_covariates': sample_inputs['future_covariates']
        })

        # After evaluating the model and getting test_metrics
        logger.info("Creating enhanced thesis visualizations")
        enhanced_viz = EnhancedGHIVisualizer(
            output_dir="thesis_visualizations")

        # Generate a complete set of thesis visualizations
        enhanced_viz.generate_thesis_visualizations(
            forecaster, datasets, test_metrics)

        # Create a specific visualization (example)
        enhanced_viz.plot_multihorizon_comparison(
            sample_targets,
            sample_predictions,
            horizons=[6, 12, 24],
            title="GHI Forecast Performance at Different Horizons",
            save_path="multihorizon_comparison.png"
        )

        # Create model architecture diagram
        enhanced_viz.plot_model_architecture(save_path="ast_architecture.png")

        # Create error analysis dashboard
        enhanced_viz.plot_error_analysis_dashboard(
            sample_targets,
            sample_predictions,
            title="GHI Forecast Error Analysis Dashboard",
            save_path="error_dashboard.png"
        )

        # Plot sample predictions
        for i in range(min(5, sample_targets.shape[0])):
            visualizer.plot_forecast_comparison(
                sample_targets[i],
                sample_predictions[i],
                title=f'Sample Forecast {i+1}',
                save_path=f'sample_forecast_{i+1}.png'
            )
        # After generating a prediction
        attention_layer = forecaster.generator.transformer_layer.encoder.layers[
            0].self_attention_layer
        attention_weights = attention_layer.last_attention_weights.numpy()[
            0]  # First batch item

        # Visualize attention patterns
        enhanced_viz.plot_attention_heatmap(
            attention_weights,
            sequence_length=168,
            title="Sparse Attention Patterns in AST",
            save_path="attention_heatmap.png"
        )
        logger.info("Training workflow completed successfully")
        return test_metrics

    except tf.errors.InvalidArgumentError as e:
        # Handle TensorFlow reshape errors specifically
        if "Cannot reshape" in str(e):
            logger.error(
                "Reshape error in model. This may be due to inconsistent tensor shapes. Try reducing batch size or model complexity.")
            logger.error(f"Detailed error: {str(e)}")
        else:
            raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def evaluate_model(config):
    """Evaluate a trained GHI forecasting model."""
    logger.info("Starting model evaluation workflow")

    try:
        # Initialize forecaster
        forecaster = GHIForecaster(config)

        # Initialize data processor for test data
        data_processor = NASAPowerDataProcessor(
            lookback_history=config.get('lookback_history', 168),
            estimate_length=config.get('forecast_horizon', 24),
            params=config.get('parameters', None),
            locations=config.get('locations', None)
        )

        # Prepare evaluation data
        logger.info("Preparing evaluation dataset")
        eval_datasets = data_processor.prepare_data(
            start_date=config.get('eval_start_date', '20240701'),
            end_date=config.get('eval_end_date', '20241031'),
            train_split=0.0,  # All data for testing
            val_split=0.0,
            batch_size=config.get('batch_size', 32),
            save_dir=os.path.join(config.get('data_dir', 'data'), 'eval')
        )

        # Evaluate model
        logger.info("Performing evaluation")
        eval_metrics = forecaster.evaluate(
            eval_datasets['test'],
            model_path=config.get('model_path', None)
        )
        logger.info(f"Evaluation metrics: {eval_metrics}")

        # Create visualizations
        logger.info("Creating evaluation visualizations")
        visualizer = GHIVisualizer(output_dir=config.get(
            'visualization_dir', 'visualizations'))

        # After evaluating the model and getting test_metrics
        logger.info("Creating enhanced thesis visualizations")
        enhanced_viz = EnhancedGHIVisualizer(
            output_dir="thesis_visualizations")

        # Get sample predictions for visualization
        sample_batches = next(iter(eval_datasets['test'].take(1)))
        sample_inputs = sample_batches[0]
        sample_targets = sample_batches[1].numpy()

        sample_predictions = forecaster.predict({
            'historical': sample_inputs['historical'],
            'future_covariates': sample_inputs['future_covariates']
        })
        # Generate a complete set of thesis visualizations
        enhanced_viz.generate_thesis_visualizations(
            forecaster, eval_datasets, eval_metrics)

        # Create a specific visualization (example)
        enhanced_viz.plot_multihorizon_comparison(
            sample_targets,
            sample_predictions,
            horizons=[6, 12, 24],
            title="GHI Forecast Performance at Different Horizons",
            save_path="multihorizon_comparison.png"
        )

        # Create model architecture diagram
        enhanced_viz.plot_model_architecture(save_path="ast_architecture.png")

        # Create error analysis dashboard
        enhanced_viz.plot_error_analysis_dashboard(
            sample_targets,
            sample_predictions,
            title="GHI Forecast Error Analysis Dashboard",
            save_path="error_dashboard.png"
        )
        # Plot sample predictions
        for i in range(min(5, sample_targets.shape[0])):
            visualizer.plot_forecast_comparison(
                sample_targets[i],
                sample_predictions[i],
                title=f'Evaluation Forecast {i+1}',
                save_path=f'eval_forecast_{i+1}.png'
            )
        # After generating a prediction
        attention_layer = forecaster.generator.transformer_layer.encoder.layers[
            0].self_attention_layer
        attention_weights = attention_layer.last_attention_weights.numpy()[
            0]  # First batch item

        # Visualize attention patterns
        enhanced_viz.plot_attention_heatmap(
            attention_weights,
            sequence_length=168,
            title="Sparse Attention Patterns in AST",
            save_path="attention_heatmap.png"
        )
        logger.info("Evaluation workflow completed successfully")
        return eval_metrics

    except Exception as e:
        logger.error(f"Error in evaluation workflow: {str(e)}")
        raise


def predict_ghi(config):
    """Generate GHI predictions for a specific case."""
    logger.info("Starting prediction workflow")

    try:
        # Initialize forecaster
        forecaster = GHIForecaster(config)

        # Initialize data processor for test data
        data_processor = NASAPowerDataProcessor(
            lookback_history=config.get('lookback_history', 168),
            estimate_length=config.get('forecast_horizon', 24),
            params=config.get('parameters', None),
            locations=[config.get('prediction_location', None)]
        )

        # Fetch data for the prediction period
        logger.info("Fetching data for prediction")
        prediction_data = data_processor.fetch_data(
            start_date=config.get('prediction_start_date'),
            end_date=config.get('prediction_end_date'),
            location=config.get('prediction_location'),
            save_path=os.path.join(config.get(
                'data_dir', 'data'), 'prediction_data.csv')
        )

        # Preprocess and engineer features
        prediction_data = data_processor.preprocess_data(prediction_data)
        prediction_data = data_processor.engineer_features(prediction_data)
        prediction_data = data_processor.normalize_features(
            prediction_data, is_training=False)

        # Create sequences
        sequences = data_processor.create_sequences(prediction_data)

        # Make predictions
        logger.info("Generating predictions")
        all_predictions = []

        for i, input_seq in enumerate(sequences['inputs']):
            prediction = forecaster.predict({
                'historical': np.expand_dims(input_seq['historical'], axis=0),
                'future_covariates': np.expand_dims(input_seq['future_covariates'], axis=0)
            })
            all_predictions.append(prediction[0])

        # Create visualizations
        logger.info("Creating prediction visualizations")
        visualizer = GHIVisualizer(output_dir=config.get(
            'visualization_dir', 'visualizations'))

        # Get dates for x-axis
        dates = prediction_data.index[data_processor.lookback_history:
                                      data_processor.lookback_history + data_processor.estimate_length]

        # Plot predictions
        for i, (target, prediction) in enumerate(zip(sequences['targets'], all_predictions)):
            # Only plot the first few predictions
            if i >= 5:
                break

            visualizer.plot_forecast_comparison(
                target,
                prediction,
                dates=dates,
                title=f'GHI Forecast for {config["prediction_location"]["name"]}',
                save_path=f'prediction_{i+1}.png'
            )

        logger.info("Prediction workflow completed successfully")

    except Exception as e:
        logger.error(f"Error in prediction workflow: {str(e)}")
        raise


def main():
    """Main function to parse arguments and execute the appropriate workflow."""
    parser = argparse.ArgumentParser(
        description='GHI Forecasting with Adversarial Sparse Transformer')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], default='train',
                        help='Mode of operation')
    parser.add_argument('--use_metal', action='store_true',
                        help='Use Metal GPU acceleration on macOS')

    args = parser.parse_args()

    # Check if platform is macOS
    import platform
    is_macos = platform.system() == 'Darwin'

    # Setup GPU - specifically for macOS
    if is_macos:
        if args.use_metal or check_metal_support():
            logger.info("Configuring Metal for GPU acceleration")
            try:
                # Make Metal device visible
                tf.config.experimental.set_visible_devices(
                    [], 'GPU')  # Hide standard GPUs
                logger.info("Metal device configured for acceleration")
            except Exception as e:
                logger.error(f"Error configuring Metal device: {str(e)}")
                logger.info("Falling back to CPU")
        else:
            logger.info("Using CPU compute on macOS")
    else:
        # Standard GPU setup for non-macOS platforms
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logger.info("Using GPU for computation")
            else:
                logger.info("No GPUs found, using CPU")
        except Exception as e:
            logger.error(f"Error setting up GPU: {str(e)}")
            logger.info("Falling back to CPU")

    # Load configuration
    config = load_config(args.config)

    # Create required directories
    create_directories(config)

    # Execute appropriate workflow
    if args.mode == 'train':
        train_model(config, args)
    elif args.mode == 'evaluate':
        evaluate_model(config)
    elif args.mode == 'predict':
        predict_ghi(config)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    start_time = datetime.now()
    logger.info(f"Starting GHI forecasting application at {start_time}")

    try:
        main()
        end_time = datetime.now()
        logger.info(
            f"Application completed successfully in {end_time - start_time}")
    except Exception as e:
        logger.exception("Application failed with exception")
        sys.exit(1)
