import tensorflow as tf
import numpy as np
import os
import time
import platform
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger('ghi_forecasting')


class GHIForecaster:
    """
    Main class for training and evaluating GHI forecasting model.
    """

    def __init__(self, config, use_metal=False):
        """
        Initialize GHI forecaster with configuration parameters.

        Parameters:
        - config: Dictionary containing model and training configuration
        """
        self.config = config
        self.model_dir = config.get('model_dir', 'models/ghi_ast')
        self.use_metal = use_metal

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

        # Set up metal-specific strategy if on macOS
        self.strategy = self._setup_strategy()

        # Initialize generator and discriminator
        self._initialize_models()

        # Initialize optimizers
        self._initialize_optimizers()

        # Define loss scaling factors
        self.lambda_adv = config.get('lambda_adversarial', 0.1)
        self.quantile = config.get('quantile', 0.5)

        # Setup checkpointing
        self.checkpoint_prefix = os.path.join(self.model_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        # For TensorBoard
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.model_dir, 'logs',
                         datetime.now().strftime("%Y%m%d-%H%M%S"))
        )

    def _setup_strategy(self):
        """Setup appropriate distribution strategy based on platform."""
        # Check if we're on macOS and Metal should be used
        is_macos = platform.system() == 'Darwin'

        if is_macos and self.use_metal:
            logger.info(
                "Using MPS (Metal) device for training as configured by main")
            return None  # No explicit strategy needed for Metal
        else:
            # For standard GPU acceleration
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    return tf.distribute.OneDeviceStrategy(device="/gpu:0")
                else:
                    logger.info("No GPUs found, using CPU")
                    return None
            except:
                logger.info("Error setting up GPU strategy, using CPU")
                return None

    def _initialize_models(self):
        """Initialize generator and discriminator with more stable settings."""
        # Import inside function to allow for potential distribution strategy
        from model import Generator, Discriminator

        # Generator with reduced initializer_range for stability
        self.generator = Generator(
            lookback_history=self.config.get('lookback_history', 168),
            estimate_length=self.config.get('forecast_horizon', 24),
            num_features=self.config.get('num_features', 9),
            embedding_size=self.config.get('embedding_size', 20),
            hidden_size=self.config.get('hidden_size', 256),
            feedforward_size=self.config.get('feedforward_size', 1024),
            num_hidden_layers=self.config.get('num_hidden_layers', 3),
            num_attention_heads=self.config.get('num_attention_heads', 8),
            head_size=self.config.get('head_size', 32),
            activation_fn=None,  # Linear output for GHI values
            dropout_prob=self.config.get('dropout_prob', 0.1),
            initializer_range=0.02,  # Reduced from 1.0 for stability
            is_training=True
        )

        # Discriminator without batch_size parameter and reduced initializer_range
        self.discriminator = Discriminator(
            sequence_length=self.config.get('forecast_horizon', 24),
            hidden_size=self.config.get('hidden_size', 256),
            dropout_prob=0.1  # Increased dropout for better generalization
        )
        # Log model summary for debuggability
        logger.info("Generator and Discriminator initialized successfully")
        logger.info(
            f"Generator input shape: batch × {self.config.get('lookback_history', 168)} × {self.config.get('num_features', 9)}")
        logger.info(
            f"Discriminator input shape: batch × {self.config.get('forecast_horizon', 24)}")

    def _initialize_optimizers(self):
        """Initialize optimizers with better learning rates and settings."""
        # Generator uses a lower learning rate for stability
        gen_lr = self.config.get('learning_rate', 2e-5)

        # Discriminator uses a higher learning rate to learn faster
        # This helps avoid the "stuck discriminator" problem
        disc_lr = self.config.get('disc_learning_rate', 5e-5)

        # Create optimizers with improved settings
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=gen_lr,
            beta_1=0.0,     # Lower beta1 helps with training stability
            beta_2=0.999,
            epsilon=1e-8
        )

        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=disc_lr,
            beta_1=0.0,     # Lower beta1 helps with training stability
            beta_2=0.999,
            epsilon=1e-8
        )

        # Learning rate schedulers for better convergence
        self.gen_lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=gen_lr,
            decay_steps=10000,
            alpha=0.1  # Minimum learning rate is 10% of initial
        )

        self.disc_lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=disc_lr,
            decay_steps=10000,
            alpha=0.1  # Minimum learning rate is 10% of initial
        )

    def _create_lr_schedule(self, initial_learning_rate, warmup_steps, decay_steps, min_learning_rate):
        """
        Create a fixed learning rate instead of a schedule to avoid TF graph issues.

        Parameters:
        - initial_learning_rate: Fixed learning rate to use

        Returns:
        - Fixed learning rate
        """
        # Instead of a complex schedule, just use a fixed learning rate
        # This avoids graph execution issues with TF 2.10 on macOS
        return initial_learning_rate

    def quantile_loss(self, y_true, y_pred, quantile=None):
        """
        Compute quantile loss for GHI prediction.

        Parameters:
        - y_true: True GHI values of shape [batch_size, sequence_length, 1]
        - y_pred: Predicted GHI values of shape [batch_size, sequence_length]
        - quantile: Quantile value (defaults to self.quantile)

        Returns:
        - Quantile loss value
        """
        if quantile is None:
            quantile = self.quantile

        # Ensure dimensions match by squeezing y_true if it has an extra dimension
        if len(tf.shape(y_true)) > len(tf.shape(y_pred)):
            y_true = tf.squeeze(y_true, axis=-1)

        # Or alternatively, add a dimension to y_pred if needed
        elif len(tf.shape(y_pred)) < len(tf.shape(y_true)):
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Now compute the error with matched dimensions
        error = y_true - y_pred
        loss = tf.maximum(quantile * error, (quantile - 1) * error)
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, historical_data, future_covariates, y_true):
        """
        Enhanced training step for GHI forecasting with better GAN stability.
        """
        # Adaptive adversarial weight - start very small and grow gradually
        iteration = tf.cast(self.generator_optimizer.iterations, tf.float32)
        adv_scale = tf.minimum(0.1, 0.001 + iteration / 10000.0)

        ##################
        # DISCRIMINATOR TRAINING
        ##################

        # Use fixed 1 iteration for discriminator (more stable)
        with tf.GradientTape() as disc_tape:
            # Generate predictions
            y_pred = self.generator(historical_data, future_covariates)

            # Ensure dimensions match
            y_pred_flat = tf.squeeze(y_pred)
            y_true_flat = tf.squeeze(y_true)

            # Replace NaN values
            y_pred_flat = tf.where(tf.math.is_nan(y_pred_flat),
                                   tf.zeros_like(y_pred_flat),
                                   y_pred_flat)

            # Add noise for stabilization
            noise_level = 0.05 * tf.exp(-iteration / 10000.0)  # Decaying noise
            real_noisy = y_true_flat + tf.random.normal(tf.shape(y_true_flat),
                                                        mean=0.0, stddev=noise_level)
            fake_noisy = y_pred_flat + tf.random.normal(tf.shape(y_pred_flat),
                                                        mean=0.0, stddev=noise_level)

            # Get discriminator outputs
            real_output = self.discriminator(real_noisy, training=True)
            fake_output = self.discriminator(fake_noisy, training=True)

            # One-sided label smoothing (0.9 for real, 0.0 for fake)
            real_labels = 0.9 * tf.ones_like(tf.squeeze(real_output))
            fake_labels = tf.zeros_like(tf.squeeze(fake_output))

            # Calculate losses
            real_loss = tf.keras.losses.binary_crossentropy(
                real_labels,
                tf.squeeze(real_output)
            )
            fake_loss = tf.keras.losses.binary_crossentropy(
                fake_labels,
                tf.squeeze(fake_output)
            )

            disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

        # Calculate and apply discriminator gradients
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        # Gradient clipping for stability
        disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 1.0)

        # Apply only if valid gradients
        if all(g is not None for g in disc_gradients):
            self.discriminator_optimizer.apply_gradients(
                zip(disc_gradients, self.discriminator.trainable_variables)
            )

        # Calculate discriminator metrics
        real_accuracy = tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32))
        fake_accuracy = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
        disc_accuracy = (real_accuracy + fake_accuracy) / 2.0

        ##################
        # GENERATOR TRAINING
        ##################

        with tf.GradientTape() as gen_tape:
            # Generate predictions again
            y_pred = self.generator(historical_data, future_covariates)

            # Ensure dimensions match
            y_pred_flat = tf.squeeze(y_pred)
            y_true_flat = tf.squeeze(y_true)

            # Replace NaN values
            y_pred_flat = tf.where(tf.math.is_nan(y_pred_flat),
                                   tf.zeros_like(y_pred_flat),
                                   y_pred_flat)

            # Get discriminator outputs for generated data
            fake_output = self.discriminator(y_pred_flat, training=False)

            # --- Standard Regression Losses ---

            # MSE loss (primary regression objective)
            mse_loss = tf.reduce_mean(tf.square(y_true_flat - y_pred_flat))

            # L1 loss (for robustness)
            l1_loss = tf.reduce_mean(tf.abs(y_true_flat - y_pred_flat))

            # --- Regularization Losses ---

            # Temporal gradient consistency loss
            if y_true_flat.shape[1] > 1:  # Only if sequence length > 1
                temp_grad_true = y_true_flat[:, 1:] - y_true_flat[:, :-1]
                temp_grad_pred = y_pred_flat[:, 1:] - y_pred_flat[:, :-1]
                grad_loss = tf.reduce_mean(
                    tf.abs(temp_grad_true - temp_grad_pred))
            else:
                grad_loss = 0.0

            # Feature matching from discriminator (better than just adversarial loss)
            # Extract features from real and fake samples
            real_features = self.discriminator.feature_matching(
                y_true_flat, training=False)
            fake_features = self.discriminator.feature_matching(
                y_pred_flat, training=False)

            # Calculate feature matching loss
            feature_loss = 0.0
            for k in real_features:
                if real_features[k] is not None and fake_features[k] is not None:
                    feature_loss += tf.reduce_mean(
                        tf.square(real_features[k] - fake_features[k])
                    )

            # --- Adversarial Loss ---

            # Non-saturating GAN loss (better gradient flow than minimax)
            adv_loss = -tf.reduce_mean(tf.math.log(fake_output + 1e-8))

            # --- Combined Loss ---

            # Weight the losses based on their importance
            # Primary objective: accurate GHI prediction (MSE)
            # Secondary: realistic time series patterns (adversarial & feature matching)
            gen_loss = mse_loss + 0.2 * l1_loss + 0.05 * grad_loss + \
                0.1 * feature_loss + adv_scale * adv_loss

        # Calculate and apply generator gradients
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)

        # Gradient clipping for stability
        gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, 1.0)

        # Apply only if valid gradients
        if all(g is not None for g in gen_gradients):
            self.generator_optimizer.apply_gradients(
                zip(gen_gradients, self.generator.trainable_variables)
            )

        return {
            'gen_loss': gen_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'grad_loss': grad_loss if y_true_flat.shape[1] > 1 else 0.0,
            'feature_loss': feature_loss,
            'adv_loss': adv_loss,
            'disc_loss': disc_loss,
            'disc_accuracy': disc_accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'adv_scale': adv_scale
        }

    @tf.function
    def pretrain_discriminator_step(self, historical_data, future_covariates, y_true):
        """
        Pre-trains the discriminator with dimension-corrected code.
        Fixes the reduction dimension issue.
        """
        # Generate predictions
        y_pred = self.generator(historical_data, future_covariates)

        # Ensure dimensions match
        y_pred_flat = tf.squeeze(y_pred)
        y_true_flat = tf.squeeze(y_true)

        # Check the dimensions
        # y_true_flat shape is likely [batch_size, sequence_length]

        with tf.GradientTape() as disc_tape:
            # Get discriminator outputs
            real_output = self.discriminator(y_true_flat, training=True)
            fake_output = self.discriminator(y_pred_flat, training=True)

            # Create proper labels based on actual output shape
            # real_output shape is likely [batch_size]
            real_labels = 0.9 * tf.ones_like(real_output)
            fake_labels = tf.zeros_like(fake_output)

            # Binary cross-entropy loss
            real_loss = tf.keras.losses.binary_crossentropy(
                real_labels, real_output)
            fake_loss = tf.keras.losses.binary_crossentropy(
                fake_labels, fake_output)

            disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)

        # Calculate and apply discriminator gradients
        disc_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        # Apply gradients if they're valid
        if all(g is not None for g in disc_gradients):
            # Clip gradients for stability
            disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 0.5)

            self.discriminator_optimizer.apply_gradients(
                zip(disc_gradients, self.discriminator.trainable_variables)
            )

        # Calculate discriminator accuracy
        real_accuracy = tf.reduce_mean(
            tf.cast(real_output > 0.5, tf.float32))
        fake_accuracy = tf.reduce_mean(
            tf.cast(fake_output < 0.5, tf.float32))
        disc_accuracy = (real_accuracy + fake_accuracy) / 2.0

        return {
            'disc_loss': disc_loss,
            'disc_accuracy': disc_accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy
        }

    def train(self, train_dataset, validation_dataset, epochs=100):
        """
        Train with improved curriculum strategy and better monitoring.
        """
        logger.info(
            f"Starting training for {epochs} epochs with improved curriculum strategy")

        # Initialize history dictionary
        history = {}

        # Load checkpoint if it exists
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            logger.info(f"Restored from checkpoint: {latest_checkpoint}")

        # Stage 1: Generator pre-training with MSE only
        # This helps establish a good starting point before adversarial training
        logger.info(
            "Stage 1: Generator pre-training with regression losses only")

        @tf.function
        def pretrain_step(historical_data, future_covariates, y_true):
            with tf.GradientTape() as tape:
                y_pred = self.generator(historical_data, future_covariates)

                # Ensure dimensions match
                y_pred_flat = tf.squeeze(y_pred)
                y_true_flat = tf.squeeze(y_true)

                # Combined loss: MSE + L1 + Gradient consistency
                mse_loss = tf.reduce_mean(tf.square(y_true_flat - y_pred_flat))
                l1_loss = tf.reduce_mean(tf.abs(y_true_flat - y_pred_flat))

                # Temporal gradient consistency
                temp_grad_true = y_true_flat[:, 1:] - y_true_flat[:, :-1]
                temp_grad_pred = y_pred_flat[:, 1:] - y_pred_flat[:, :-1]
                grad_loss = tf.reduce_mean(
                    tf.abs(temp_grad_true - temp_grad_pred))

                total_loss = mse_loss + 0.2 * l1_loss + 0.05 * grad_loss

            gradients = tape.gradient(
                total_loss, self.generator.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            self.generator_optimizer.apply_gradients(
                zip(gradients, self.generator.trainable_variables)
            )

            return {
                'mse_loss': mse_loss,
                'l1_loss': l1_loss,
                'grad_loss': grad_loss,
                'total_loss': total_loss
            }

        # Pre-training phase
        pretrain_epochs = self.config.get('pretrain_epochs', 1)
        for epoch in range(pretrain_epochs):
            start_time = time.time()

            # Track metrics - initialize empty dictionary first
            epoch_metrics = {}

            step = 0
            for inputs, targets in train_dataset:
                historical_data = inputs['historical']
                future_covariates = inputs['future_covariates']

                metrics = pretrain_step(
                    historical_data, future_covariates, targets)

                # Safely record metrics by ensuring keys exist
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(v.numpy())

                if step % 20 == 0:
                    metrics_str = " - ".join([f"{k}: {v:.4f}" for k,
                                              v in metrics.items()])
                    logger.info(
                        f"Pre-training Epoch {epoch+1}, Step {step}: {metrics_str}")

                step += 1

            # Validate after each epoch
            val_metrics = self.evaluate(validation_dataset)

            # Calculate average metrics for display
            avg_metrics = {k: np.mean(v)
                           for k, v in epoch_metrics.items() if v}

            epoch_time = time.time() - start_time
            logger.info(f"Pre-training Epoch {epoch+1}/{pretrain_epochs} - {epoch_time:.2f}s - "
                        f"mse_loss: {avg_metrics.get('mse_loss', 0):.4f} - "
                        f"val_mse: {val_metrics['mse']:.4f} - val_mae: {val_metrics['mae']:.4f}")

            # Save checkpoint
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        # Stage 2: Discriminator pre-training
        # Train discriminator to recognize real vs generated sequences
        logger.info("Stage 2: Discriminator pre-training")

        # Discriminator pre-training phase
        disc_pretrain_epochs = max(1, min(self.config.get(
            'disc_pretrain_epochs', 5), pretrain_epochs))
        logger.info(
            f"Starting discriminator pre-training for {disc_pretrain_epochs} epochs")

        for epoch in range(disc_pretrain_epochs):
            start_time = time.time()

            # Track metrics - initialize empty dictionary
            epoch_metrics = {}

            step = 0
            for inputs, targets in train_dataset:
                historical_data = inputs['historical']
                future_covariates = inputs['future_covariates']

                metrics = self.pretrain_discriminator_step(
                    historical_data, future_covariates, targets)

                # Safely record all metrics
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(v.numpy())

                if step % 20 == 0:
                    metrics_str = " - ".join([f"{k}: {v:.4f}" for k,
                                              v in metrics.items()])
                    logger.info(
                        f"Disc Pre-training Epoch {epoch+1}, Step {step}: {metrics_str}")

                step += 1

            # Calculate average metrics
            avg_metrics = {k: np.mean(v)
                           for k, v in epoch_metrics.items() if v}

            epoch_time = time.time() - start_time
            logger.info(f"Disc Pre-training Epoch {epoch+1}/{disc_pretrain_epochs} - {epoch_time:.2f}s - "
                        f"disc_loss: {avg_metrics.get('disc_loss', 0):.4f} - "
                        f"disc_accuracy: {avg_metrics.get('disc_accuracy', 0):.4f}")

            # Save checkpoint
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        # Stage 3: Full adversarial training with the improved training step
        logger.info(
            "Stage 3: Full adversarial training with curriculum strategy")

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.get('early_stopping_patience', 10)

        # Remaining epochs for full adversarial training
        remaining_epochs = epochs

        for epoch in range(remaining_epochs):
            start_time = time.time()

            # Reset metrics for this epoch - initialize empty dictionary
            epoch_metrics = {}

            step = 0
            for inputs, targets in train_dataset:
                historical_data = inputs['historical']
                future_covariates = inputs['future_covariates']

                # Use the improved training step
                metrics = self.train_step(
                    historical_data, future_covariates, targets)

                # Safely record all metrics
                for k, v in metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(v.numpy())

                if step % 20 == 0:
                    metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()
                                              if k in ['gen_loss', 'mse_loss', 'adv_loss', 'disc_accuracy']])
                    logger.info(
                        f"Adversarial Epoch {epoch+1}, Step {step}: {metrics_str}")

                step += 1

            # Compute average metrics for the epoch
            for k, v in epoch_metrics.items():
                if v:  # Only if we have values
                    if k not in history:
                        history[k] = []
                    history[k].append(np.mean(v))

            # Validation
            val_metrics = self.evaluate(validation_dataset)

            # Add validation metrics to history
            for k, v in val_metrics.items():
                val_key = f'val_{k}'
                if val_key not in history:
                    history[val_key] = []
                history[val_key].append(v)

            # Log epoch summary
            epoch_time = time.time() - start_time
            logger.info(f"Adversarial Epoch {epoch+1}/{remaining_epochs} - {epoch_time:.2f}s - "
                        f"gen_loss: {history['gen_loss'][-1]:.4f} - "
                        f"disc_loss: {history['disc_loss'][-1]:.4f} - "
                        f"disc_accuracy: {history['disc_accuracy'][-1]:.4f} - "
                        f"val_mse: {val_metrics['mse']:.4f} - "
                        f"val_mae: {val_metrics['mae']:.4f}")

            # Early stopping check
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                patience_counter = 0

                # Save the best model
                self.checkpoint.save(file_prefix=os.path.join(
                    self.model_dir, "best_model"))
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Save checkpoint at the end of each epoch
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            # Save training history
            with open(os.path.join(self.model_dir, 'history.json'), 'w') as f:
                json.dump({k: [float(val) for val in v]
                           for k, v in history.items()}, f)

        logger.info("Training completed with improved curriculum strategy")
        return history

    def evaluate(self, dataset, model_path=None):
        """
        Evaluate the model on a dataset.

        Parameters:
        - dataset: TensorFlow dataset for evaluation
        - model_path: Optional path to a specific checkpoint

        Returns:
        - Dictionary of evaluation metrics
        """
        # Load specific model if requested
        if model_path:
            status = self.checkpoint.restore(model_path)
            status.expect_partial()  # Silence warnings about not-restored optimizer variables

        # Initialize metrics
        all_targets = []
        all_predictions = []

        # Prediction loop
        for inputs, targets in dataset:
            historical_data = inputs['historical']
            future_covariates = inputs['future_covariates']

            # Generate predictions
            predictions = self.generator(historical_data, future_covariates)
            # Remove last dimension
            predictions = tf.squeeze(predictions, axis=-1)

            # Store targets and predictions
            all_targets.append(targets.numpy())
            all_predictions.append(predictions.numpy())

        # Check if we have any predictions
        if not all_targets or not all_predictions:
            logger.warning("No evaluation data available")
            return {"mae": 0, "mse": 0, "rmse": 0, "mape": 0}

        # Concatenate batches
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions)

        return metrics

    # Replace your current _calculate_metrics method with this improved version:
    def _calculate_metrics(self, actual, predicted):
        """
        Calculate evaluation metrics with improved MAPE handling.

        Parameters:
        - actual: True GHI values
        - predicted: Predicted GHI values

        Returns:
        - Dictionary of metrics
        """
        # Ensure consistent dimensions
        if len(actual.shape) == 3 and actual.shape[-1] == 1:
            actual = np.squeeze(actual, axis=-1)

        if len(predicted.shape) < len(actual.shape):
            predicted = np.expand_dims(predicted, axis=-1)

        # Create day/night mask (values > 20 W/m² are considered day)
        day_mask = actual > 20.0

        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))

        # Mean Squared Error
        mse = np.mean(np.square(actual - predicted))

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        # Daytime-only metrics (more meaningful for solar forecasting)
        if np.any(day_mask):
            day_mae = np.mean(np.abs(actual[day_mask] - predicted[day_mask]))
            day_mse = np.mean(
                np.square(actual[day_mask] - predicted[day_mask]))
            day_rmse = np.sqrt(day_mse)
        else:
            day_mae = 0
            day_rmse = 0

        # CRITICAL FIX: Better MAPE calculation
        # 1. Use only daytime values (avoids division by small numbers)
        # 2. Apply reasonable minimum threshold (10 W/m²)
        # 3. Cap maximum percentage error at 100% (common practice)
        threshold = 10.0  # Minimum value threshold
        mask = actual > threshold

        if np.any(mask):
            # Calculate normal MAPE with safeguards
            errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
            # Cap at 100% per value to avoid extreme outliers
            errors = np.minimum(errors, 1.0)
            mape = np.mean(errors) * 100
        else:
            mape = 0

        # Alternative: Symmetric MAPE (sMAPE) - more stable for solar forecasting
        # Uses average of actual and predicted in denominator
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        valid_idx = denominator > threshold
        if np.any(valid_idx):
            smape = np.mean(np.abs(actual[valid_idx] - predicted[valid_idx]) /
                            denominator[valid_idx]) * 100
        else:
            smape = 0

        # Horizon-specific Stability Index (HSI) - fixed calculation
        stability_metrics = {}
        for horizon in ['short', 'medium', 'long']:
            if horizon == 'short':
                # 1-6 hours
                h_slice = slice(0, min(6, predicted.shape[1]))
            elif horizon == 'medium':
                # 6-12 hours
                h_slice = slice(6, min(12, predicted.shape[1]))
            else:
                # 12-24 hours (or however many we have)
                h_slice = slice(12, None)

            if h_slice.stop is not None and h_slice.start >= predicted.shape[1]:
                continue

            h_actual = actual[:, h_slice]
            h_predicted = predicted[:, h_slice]

            # Day mask for this horizon
            h_day_mask = h_actual > threshold

            if np.any(h_day_mask):
                # Calculate error for daytime values only
                h_error = h_actual[h_day_mask] - h_predicted[h_day_mask]

                # Standard deviation of errors
                error_std = np.std(h_error)

                # Mean actual value (daytime only)
                mean_actual = np.mean(h_actual[h_day_mask])

                # Modified HSI calculation (guaranteed to be between 0 and 1)
                # Capped to prevent negative values
                if mean_actual > 0:
                    hsi = max(0, 1 - min(1, error_std / mean_actual))
                else:
                    hsi = 0
            else:
                hsi = 0

            stability_metrics[f'hsi_{horizon}'] = hsi

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'day_mae': day_mae,
            'day_rmse': day_rmse,
            'mape': mape,
            'smape': smape,
            **stability_metrics
        }

    def predict(self, input_data):
        """
        Generate predictions for new input data.

        Parameters:
        - input_data: Dictionary with 'historical' and 'future_covariates' keys

        Returns:
        - Predicted GHI values
        """
        historical_data = input_data['historical']
        future_covariates = input_data['future_covariates']

        # Ensure inputs are tensors with batch dimension
        if not isinstance(historical_data, tf.Tensor):
            historical_data = tf.convert_to_tensor(
                historical_data, dtype=tf.float32)
        if not isinstance(future_covariates, tf.Tensor):
            future_covariates = tf.convert_to_tensor(
                future_covariates, dtype=tf.float32)

        # Add batch dimension if needed
        if len(historical_data.shape) == 2:
            historical_data = tf.expand_dims(historical_data, 0)
        if len(future_covariates.shape) == 2:
            future_covariates = tf.expand_dims(future_covariates, 0)

        # Generate predictions
        predictions = self.generator(historical_data, future_covariates)

        # Remove last dimension
        predictions = tf.squeeze(predictions, axis=-1)

        return predictions.numpy()
