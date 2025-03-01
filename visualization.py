import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import json
from datetime import datetime, timedelta


class GHIVisualizer:
    """
    Visualization tools for GHI forecasting results.
    """

    def __init__(self, output_dir='visualizations'):
        """
        Initialize the visualizer.

        Parameters:
        - output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set Seaborn style
        sns.set_style('whitegrid')
        sns.set_context('talk')

        # Custom color palette for GHI visualizations
        self.ghi_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',
                           '#4292c6', '#2171b5', '#08519c', '#08306b']
        self.ghi_cmap = LinearSegmentedColormap.from_list(
            'GHI_cmap', self.ghi_colors)

        # For attention visualization
        self.attention_cmap = 'viridis'

    def plot_forecast_comparison(self, actual, predicted, dates=None,
                                 title='GHI Forecast Comparison',
                                 save_path=None):
        """
        Plot actual vs predicted GHI values.

        Parameters:
        - actual: True GHI values
        - predicted: Predicted GHI values
        - dates: Datetime indices for x-axis
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create x-axis values
        if dates is None:
            x = np.arange(len(actual))
            x_label = 'Hours'
        else:
            x = dates
            x_label = 'Date'

        # Plot data
        ax.plot(x, actual, 'b-', linewidth=2, label='Actual GHI')
        ax.plot(x, predicted, 'r--', linewidth=2, label='Predicted GHI')

        # Add uncertainty band (example - would be replaced with actual prediction intervals)
        ax.fill_between(x, predicted*0.9, predicted*1.1, color='r', alpha=0.2)

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel('Global Horizontal Irradiance (kW/m²)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if isinstance(x[0], datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)

        plt.tight_layout()

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def plot_forecast_horizon_comparison(self, actual, predicted, horizons=[6, 12, 24],
                                         title='Forecast Accuracy by Horizon',
                                         save_path=None):
        """
        Plot forecast accuracy across different horizons.

        Parameters:
        - actual: True GHI values
        - predicted: Predicted GHI values
        - horizons: List of forecast horizons to visualize
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        fig, axes = plt.subplots(
            len(horizons), 1, figsize=(12, 4*len(horizons)))

        if len(horizons) == 1:
            axes = [axes]

        for i, horizon in enumerate(horizons):
            if horizon > predicted.shape[1]:
                continue

            ax = axes[i]

            # Get data for this horizon
            horizon_actual = actual[:, :horizon]
            horizon_predicted = predicted[:, :horizon]

            # Calculate mean across all sequences
            mean_actual = np.mean(horizon_actual, axis=0)
            mean_predicted = np.mean(horizon_predicted, axis=0)

            # Calculate confidence intervals (95%)
            std_actual = np.std(horizon_actual, axis=0)
            std_predicted = np.std(horizon_predicted, axis=0)
            ci_actual = 1.96 * std_actual / np.sqrt(horizon_actual.shape[0])
            ci_predicted = 1.96 * std_predicted / \
                np.sqrt(horizon_predicted.shape[0])

            hours = np.arange(horizon)

            # Plot means
            ax.plot(hours, mean_actual, 'b-', linewidth=2, label='Actual GHI')
            ax.plot(hours, mean_predicted, 'r--',
                    linewidth=2, label='Predicted GHI')

            # Plot confidence intervals
            ax.fill_between(hours, mean_actual - ci_actual, mean_actual + ci_actual,
                            color='blue', alpha=0.2)
            ax.fill_between(hours, mean_predicted - ci_predicted, mean_predicted + ci_predicted,
                            color='red', alpha=0.2)

            # Calculate metrics for this horizon
            mae = np.mean(np.abs(horizon_actual - horizon_predicted))
            rmse = np.sqrt(
                np.mean(np.square(horizon_actual - horizon_predicted)))

            # Add metrics to plot
            ax.text(0.02, 0.92, f'MAE: {mae:.4f}, RMSE: {rmse:.4f}',
                    transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

            # Customize plot
            ax.set_xlabel('Hours ahead')
            ax.set_ylabel('Global Horizontal Irradiance (kW/m²)')
            ax.set_title(f'{horizon}-hour Forecast Horizon')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def plot_error_distribution(self, actual, predicted, by_hour=True,
                                title='GHI Forecast Error Distribution',
                                save_path=None):
        """
        Plot distribution of forecast errors.

        Parameters:
        - actual: True GHI values
        - predicted: Predicted GHI values
        - by_hour: Whether to break down errors by hour of day
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        errors = predicted - actual

        if by_hour:
            # Reshape errors to (n_days, 24) if not already
            if errors.shape[1] % 24 != 0:
                # Truncate to multiple of 24
                truncate_len = (errors.shape[1] // 24) * 24
                errors = errors[:, :truncate_len]

            n_sequences = errors.shape[0]
            n_days = errors.shape[1] // 24
            errors_by_hour = errors.reshape(n_sequences * n_days, 24)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Overall error distribution
            sns.histplot(errors.flatten(), kde=True, ax=ax1)
            ax1.set_xlabel('Forecast Error (kW/m²)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Overall Error Distribution')

            # Mean absolute error by hour
            mae_by_hour = np.abs(errors_by_hour).mean(axis=0)
            hours = np.arange(24)

            ax2.bar(hours, mae_by_hour, color='skyblue')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Mean Absolute Error (kW/m²)')
            ax2.set_title('MAE by Hour of Day')
            ax2.set_xticks(hours)

            # Add daylight indicator (example - would be location specific)
            ax2.axvspan(6, 18, alpha=0.2, color='yellow')

        else:
            # Simple error distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(errors.flatten(), kde=True, ax=ax)
            ax.set_xlabel('Forecast Error (kW/m²)')
            ax.set_ylabel('Frequency')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def plot_attention_patterns(self, attention_weights, input_seq=None, output_seq=None,
                                title='Attention Patterns', save_path=None):
        """
        Visualize attention weights.

        Parameters:
        - attention_weights: Attention weights from the transformer
        - input_seq: Input sequence (optional)
        - output_seq: Output sequence (optional)
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot attention weights
        im = ax.imshow(attention_weights,
                       cmap=self.attention_cmap, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')

        # Set axis labels
        ax.set_xlabel('Input Sequence (Time Steps)')
        ax.set_ylabel('Output Sequence (Time Steps)')

        # Add title
        ax.set_title(title)

        # Add custom ticks if sequences are provided
        if input_seq is not None and isinstance(input_seq, pd.Series):
            # Use datetime indices if available
            xticks = np.arange(0, len(input_seq), max(1, len(input_seq)//10))
            xticklabels = [input_seq.index[i].strftime(
                '%m-%d %H:%M') for i in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45)

        if output_seq is not None and isinstance(output_seq, pd.Series):
            yticks = np.arange(0, len(output_seq), max(1, len(output_seq)//10))
            yticklabels = [output_seq.index[i].strftime(
                '%m-%d %H:%M') for i in yticks]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)

        plt.tight_layout()

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def plot_feature_importance(self, feature_importance, feature_names=None,
                                title='Feature Importance', save_path=None):
        """
        Visualize feature importance.

        Parameters:
        - feature_importance: Importance scores for features
        - feature_names: Names of features (optional)
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # If feature names not provided, use defaults
        if feature_names is None:
            feature_names = [
                f'Feature {i+1}' for i in range(len(feature_importance))]

        # Sort by importance
        sorted_idx = np.argsort(feature_importance)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = feature_importance[sorted_idx]

        # Create horizontal bar plot
        ax.barh(range(len(sorted_features)),
                sorted_importance, color='skyblue')
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)

        # Add values to bars
        for i, v in enumerate(sorted_importance):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def plot_training_history(self, history, metrics=None, title='Training History',
                              save_path=None):
        """
        Plot training metrics history.

        Parameters:
        - history: Dictionary containing training history
        - metrics: List of metrics to plot (defaults to all)
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        if metrics is None:
            # Group related metrics
            metric_groups = {
                'Loss': ['gen_loss', 'disc_loss', 'mse_loss', 'q_loss'],
                'Validation': ['val_mse', 'val_mae', 'val_rmse'],
                'Adversarial': ['adv_loss', 'disc_accuracy']
            }
        else:
            # Use provided metrics
            metric_groups = {'Metrics': metrics}

        n_groups = len(metric_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(12, 5*n_groups))

        if n_groups == 1:
            axes = [axes]

        for i, (group_name, group_metrics) in enumerate(metric_groups.items()):
            ax = axes[i]

            for metric in group_metrics:
                if metric in history and len(history[metric]) > 0:
                    ax.plot(history[metric], label=metric)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(f'{group_name} Metrics')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def plot_temporal_stability(self, predictions, timestamps=None, stability_metric='cv',
                                title='Temporal Stability Analysis', save_path=None):
        """
        Visualize temporal stability of predictions.

        Parameters:
        - predictions: Multiple prediction sequences
        - timestamps: Timestamp indices
        - stability_metric: Metric to use ('cv' for coefficient of variation, 'range', 'std')
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        if predictions.ndim != 3:
            raise ValueError(
                "Predictions should have shape (n_predictions, sequence_length, features)")

        n_predictions, seq_length, _ = predictions.shape

        # Calculate stability metric over the predictions dimension
        if stability_metric == 'cv':
            # Coefficient of variation (std / mean)
            means = np.mean(predictions, axis=0)
            stds = np.std(predictions, axis=0)
            # Avoid division by zero
            stability = np.divide(
                stds, means, out=np.zeros_like(stds), where=means != 0)
            metric_name = 'Coefficient of Variation'
        elif stability_metric == 'range':
            # Range of predictions
            max_vals = np.max(predictions, axis=0)
            min_vals = np.min(predictions, axis=0)
            stability = max_vals - min_vals
            metric_name = 'Range'
        elif stability_metric == 'std':
            # Standard deviation
            stability = np.std(predictions, axis=0)
            metric_name = 'Standard Deviation'
        else:
            raise ValueError(f"Unknown stability metric: {stability_metric}")

        stability = stability[:, 0]  # Take only the GHI dimension

        fig, ax = plt.subplots(figsize=(12, 6))

        # X-axis
        if timestamps is None:
            x = np.arange(seq_length)
            x_label = 'Time Step'
        else:
            x = timestamps
            x_label = 'Time'

        # Plot stability metric
        ax.plot(x, stability, 'b-', linewidth=2)

        # Add smoothed trend
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(stability, sigma=2)
        ax.plot(x, smoothed, 'r--', linewidth=2, label='Smoothed Trend')

        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        ax.set_title(title)

        # Add horizon divisions
        if seq_length >= 24:
            for i in range(6, seq_length, 6):
                ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

            # Add horizon labels
            ax.text(3, ax.get_ylim()[1]*0.9, 'Short-term', ha='center')
            if seq_length >= 12:
                ax.text(9, ax.get_ylim()[1]*0.9, 'Medium-term', ha='center')
            if seq_length >= 18:
                ax.text(18, ax.get_ylim()[1]*0.9, 'Long-term', ha='center')

        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def plot_multi_location_comparison(self, location_results, metric='rmse',
                                       title='GHI Forecast Performance by Location',
                                       save_path=None):
        """
        Compare forecast performance across locations.

        Parameters:
        - location_results: Dictionary with location names as keys and metrics as values
        - metric: Metric to compare
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        locations = list(location_results.keys())
        values = [location_results[loc][metric] for loc in locations]

        # Create bar chart
        bars = ax.bar(locations, values, color='skyblue')

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')

        # Customize plot
        ax.set_xlabel('Location')
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save if requested
        if save_path:
            if not save_path.endswith('.png') and not save_path.endswith('.jpg'):
                save_path += '.png'
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {full_path}")

        return fig

    def generate_comparison_report(self, actual_data, predicted_data, dates=None,
                                   location=None, output_file='ghi_forecast_report.html'):
        """
        Generate an HTML report comparing actual and predicted GHI values.

        Parameters:
        - actual_data: True GHI values
        - predicted_data: Predicted GHI values
        - dates: Datetime indices
        - location: Location name
        - output_file: HTML file name

        Returns:
        - Path to generated HTML file
        """
        # Calculate metrics
        mae = np.mean(np.abs(actual_data - predicted_data))
        mse = np.mean(np.square(actual_data - predicted_data))
        rmse = np.sqrt(mse)
        mask = actual_data != 0
        mape = np.mean(
            np.abs((actual_data[mask] - predicted_data[mask]) / actual_data[mask])) * 100

        # Create figures
        comparison_fig = self.plot_forecast_comparison(actual_data, predicted_data, dates=dates,
                                                       title='GHI Forecast Comparison')
        error_fig = self.plot_error_distribution(actual_data, predicted_data,
                                                 title='GHI Forecast Error Distribution')

        # Save figures for the report
        comparison_path = os.path.join(self.output_dir, 'comparison_chart.png')
        error_path = os.path.join(self.output_dir, 'error_distribution.png')

        comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
        error_fig.savefig(error_path, dpi=300, bbox_inches='tight')

        # Close figures to free memory
        plt.close(comparison_fig)
        plt.close(error_fig)

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GHI Forecast Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; border: 1px solid #ddd; padding: 10px; width: 20%; }}
                .chart {{ margin: 20px 0; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GHI Forecast Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                {f"<p>Location: {location}</p>" if location else ""}
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>MAE</h3>
                    <p>{mae:.4f}</p>
                </div>
                <div class="metric">
                    <h3>RMSE</h3>
                    <p>{rmse:.4f}</p>
                </div>
                <div class="metric">
                    <h3>MAPE</h3>
                    <p>{mape:.2f}%</p>
                </div>
            </div>
            
            <div class="chart">
                <h2>Forecast Comparison</h2>
                <img src="comparison_chart.png" alt="Forecast Comparison">
            </div>
            
            <div class="chart">
                <h2>Error Distribution</h2>
                <img src="error_distribution.png" alt="Error Distribution">
            </div>
        </body>
        </html>
        """

        # Write HTML file
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Generated report at {output_path}")
        return output_path
