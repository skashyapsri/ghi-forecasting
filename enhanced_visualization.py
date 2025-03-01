import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error


class EnhancedGHIVisualizer:
    """
    Advanced visualization tools for GHI forecasting thesis documentation.
    Builds on the existing GHIVisualizer with specialized thesis visualizations.
    """

    def __init__(self, output_dir='thesis_visualizations'):
        """Initialize with improved styling for publication-quality figures"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set publication-quality style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 20

        # Custom color palettes
        self.forecast_colors = {'actual': '#1f77b4', 'predicted': '#ff7f0e',
                                'lower_bound': '#d3d3d3', 'upper_bound': '#d3d3d3'}
        self.attention_cmap = 'viridis'
        self.error_cmap = sns.diverging_palette(10, 220, as_cmap=True)

    def plot_multihorizon_comparison(self, actual_data, predicted_data,
                                     dates=None, horizons=[1, 6, 12, 24],
                                     title="Multi-horizon GHI Forecast Performance",
                                     save_path=None):
        """
        Create a multi-panel plot showing performance at different forecast horizons.

        Parameters:
        - actual_data: Array of actual GHI values
        - predicted_data: Array of predicted GHI values
        - dates: Optional datetime index for x-axis
        - horizons: List of forecast horizons to visualize
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - Matplotlib figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(len(horizons), 2, width_ratios=[3, 1], figure=fig)

        metrics_table = []

        for i, horizon in enumerate(horizons):
            # Skip if horizon exceeds data length
            if horizon > predicted_data.shape[1]:
                continue

            # Get data for this horizon
            actual_horizon = actual_data[:, :horizon]
            predicted_horizon = predicted_data[:, :horizon]

            # Ensure shapes match by squeezing extra dimensions if necessary
            if actual_horizon.ndim > 2:
                actual_horizon = np.squeeze(actual_horizon)
            if predicted_horizon.ndim > 2:
                predicted_horizon = np.squeeze(predicted_horizon)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(
                actual_horizon.flatten(), predicted_horizon.flatten()))
            mae = mean_absolute_error(
                actual_horizon.flatten(), predicted_horizon.flatten())

            # Compute bias and variance components - ensure shapes match
            error_diff = predicted_horizon - actual_horizon
            bias = np.mean(error_diff)
            variance = np.var(error_diff)

            # Store metrics
            metrics_table.append([horizon, rmse, mae, bias, variance])

            # Time series plot
            ax1 = fig.add_subplot(gs[i, 0])

            # Use dates if provided
            if dates is not None and len(dates) >= horizon:
                x = dates[:horizon]
                x_label = 'Date'
            else:
                x = np.arange(horizon)
                x_label = 'Hours Ahead'

            # Plot mean and 90% confidence interval
            mean_actual = np.mean(actual_horizon, axis=0)
            mean_pred = np.mean(predicted_horizon, axis=0)

            p10_pred = np.percentile(predicted_horizon, 10, axis=0)
            p90_pred = np.percentile(predicted_horizon, 90, axis=0)

            ax1.plot(x, mean_actual, '-', color=self.forecast_colors['actual'],
                     label='Actual', linewidth=2)
            ax1.plot(x, mean_pred, '--', color=self.forecast_colors['predicted'],
                     label='Predicted', linewidth=2)
            ax1.fill_between(x, p10_pred, p90_pred, color=self.forecast_colors['lower_bound'],
                             alpha=0.3, label='80% Confidence Interval')

            ax1.set_xlabel(x_label)
            ax1.set_ylabel('GHI (kW/m²)')
            ax1.set_title(f'{horizon}-Hour Forecast')
            ax1.legend(loc='upper right')

            if isinstance(x[0], datetime):
                ax1.xaxis.set_major_formatter(
                    mdates.DateFormatter('%m-%d %H:%M'))
                plt.setp(ax1.xaxis.get_majorticklabels(),
                         rotation=45, ha='right')

            # Error distribution plot
            ax2 = fig.add_subplot(gs[i, 1])
            errors = (predicted_horizon - actual_horizon).flatten()

            sns.histplot(errors, kde=True, ax=ax2,
                         color=self.forecast_colors['predicted'])
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Forecast Error')
            ax2.set_ylabel('Frequency')

            # Add metrics annotation
            ax2.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nBias: {bias:.4f}',
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(title, fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save if requested
        if save_path:
            plt.savefig(f"{self.output_dir}/{save_path}",
                        dpi=300, bbox_inches='tight')
            print(f"Saved figure to {self.output_dir}/{save_path}")

        return fig, pd.DataFrame(metrics_table,
                                 columns=['Horizon', 'RMSE', 'MAE', 'Bias', 'Variance'])

    def plot_attention_heatmap(self, attention_weights, sequence_length=168,
                               title="Sparse Attention Patterns Analysis", save_path=None):
        """
        Visualize attention weights from the AST model.

        Parameters:
        - attention_weights: 2D array of attention weights
        - sequence_length: Length of the input sequence
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - Matplotlib figure
        """
        # Ensure attention_weights is 2D
        if attention_weights.ndim > 2:
            print(
                f"Warning: Attention weights have shape {attention_weights.shape}, expected 2D array. Taking first batch item.")
            attention_weights = attention_weights[0]  # Take first batch item

        fig = plt.figure(figsize=(14, 10))
        ax = plt.gca()

        # Create heatmap
        im = ax.imshow(attention_weights, aspect='auto',
                       cmap=self.attention_cmap, interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')

        # Add time demarcations (day boundaries)
        for day in range(sequence_length // 24):
            ax.axvline(x=day*24, color='white', linestyle='--', alpha=0.5)
            ax.axhline(y=day*24, color='white', linestyle='--', alpha=0.5)

        # Annotate sparsity pattern
        non_zero_attn = np.count_nonzero(
            attention_weights) / attention_weights.size
        ax.text(0.02, 0.02, f'Attention Density: {non_zero_attn:.2%}',
                transform=ax.transAxes, color='white', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.6))

        # Calculate and visualize key attention patterns
        # Top 5 attended positions
        top_k_indices = np.argsort(-attention_weights, axis=1)[:, :5]

        # First day outputs
        for i in range(min(24, attention_weights.shape[0])):
            # Top 3 for clarity
            for j in range(min(3, top_k_indices.shape[1])):
                try:
                    # Get scalar index value
                    idx = top_k_indices[i, j]
                    if isinstance(idx, (np.ndarray, list)):
                        # If it's still an array, take the first element
                        idx = idx.item() if hasattr(idx, 'item') else idx[0]

                    # Highlight top attention connections
                    ax.plot([idx, i], [i, i], 'o-', color='yellow', alpha=0.7,
                            markersize=4, linewidth=1)
                except (TypeError, ValueError, IndexError) as e:
                    print(
                        f"Warning: Could not plot connection for i={i}, j={j}: {e}")
                    continue

        # Set labels and title
        ax.set_xlabel('Input Sequence (Time Steps)')
        ax.set_ylabel('Output Sequence (Time Steps)')
        ax.set_title(title)

        # Add hour markers every 24 steps
        x_ticks = np.arange(0, sequence_length, 24)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'Day {i+1}' for i in range(len(x_ticks))])

        # Every 6 hours of output
        y_ticks = np.arange(0, attention_weights.shape[0], 6)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'H+{i}' for i in y_ticks])

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(f"{self.output_dir}/{save_path}",
                        dpi=300, bbox_inches='tight')
            print(f"Saved figure to {self.output_dir}/{save_path}")

        return fig

    def plot_error_analysis_dashboard(self, actual, predicted,
                                      dates=None, features=None, weather_data=None,
                                      title="Comprehensive Error Analysis Dashboard",
                                      save_path=None):
        """
        Create a comprehensive dashboard for error analysis.

        Parameters:
        - actual: True GHI values
        - predicted: Predicted GHI values
        - dates: Optional datetime index
        - features: Feature importance data
        - weather_data: Additional weather variables for correlation analysis
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - Matplotlib figure
        """
        fig = plt.figure(figsize=(18, 15))
        gs = GridSpec(3, 3, figure=fig)

        # Ensure shapes match by squeezing extra dimensions if necessary
        if actual.ndim > 2:
            actual = np.squeeze(actual)
        if predicted.ndim > 2:
            predicted = np.squeeze(predicted)

        # Calculate errors
        errors = predicted - actual
        # Add epsilon to avoid division by zero
        mape_values = np.abs(errors) / (np.abs(actual) + 1e-10) * 100

        # 1. Actual vs Predicted scatter plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(actual.flatten(), predicted.flatten(), alpha=0.5,
                    edgecolor='k', linewidth=0.5)

        # Add perfect prediction line
        max_val = max(np.max(actual), np.max(predicted))
        ax1.plot([0, max_val], [0, max_val], 'r--')

        ax1.set_xlabel('Actual GHI')
        ax1.set_ylabel('Predicted GHI')
        ax1.set_title('Predicted vs Actual GHI')

        # Add R² value
        corr_coef = np.corrcoef(actual.flatten(), predicted.flatten())[0, 1]
        r_squared = corr_coef**2
        ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        # 2. Error histogram (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(errors.flatten(), kde=True, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Forecast Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')

        # Add descriptive statistics
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        ax2.text(0.05, 0.95, f'Mean: {error_mean:.4f}\nStd Dev: {error_std:.4f}',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))

        # 3. MAPE by forecast horizon (top right)
        ax3 = fig.add_subplot(gs[0, 2])

        # Calculate MAPE by horizon (averaging across samples)
        horizon_mape = np.nanmean(mape_values, axis=0)
        x_horizon = np.arange(1, len(horizon_mape) + 1)

        ax3.plot(x_horizon, horizon_mape, 'o-', linewidth=2)
        ax3.set_xlabel('Forecast Horizon (Hours)')
        ax3.set_ylabel('MAPE (%)')
        ax3.set_title('Error by Forecast Horizon')
        ax3.grid(True, alpha=0.3)

        # 4. Time series of predictions with error bands (middle row)
        ax4 = fig.add_subplot(gs[1, :])

        # Use sample index or dates if provided
        if dates is not None:
            x = dates
        else:
            x = np.arange(actual.shape[1])

        # Calculate mean and confidence intervals
        mean_actual = np.mean(actual, axis=0)
        mean_pred = np.mean(predicted, axis=0)

        pred_std = np.std(predicted, axis=0)
        lower_bound = mean_pred - 1.96 * pred_std
        upper_bound = mean_pred + 1.96 * pred_std

        ax4.plot(x, mean_actual, '-', label='Actual', linewidth=2,
                 color=self.forecast_colors['actual'])
        ax4.plot(x, mean_pred, '--', label='Predicted', linewidth=2,
                 color=self.forecast_colors['predicted'])
        ax4.fill_between(x, lower_bound, upper_bound, alpha=0.3,
                         color=self.forecast_colors['lower_bound'],
                         label='95% Confidence Interval')

        ax4.set_xlabel('Time')
        ax4.set_ylabel('GHI (kW/m²)')
        ax4.set_title('GHI Forecast with Uncertainty')
        ax4.legend(loc='upper right')

        if isinstance(x[0], datetime):
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Error heatmap by hour and day (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])

        # Reshape errors for day/hour analysis - for example with 7 days of hourly data
        days = min(7, actual.shape[1] // 24)
        hours = 24

        if actual.shape[1] >= days * hours:
            error_by_hour_day = np.mean(
                np.abs(errors[:, :days*hours]), axis=0).reshape(days, hours).T

            sns.heatmap(error_by_hour_day, cmap=self.error_cmap, annot=False, fmt=".2f",
                        ax=ax5, cbar_kws={'label': 'MAE'})

            ax5.set_xlabel('Day')
            ax5.set_ylabel('Hour of Day')
            ax5.set_title('Error by Hour and Day')
            ax5.set_yticks(np.arange(0, 24, 3))
            ax5.set_yticklabels(np.arange(0, 24, 3))
            ax5.set_xticks(np.arange(days))
            ax5.set_xticklabels([f'Day {i+1}' for i in range(days)])
        else:
            ax5.text(0.5, 0.5, 'Insufficient data for day/hour analysis',
                     ha='center', va='center', transform=ax5.transAxes)

        # 6. Feature importance (if provided) or weather correlation (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])

        if features is not None and len(features) > 0:
            # Sort features by importance
            sorted_idx = np.argsort(features)
            feature_names = [f'Feature {i}' for i in range(len(features))]

            # Plot horizontal bar chart
            y_pos = np.arange(len(features))
            ax6.barh(y_pos, features[sorted_idx], align='center')
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax6.invert_yaxis()  # Highest values at the top
            ax6.set_xlabel('Importance')
            ax6.set_title('Feature Importance')
        elif weather_data is not None:
            # Compute correlation between errors and weather variables
            correlations = []
            weather_vars = []

            for var, values in weather_data.items():
                if len(values) == len(errors.flatten()):
                    corr = np.corrcoef(np.abs(errors.flatten()), values)[0, 1]
                    correlations.append(corr)
                    weather_vars.append(var)

            if correlations:
                # Sort correlations
                sorted_idx = np.argsort(np.abs(correlations))

                # Plot horizontal bar chart
                y_pos = np.arange(len(correlations))
                ax6.barh(y_pos, [correlations[i]
                                 for i in sorted_idx], align='center')
                ax6.set_yticks(y_pos)
                ax6.set_yticklabels([weather_vars[i] for i in sorted_idx])
                ax6.invert_yaxis()  # Highest values at the top
                ax6.set_xlabel('Correlation with |Error|')
                ax6.set_title('Weather Variable Correlation')
                ax6.axvline(x=0, color='k', linestyle='--', alpha=0.7)
            else:
                ax6.text(0.5, 0.5, 'No weather correlation data available',
                         ha='center', va='center', transform=ax6.transAxes)
        else:
            ax6.text(0.5, 0.5, 'No feature importance data available',
                     ha='center', va='center', transform=ax6.transAxes)

        # 7. Forecast stability visualization (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])

        # Calculate coefficient of variation by forecast hour
        cv_by_hour = np.std(predicted, axis=0) / \
            (np.mean(np.abs(actual), axis=0) + 1e-10)

        ax7.plot(np.arange(1, len(cv_by_hour) + 1),
                 cv_by_hour, 'o-', linewidth=2)
        ax7.set_xlabel('Forecast Horizon (Hours)')
        ax7.set_ylabel('Coefficient of Variation')
        ax7.set_title('Forecast Stability by Horizon')
        ax7.grid(True, alpha=0.3)

        # Add HSI values if applicable
        horizons = {'short': (0, min(6, len(cv_by_hour))),
                    'medium': (min(6, len(cv_by_hour)), min(12, len(cv_by_hour))),
                    'long': (min(12, len(cv_by_hour)), len(cv_by_hour))}

        hsi_values = {}
        for name, (start, end) in horizons.items():
            if start < end:
                slice_cv = cv_by_hour[start:end]
                slice_mean = np.mean(np.abs(actual), axis=0)[start:end]
                hsi = 1 - np.mean(slice_cv)
                hsi_values[name] = hsi

        if hsi_values:
            hsi_text = '\n'.join(
                [f'HSI_{k}: {v:.4f}' for k, v in hsi_values.items()])
            ax7.text(0.05, 0.95, hsi_text, transform=ax7.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(title, fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save if requested
        if save_path:
            plt.savefig(f"{self.output_dir}/{save_path}",
                        dpi=300, bbox_inches='tight')
            print(f"Saved figure to {self.output_dir}/{save_path}")

        return fig

    def plot_model_architecture(self, save_path=None):
        """
        Create a visual representation of the AST model architecture.

        Parameters:
        - save_path: Path to save the figure

        Returns:
        - Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create an empty plot
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        # Define colors
        colors = {
            'encoder': '#66c2a5',
            'decoder': '#fc8d62',
            'attention': '#8da0cb',
            'discriminator': '#e78ac3',
            'text': '#000000',
            'arrow': '#666666',
            'background': '#f8f8f8'
        }

        # Draw model components with better formatting
        self._draw_box(
            ax, 10, 70, 25, 20, colors['encoder'], 'Encoder Stack\n(3 Layers)', colors['text'])
        self._draw_box(
            ax, 10, 40, 25, 20, colors['decoder'], 'Decoder Stack\n(3 Layers)', colors['text'])
        self._draw_box(
            ax, 65, 55, 25, 20, colors['discriminator'], 'Discriminator\nNetwork', colors['text'])

        # Draw attention mechanism (central to the AST architecture)
        self._draw_box(ax, 40, 55, 20, 15, colors['attention'],
                       'α-entmax\nSparse Attention\n(α=1.5)', colors['text'])

        # Label inputs and outputs
        ax.text(5, 95, 'Adversarial Sparse Transformer Architecture',
                fontsize=16, fontweight='bold')
        ax.text(10, 95, 'Historical Data\n(168 hrs, 9 features)',
                fontsize=10, ha='center')
        ax.text(10, 25, 'Future Covariates\n(24 hrs, 8 features)',
                fontsize=10, ha='center')
        ax.text(87, 40, 'GHI Predictions\n(24 hrs)', fontsize=10, ha='center')

        # Connect components with arrows
        # Input to Encoder
        self._draw_arrow(ax, 10, 90, 10, 75, colors['arrow'])
        # Future covariates to Decoder
        self._draw_arrow(ax, 10, 35, 10, 30, colors['arrow'])
        # Encoder to Attention
        self._draw_arrow(ax, 22.5, 70, 22.5, 60, colors['arrow'])
        # Attention to Decoder
        self._draw_arrow(ax, 35, 55, 40, 55, colors['arrow'])
        # Decoder to Discriminator
        self._draw_arrow(ax, 35, 40, 65, 45, colors['arrow'])
        # Decoder to Output
        self._draw_arrow(ax, 35, 40, 80, 40, colors['arrow'])

        # Add explanatory notes
        notes = [
            "Encoder: Processes historical data through self-attention",
            "Decoder: Generates forecasts using encoder context",
            "α-entmax: Creates sparse attention patterns",
            "Discriminator: Evaluates forecast sequence realism",
            "Training: Adversarial objective + MSE loss"
        ]

        for i, note in enumerate(notes):
            ax.text(5, 15 - i*2.5, f"• {note}", fontsize=9, ha='left')

        # Add title
        plt.suptitle(
            'Adversarial Sparse Transformer for GHI Forecasting', fontsize=20)

        # Save if requested
        if save_path:
            plt.savefig(f"{self.output_dir}/{save_path}",
                        dpi=300, bbox_inches='tight')
            print(f"Saved figure to {self.output_dir}/{save_path}")

        return fig

    def _draw_box(self, ax, x, y, width, height, facecolor, text, textcolor):
        """Helper method to draw a box with text"""
        rect = plt.Rectangle((x, y), width, height, facecolor=facecolor, alpha=0.7,
                             edgecolor='black', linewidth=1, zorder=1)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                color=textcolor, fontweight='bold', zorder=2)

    def _draw_arrow(self, ax, x1, y1, x2, y2, color):
        """Helper method to draw an arrow"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, linewidth=1.5))

    def plot_comparative_analysis(self, ast_results, baseline_results,
                                  metrics=['MAE', 'RMSE', 'MAPE'],
                                  title="Model Comparative Analysis",
                                  save_path=None):
        """
        Create a comparative visualization of AST performance against baselines.

        Parameters:
        - ast_results: Dictionary with AST performance metrics
        - baseline_results: Dictionary with baseline model performance metrics
        - metrics: List of metrics for comparison
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - Matplotlib figure
        """
        # Get all model names
        models = ['AST'] + list(baseline_results.keys())

        # Create figure with subplots for each metric
        fig, axs = plt.subplots(1, len(metrics), figsize=(15, 6))

        # Handle case with only one metric
        if len(metrics) == 1:
            axs = [axs]

        for i, metric in enumerate(metrics):
            # Extract values for this metric
            values = [ast_results.get(metric.lower(), 0)]

            for model in baseline_results:
                values.append(baseline_results[model].get(metric.lower(), 0))

            # Create bar chart
            axs[i].bar(models, values, alpha=0.7)

            # Add value labels on top of bars
            for j, v in enumerate(values):
                axs[i].text(j, v + max(values)*0.02, f'{v:.4f}',
                            ha='center', va='bottom', fontsize=9)

            # Add metric-specific formatting
            axs[i].set_title(metric)
            axs[i].set_ylabel(metric)
            axs[i].grid(axis='y', alpha=0.3)

            # Highlight best model
            # Assuming lower is better for all metrics
            best_idx = np.argmin(values)
            axs[i].get_children()[best_idx].set_facecolor('green')

            # Rotate x-tick labels for readability if needed
            if len(models) > 3:
                plt.setp(axs[i].xaxis.get_majorticklabels(),
                         rotation=45, ha='right')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save if requested
        if save_path:
            plt.savefig(f"{self.output_dir}/{save_path}",
                        dpi=300, bbox_inches='tight')
            print(f"Saved figure to {self.output_dir}/{save_path}")

        return fig

    def plot_ablation_study(self, ablation_results,
                            title="Ablation Study Results",
                            save_path=None):
        """
        Visualize the results of an ablation study.

        Parameters:
        - ablation_results: Dictionary with component names as keys and result dictionaries as values
        - title: Plot title
        - save_path: Path to save the figure

        Returns:
        - Matplotlib figure
        """
        components = list(ablation_results.keys())
        metrics = ['rmse', 'mae', 'mape']  # Standard metrics

        # Extract RMSE values
        rmse_values = [ablation_results[comp]['rmse'] for comp in components]
        mae_values = [ablation_results[comp]['mae'] for comp in components]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Create bar chart for RMSE
        bars1 = ax1.bar(components, rmse_values, alpha=0.7)
        ax1.set_ylabel('RMSE')
        ax1.set_title('Effect on RMSE')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        # Create bar chart for MAE
        bars2 = ax2.bar(components, mae_values, alpha=0.7)
        ax2.set_ylabel('MAE')
        ax2.set_title('Effect on MAE')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        # Highlight best configuration
        best_rmse_idx = np.argmin(rmse_values)
        best_mae_idx = np.argmin(mae_values)

        bars1[best_rmse_idx].set_facecolor('green')
        bars2[best_mae_idx].set_facecolor('green')

        # Rotate x-tick labels if needed
        if len(components) > 4:
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save if requested
        if save_path:
            plt.savefig(f"{self.output_dir}/{save_path}",
                        dpi=300, bbox_inches='tight')
            print(f"Saved figure to {self.output_dir}/{save_path}")

        return fig

    def generate_thesis_visualizations(self, forecaster, datasets, test_metrics):
        """
        Generate a comprehensive set of visualizations for thesis documentation.

        Parameters:
        - forecaster: Trained GHI Forecaster model
        - datasets: Dictionary with train, validation, and test datasets
        - test_metrics: Dictionary with test evaluation metrics

        Returns:
        - Dictionary of created figures
        """
        figures = {}

        # 1. Get sample test data
        sample_batches = next(iter(datasets['test'].take(1)))
        sample_inputs = sample_batches[0]
        sample_targets = sample_batches[1].numpy()

        # 2. Get model predictions
        sample_predictions = forecaster.predict({
            'historical': sample_inputs['historical'],
            'future_covariates': sample_inputs['future_covariates']
        })

        # 3. Create multi-horizon comparison
        figures['multihorizon'] = self.plot_multihorizon_comparison(
            sample_targets,
            sample_predictions,
            horizons=[6, 12, 24],
            title="GHI Forecast Performance at Different Horizons",
            save_path="multihorizon_comparison.png"
        )[0]

        # 4. Create error analysis dashboard
        figures['error_dashboard'] = self.plot_error_analysis_dashboard(
            sample_targets,
            sample_predictions,
            title="GHI Forecast Error Analysis Dashboard",
            save_path="error_dashboard.png"
        )

        # 5. Create model architecture diagram
        figures['architecture'] = self.plot_model_architecture(
            save_path="ast_architecture.png"
        )

        # 6. Create comparison with baselines (placeholder)
        baseline_results = {
            'ARIMA': {'mae': 0.56, 'rmse': 0.73, 'mape': 280.0},
            'LSTM': {'mae': 0.41, 'rmse': 0.56, 'mape': 240.0},
            'CNN-LSTM': {'mae': 0.38, 'rmse': 0.52, 'mape': 230.0}
        }

        figures['comparison'] = self.plot_comparative_analysis(
            test_metrics,  # Your AST results
            baseline_results,
            title="AST Performance Compared to Baseline Models",
            save_path="model_comparison.png"
        )

        # 7. Create results dashboard
        self.create_results_dashboard(
            forecaster,
            datasets['test'],
            test_metrics,
            save_path="results_dashboard.png"
        )

        print(f"All thesis visualizations created in {self.output_dir}/")
        return figures

    def create_results_dashboard(self, forecaster, test_dataset, test_metrics, save_path="results_dashboard.png"):
        """
        Create a comprehensive dashboard of all results.

        Parameters:
        - forecaster: Trained GHI Forecaster model
        - test_dataset: Test dataset
        - test_metrics: Dictionary with test evaluation metrics
        - save_path: Path to save the figure

        Returns:
        - Matplotlib figure
        """
        # Get multiple test batches
        all_targets = []
        all_predictions = []

        # Process multiple batches for better representation
        for i, (inputs, targets) in enumerate(test_dataset.take(5)):
            predictions = forecaster.predict({
                'historical': inputs['historical'],
                'future_covariates': inputs['future_covariates']
            })
            all_targets.append(targets.numpy())
            all_predictions.append(predictions)

            if i >= 4:  # Limit to 5 batches
                break

        # Concatenate results
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Create visualization
        # Calculate feature importance if available
        # This is just an example with placeholder values
        feature_importance = np.array(
            [0.35, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02])

        # Create dashboard with all metrics
        fig = self.plot_error_analysis_dashboard(
            all_targets,
            all_predictions,
            features=feature_importance,
            title=f"AST Model Performance: MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}",
            save_path=save_path
        )

        print(f"Results dashboard created at {self.output_dir}/{save_path}")
        return fig

    def extract_attention_weights(self, forecaster, inputs):
        """
        Extract attention weights from the model for visualization.
        Note: This requires modifying the model to store attention weights.

        Parameters:
        - forecaster: AST forecaster model
        - inputs: Input data to the model

        Returns:
        - Attention weights arrays
        """
        # This is a placeholder function - in a real implementation,
        # you would need to modify your model to store attention weights
        print("Note: To extract real attention weights, you need to modify your model code")
        print("to store attention weights during the forward pass")

        # Mock attention weights for example purposes
        seq_len = inputs['historical'].shape[1]
        out_len = inputs['future_covariates'].shape[1]

        # Generate mock attention weights for visualization
        attention_weights = np.zeros((out_len, seq_len))

        # Add some patterns - in a real implementation, these would be actual attention weights
        # Diagonal pattern (attending to matching timepoints)
        for i in range(min(out_len, seq_len)):
            attention_weights[i, i:i +
                              24] = np.random.exponential(0.5, size=min(24, seq_len-i))

        # Normalize rows to sum to 1
        for i in range(out_len):
            row_sum = attention_weights[i, :].sum()
            if row_sum > 0:
                attention_weights[i, :] = attention_weights[i, :] / row_sum

        return attention_weights
