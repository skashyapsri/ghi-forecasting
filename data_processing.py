import datetime
import os
import logging
import numpy as np
import pandas as pd
import requests
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NASAPowerDataProcessor:
    """
    Process NASA POWER data for GHI forecasting.

    This class handles data acquisition from NASA POWER API,
    preprocessing, feature engineering, and sequence formation.
    """

    def __init__(self,
                 lookback_history=168,  # One week of hourly data
                 estimate_length=24,    # One day forecast
                 params=None,
                 locations=None):
        """
        Initialize the data processor.

        Parameters:
        - lookback_history: Number of historical hours to use as input
        - estimate_length: Number of future hours to predict
        - params: List of NASA POWER parameters to retrieve
        - locations: List of locations (lat, lon) to retrieve data
        """
        self.lookback_history = lookback_history
        self.estimate_length = estimate_length

        # Default parameters from thesis requirements
        self.params = params or [
            'ALLSKY_SFC_SW_DWN',  # Target variable (GHI)
            'ALLSKY_TOA_SW_DWN',
            'ALLSKY_KT',
            'T2M',
            'RH2M',
            'PS',
            'WS2M',
            'WD2M',
            'CLOUD_AMT'
        ]

        # Default to major Indian cities as per thesis
        self.locations = locations or [
            {'name': 'Delhi', 'lat': 28.6139, 'lon': 77.2090},
            {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
            {'name': 'Chennai', 'lat': 13.0827, 'lon': 80.2707},
            {'name': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639},
            {'name': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946}
        ]

        self.logger = logging.getLogger('NASAPowerDataProcessor')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Feature scalers
        self.feature_scalers = {}

    def fetch_data(self, start_date, end_date, location, save_path=None):
        """
        Fetch data from NASA POWER API for a specific location and time range.
        Checks for locally stored data first before making API requests.

        Parameters:
        - start_date: Start date in 'YYYYMMDD' format
        - end_date: End date in 'YYYYMMDD' format
        - location: Dictionary with 'lat' and 'lon' keys
        - save_path: Optional path to save raw data

        Returns:
        - pandas DataFrame with hourly data
        """
        # Define paths for JSON and CSV data
        json_path = save_path.replace('.csv', '.json') if save_path else None

        # Check if data already exists locally
        if json_path and os.path.exists(json_path):
            self.logger.info(f"Loading existing data from {json_path}")
            try:
                # Load and process the existing JSON file
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Extract the parameter data from the JSON
                parameters_dict = data['properties']['parameter']

                # Get the first parameter to determine date/times
                first_param = next(iter(parameters_dict))
                date_keys = list(parameters_dict[first_param].keys())

                # Parse date keys to create datetime objects
                datetimes = []
                for date_key in date_keys:
                    try:
                        # Format is expected to be YYYYMMDDHH
                        dt = datetime.datetime.strptime(date_key, "%Y%m%d%H")
                        datetimes.append(dt)
                    except ValueError:
                        self.logger.warning(
                            f"Could not parse date key: {date_key}")

                # Create a DataFrame with the parsed datetimes as index
                df = pd.DataFrame(index=datetimes)

                # Fill the DataFrame with parameter values
                for param in self.params:
                    if param in parameters_dict:
                        param_values = []
                        for date_key in date_keys:
                            if date_key in parameters_dict[param]:
                                param_values.append(
                                    parameters_dict[param][date_key])
                            else:
                                param_values.append(np.nan)
                        df[param] = param_values
                    else:
                        self.logger.warning(
                            f"Parameter {param} not found in stored JSON")
                        df[param] = np.nan

                # Sort by datetime index
                df.sort_index(inplace=True)

                # Convert timestamps to pandas datetimes if they aren't already
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.DatetimeIndex(df.index)

                # Save processed data as CSV if requested
                if save_path:
                    df.to_csv(save_path)
                    self.logger.info(f"Processed data saved to {save_path}")

                self.logger.info(
                    f"Successfully loaded {len(df)} data points from local file")
                return df

            except Exception as e:
                self.logger.warning(
                    f"Error loading local data: {str(e)}. Will attempt API fetch.")
                # Continue to API fetch if there's an error loading local data

        # If we don't have local data or couldn't load it, fetch from API
        self.logger.info(
            f"Fetching data for {location['name']} from {start_date} to {end_date}")

        base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

        params = {
            'start': start_date,
            'end': end_date,
            'latitude': location['lat'],
            'longitude': location['lon'],
            'community': 'RE',
            'parameters': ','.join(self.params),
            'format': 'JSON',
            'header': 'true'
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            # Save raw JSON for future use
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path.replace('.csv', '.json'), 'w') as f:
                    json.dump(response.json(), f)
                self.logger.info(
                    f"Raw JSON saved to {save_path.replace('.csv', '.json')}")

            data = response.json()

            # Extract the parameter data from the API response
            parameters_dict = data['properties']['parameter']

            # Get the first parameter to determine date/times
            first_param = next(iter(parameters_dict))
            date_keys = list(parameters_dict[first_param].keys())

            # Parse date keys to create datetime objects
            datetimes = []
            for date_key in date_keys:
                try:
                    # Format is expected to be YYYYMMDDHH
                    dt = datetime.datetime.strptime(date_key, "%Y%m%d%H")
                    datetimes.append(dt)
                except ValueError:
                    self.logger.warning(
                        f"Could not parse date key: {date_key}")

            # Create a DataFrame with the parsed datetimes as index
            df = pd.DataFrame(index=datetimes)

            # Fill the DataFrame with parameter values
            for param in self.params:
                if param in parameters_dict:
                    param_values = []
                    for date_key in date_keys:
                        if date_key in parameters_dict[param]:
                            param_values.append(
                                parameters_dict[param][date_key])
                        else:
                            param_values.append(np.nan)
                    df[param] = param_values
                else:
                    self.logger.warning(
                        f"Parameter {param} not found in API response")
                    df[param] = np.nan

            # Sort by datetime index
            df.sort_index(inplace=True)

            # Save processed data
            if save_path:
                df.to_csv(save_path)
                self.logger.info(f"Processed data saved to {save_path}")

            self.logger.info(f"Successfully fetched {len(df)} data points")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")

            # For debugging, try to print more information about the response
            try:
                if 'response' in locals():
                    self.logger.info(
                        f"Response status code: {response.status_code}")
                    self.logger.info(
                        f"Response content snippet: {response.text[:1000]}...")
            except:
                pass

            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=self.params)
    # Add this to your data_processing.py in the preprocess_data method

    def preprocess_data(self, df, fill_method='interpolate'):
        """
        Preprocess raw data by handling missing values and normalizing features.

        Parameters:
        - df: Input DataFrame with raw data
        - fill_method: Method for filling missing values ('interpolate' or 'forward')

        Returns:
        - Preprocessed DataFrame
        """
        self.logger.info("Preprocessing data...")

        # Check if dataframe is empty
        if df.empty:
            self.logger.warning("Empty dataframe provided for preprocessing!")
            return df

        # Check for missing values
        missing = df.isna().sum()
        if missing.sum() > 0:
            self.logger.info(f"Found missing values: {missing[missing > 0]}")

            # IMPORTANT FIX: Handle completely missing ALLSKY_TOA_SW_DWN column
            if 'ALLSKY_TOA_SW_DWN' in df.columns and missing['ALLSKY_TOA_SW_DWN'] == len(df):
                self.logger.info(
                    "Generating synthetic ALLSKY_TOA_SW_DWN values")

                # Create synthetic ALLSKY_TOA_SW_DWN from ALLSKY_SFC_SW_DWN
                if 'ALLSKY_SFC_SW_DWN' in df.columns:
                    # TOA is typically 30-40% higher than surface radiation
                    df['ALLSKY_TOA_SW_DWN'] = df['ALLSKY_SFC_SW_DWN'] * 1.35

                    # Add day/night pattern
                    hour_of_day = df.index.hour
                    # Lower multiplier at night (roughly 0 during night hours)
                    night_mask = (hour_of_day < 6) | (hour_of_day > 18)
                    df.loc[night_mask, 'ALLSKY_TOA_SW_DWN'] = df.loc[night_mask,
                                                                     'ALLSKY_SFC_SW_DWN'] * 0.1

                    self.logger.info(
                        "Created synthetic ALLSKY_TOA_SW_DWN based on ALLSKY_SFC_SW_DWN")
                else:
                    # If no reference radiation data, use a simple day/night pattern
                    hour_of_day = df.index.hour
                    day_of_year = df.index.dayofyear

                    # Simple model: peak at noon, seasonal variation
                    hour_factor = np.sin(
                        np.pi * (hour_of_day - 6) / 12) * (hour_of_day >= 6) * (hour_of_day <= 18)
                    season_factor = 0.7 + 0.3 * \
                        np.sin(2 * np.pi * (day_of_year - 80) /
                               365)  # Peak in summer

                    # Generate synthetic values (scale 0-1, multiply by typical value)
                    df['ALLSKY_TOA_SW_DWN'] = hour_factor * \
                        season_factor * 1.2  # 1.2 kW/m²/day is typical peak
                    self.logger.info(
                        "Created synthetic ALLSKY_TOA_SW_DWN using time-based model")

            # Proceed with normal filling for other missing values
            if fill_method == 'interpolate':
                # For short gaps, use interpolation
                df = df.interpolate(method='linear', limit=3)

                # For remaining gaps, use forward fill
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')
            elif fill_method == 'forward':
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')

            # Check if we still have missing values
            still_missing = df.isna().sum()
            if still_missing.sum() > 0:
                self.logger.warning(
                    f"Still have missing values after filling: {still_missing[still_missing > 0]}")

                # SAFETY: Replace any remaining NaN with zeros
                df = df.fillna(0)
                self.logger.info("Replaced remaining NaN values with zeros")

        # Handle outliers - replace values outside 3 std with boundary values
        for col in df.columns:
            mean, std = df[col].mean(), df[col].std()

            # Skip columns with all zeros or NaN
            if std == 0 or np.isnan(std):
                self.logger.info(
                    f"Skipping outlier detection for {col} (no variation)")
                continue

            lower_bound, upper_bound = mean - 3*std, mean + 3*std
            outliers = ((df[col] < lower_bound) |
                        (df[col] > upper_bound)).sum()
            if outliers > 0:
                self.logger.info(f"Found {outliers} outliers in {col}")
                df[col] = df[col].clip(lower_bound, upper_bound)

        return df

    def engineer_features(self, df):
        """
        Create derived features from the raw data.

        Parameters:
        - df: Preprocessed DataFrame

        Returns:
        - DataFrame with additional engineered features
        """
        self.logger.info("Engineering features...")

        # Check if dataframe is empty
        if df.empty:
            self.logger.warning(
                "Empty dataframe provided for feature engineering!")
            return df

        # Create a copy to avoid modifying the original
        enhanced_df = df.copy()

        # 1. Time features
        # Hour of day (sinusoidal encoding)
        hour = enhanced_df.index.hour
        enhanced_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        enhanced_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Day of week (sinusoidal encoding)
        day_of_week = enhanced_df.index.dayofweek
        enhanced_df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        enhanced_df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

        # Day of year (sinusoidal encoding)
        day_of_year = enhanced_df.index.dayofyear
        enhanced_df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        enhanced_df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Month (sinusoidal encoding)
        month = enhanced_df.index.month
        enhanced_df['month_sin'] = np.sin(2 * np.pi * month / 12)
        enhanced_df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # 2. Clear sky index (ratio of GHI to extraterrestrial radiation)
        if 'ALLSKY_KT' not in enhanced_df.columns and 'ALLSKY_SFC_SW_DWN' in enhanced_df.columns and 'ALLSKY_TOA_SW_DWN' in enhanced_df.columns:
            enhanced_df['clear_sky_index'] = enhanced_df['ALLSKY_SFC_SW_DWN'] / \
                enhanced_df['ALLSKY_TOA_SW_DWN'].replace(0, np.nan)
            enhanced_df['clear_sky_index'] = enhanced_df['clear_sky_index'].fillna(
                0)  # Handle division by zero

        # 3. Temperature-humidity interaction
        if 'T2M' in enhanced_df.columns and 'RH2M' in enhanced_df.columns:
            enhanced_df['temp_humidity'] = enhanced_df['T2M'] * \
                enhanced_df['RH2M'] / 100.0

        # 4. Moving statistics for key variables
        for col in ['ALLSKY_SFC_SW_DWN', 'CLOUD_AMT', 'T2M']:
            if col in enhanced_df.columns:
                # 24h rolling mean (previous day average)
                enhanced_df[f'{col}_24h_mean'] = enhanced_df[col].rolling(
                    window=24, min_periods=1).mean()

                # 24h rolling std (variability)
                enhanced_df[f'{col}_24h_std'] = enhanced_df[col].rolling(
                    window=24, min_periods=1).std().fillna(0)

        # 5. Lag features for key variables
        for col in ['ALLSKY_SFC_SW_DWN', 'CLOUD_AMT']:
            if col in enhanced_df.columns:
                # 24h lag (same hour previous day)
                enhanced_df[f'{col}_24h_lag'] = enhanced_df[col].shift(
                    24).fillna(method='bfill')

                # 168h lag (same hour previous week)
                if len(enhanced_df) > 168:  # Only if we have enough data
                    enhanced_df[f'{col}_168h_lag'] = enhanced_df[col].shift(
                        168).fillna(method='bfill')

        # 6. Wind vector components
        if 'WS2M' in enhanced_df.columns and 'WD2M' in enhanced_df.columns:
            # Convert wind direction to radians
            wind_dir_rad = np.radians(enhanced_df['WD2M'])

            # Calculate u (east-west) and v (north-south) components
            enhanced_df['wind_u'] = -enhanced_df['WS2M'] * np.sin(wind_dir_rad)
            enhanced_df['wind_v'] = -enhanced_df['WS2M'] * np.cos(wind_dir_rad)

        # Fill any nulls created during feature engineering
        enhanced_df = enhanced_df.fillna(method='ffill').fillna(method='bfill')

        return enhanced_df

    def normalize_features(self, df, is_training=True):
        """
        Normalize features to improve model training.

        Parameters:
        - df: DataFrame with features
        - is_training: Whether this is training data (to fit scalers) or test/validation data

        Returns:
        - DataFrame with normalized features
        """
        self.logger.info("Normalizing features...")

        # Check if dataframe is empty
        if df.empty:
            self.logger.warning("Empty dataframe provided for normalization!")
            return df

        normalized_df = df.copy()

        # Identify different types of features for appropriate normalization
        # 1. Bounded features (use MinMaxScaler)
        bounded_features = ['RH2M', 'CLOUD_AMT', 'ALLSKY_KT', 'clear_sky_index',
                            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                            'doy_sin', 'doy_cos', 'month_sin', 'month_cos']
        bounded_features = [f for f in bounded_features if f in df.columns]

        # 2. Unbounded features (use StandardScaler)
        unbounded_features = [col for col in df.columns
                              if col not in bounded_features
                              and not (col.endswith('_sin') or col.endswith('_cos'))
                              and col != 'location_id']  # Don't normalize location ID

        # Initialize or use existing scalers
        if is_training:
            # Create new scalers for training data
            self.feature_scalers['bounded'] = MinMaxScaler()
            self.feature_scalers['unbounded'] = StandardScaler()

            # Fit scalers if we have the features
            if bounded_features:
                self.feature_scalers['bounded'].fit(df[bounded_features])
            if unbounded_features:
                self.feature_scalers['unbounded'].fit(df[unbounded_features])

        # Transform using scalers
        if bounded_features and 'bounded' in self.feature_scalers:
            normalized_df[bounded_features] = self.feature_scalers['bounded'].transform(
                df[bounded_features])
        if unbounded_features and 'unbounded' in self.feature_scalers:
            normalized_df[unbounded_features] = self.feature_scalers['unbounded'].transform(
                df[unbounded_features])

        return normalized_df

    def create_sequences(self, df, target_col='ALLSKY_SFC_SW_DWN', overlap=12):
        """
        Create input-target sequences for model training.

        Parameters:
        - df: Normalized DataFrame
        - target_col: Column name of the target variable
        - overlap: Number of overlapping time steps between sequences

        Returns:
        - Dictionary with input and target tensors
        """
        self.logger.info("Creating training sequences...")

        # Check if we have enough data
        if len(df) < self.lookback_history + self.estimate_length:
            self.logger.warning(
                f"DataFrame length ({len(df)}) is less than required sequence length ({self.lookback_history + self.estimate_length})!")
            return {'inputs': [], 'targets': []}

        # Check if target column exists
        if target_col not in df.columns:
            self.logger.error(
                f"Target column '{target_col}' not found in DataFrame!")
            available_cols = ', '.join(df.columns)
            self.logger.info(f"Available columns: {available_cols}")
            return {'inputs': [], 'targets': []}

        # Separate target variable from features
        features = df.drop(columns=[target_col])
        target = df[target_col]

        sequence_length = self.lookback_history + self.estimate_length
        stride = self.estimate_length - overlap

        # Ensure stride is at least 1
        stride = max(1, stride)

        input_sequences = []
        target_sequences = []

        # Create sequences
        i = 0
        while i + sequence_length <= len(df):
            # Extract sequence
            sequence = df.iloc[i:i+sequence_length]

            # Feature sequence (historical)
            x = sequence.iloc[:self.lookback_history].values

            # Target sequence (future GHI)
            y_ghi = sequence.iloc[self.lookback_history:
                                  ][target_col].values.reshape(-1, 1)

            # Future covariates (without GHI)
            y_covs = sequence.iloc[self.lookback_history:].drop(
                columns=[target_col]).values

            input_sequences.append({
                'historical': x,
                'future_covariates': y_covs
            })

            target_sequences.append(y_ghi)

            # Move forward by stride
            i += stride

        self.logger.info(f"Created {len(input_sequences)} sequences")

        return {
            'inputs': input_sequences,
            'targets': target_sequences
        }

    def ensure_float32_types(self, sequences):
        """Ensure all numpy arrays in sequences are float32 type."""

        # Convert inputs
        for i in range(len(sequences['inputs'])):
            # Convert historical data
            if isinstance(sequences['inputs'][i]['historical'], np.ndarray):
                sequences['inputs'][i]['historical'] = sequences['inputs'][i]['historical'].astype(
                    np.float32)

            # Convert future covariates
            if isinstance(sequences['inputs'][i]['future_covariates'], np.ndarray):
                sequences['inputs'][i]['future_covariates'] = sequences['inputs'][i]['future_covariates'].astype(
                    np.float32)

        # Convert targets
        for i in range(len(sequences['targets'])):
            if isinstance(sequences['targets'][i], np.ndarray):
                sequences['targets'][i] = sequences['targets'][i].astype(
                    np.float32)

        return sequences

    def create_tf_dataset(self, sequences, batch_size=32, shuffle=True, cache=True):
        """
        Create a TensorFlow dataset from sequences.

        Parameters:
        - sequences: Dictionary with input and target sequences
        - batch_size: Batch size for training
        - shuffle: Whether to shuffle the data
        - cache: Whether to cache the dataset

        Returns:
        - TensorFlow dataset
        """
        import tensorflow as tf

        # Check if we have sequences
        if not sequences['inputs']:
            self.logger.warning("No sequences to create dataset from!")
            # Return an empty dataset with proper structure
            return tf.data.Dataset.from_tensor_slices((
                {
                    'historical': tf.zeros([0, self.lookback_history, 1]),
                    'future_covariates': tf.zeros([0, self.estimate_length, 1])
                },
                tf.zeros([0, self.estimate_length, 1])
            ))

        # Convert sequence dictionaries to tensors
        try:
            sequences = self.ensure_float32_types(sequences)
            historical_data = np.array(
                [seq['historical'] for seq in sequences['inputs']])
            future_covariates = np.array(
                [seq['future_covariates'] for seq in sequences['inputs']])
            targets = np.array(sequences['targets'])

            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'historical': historical_data,
                    'future_covariates': future_covariates
                },
                targets
            ))
        except Exception as e:
            self.logger.error(f"Error creating TensorFlow dataset: {str(e)}")
            # Return empty dataset
            return tf.data.Dataset.from_tensor_slices((
                {
                    'historical': tf.zeros([0, self.lookback_history, 1]),
                    'future_covariates': tf.zeros([0, self.estimate_length, 1])
                },
                tf.zeros([0, self.estimate_length, 1])
            ))

        # Apply dataset transformations
        if cache:
            dataset = dataset.cache()

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(sequences['inputs']))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def prepare_data(self,
                     start_date,
                     end_date,
                     train_split=0.8,
                     val_split=0.1,
                     batch_size=32,
                     save_dir=None):
        """
        End-to-end data preparation pipeline.

        Parameters:
        - start_date: Start date for data collection
        - end_date: End date for data collection
        - train_split: Proportion of data for training
        - val_split: Proportion of data for validation
        - batch_size: Batch size for datasets
        - save_dir: Directory to save processed data

        Returns:
        - Dictionary with train, validation, and test datasets
        """
        all_data = []

        # For very small datasets/testing, we might only use one location
        for location in self.locations:
            # 1. Fetch data
            if save_dir:
                raw_path = os.path.join(
                    save_dir, 'raw', f"{location['name']}.csv")
                os.makedirs(os.path.join(save_dir, 'raw'), exist_ok=True)
            else:
                raw_path = None

            df = self.fetch_data(start_date, end_date,
                                 location, save_path=raw_path)

            # Check if we got any data
            if df.empty:
                self.logger.warning(
                    f"No data retrieved for {location['name']}, skipping...")
                continue

            # 2. Preprocess
            df = self.preprocess_data(df)

            # 3. Engineer features
            df = self.engineer_features(df)

            # Add location identifier
            df['location_id'] = self.locations.index(location)

            all_data.append(df)

        # Check if we have any data
        if not all_data:
            self.logger.error("No data available for any location!")
            # Return empty datasets
            import tensorflow as tf
            empty_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'historical': tf.zeros([0, self.lookback_history, 1]),
                    'future_covariates': tf.zeros([0, self.estimate_length, 1])
                },
                tf.zeros([0, self.estimate_length, 1])
            ))
            return {
                'train': empty_dataset,
                'validation': empty_dataset,
                'test': empty_dataset,
                'scalers': self.feature_scalers
            }

        # Combine data from all locations
        combined_df = pd.concat(all_data)

        # Sort by datetime index
        combined_df = combined_df.sort_index()

        # Print dataset info
        self.logger.info(f"Combined dataset shape: {combined_df.shape}")
        self.logger.info(
            f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

        # 4. Normalize features
        normalized_df = self.normalize_features(combined_df, is_training=True)

        # 5. Split data (time-based split)
        train_end_idx = int(len(normalized_df) * train_split)
        val_end_idx = int(len(normalized_df) * (train_split + val_split))

        # Ensure we have enough data for each split
        if train_end_idx < self.lookback_history + 1:
            self.logger.warning(
                f"Not enough data for training! Only {train_end_idx} samples.")
            # Default to 60% if not enough data
            train_end_idx = int(len(normalized_df) * 0.6)
            # Default to 20% for validation
            val_end_idx = int(len(normalized_df) * 0.8)

        train_df = normalized_df.iloc[:train_end_idx]
        val_df = normalized_df.iloc[train_end_idx:val_end_idx]
        test_df = normalized_df.iloc[val_end_idx:]

        # 6. Create sequences
        train_sequences = self.create_sequences(train_df)
        val_sequences = self.create_sequences(val_df)
        test_sequences = self.create_sequences(test_df)

        # 7. Create TensorFlow datasets
        train_dataset = self.create_tf_dataset(
            train_sequences, batch_size=batch_size, shuffle=True)
        val_dataset = self.create_tf_dataset(
            val_sequences, batch_size=batch_size, shuffle=False)
        test_dataset = self.create_tf_dataset(
            test_sequences, batch_size=batch_size, shuffle=False)

        # 8. Save data if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Save scalers
            import joblib
            joblib.dump(self.feature_scalers, os.path.join(
                save_dir, 'scalers.joblib'))

            # Save split indices for reference
            with open(os.path.join(save_dir, 'splits.txt'), 'w') as f:
                f.write(f"Train: 0 to {train_end_idx-1}\n")
                f.write(f"Validation: {train_end_idx} to {val_end_idx-1}\n")
                f.write(f"Test: {val_end_idx} to {len(normalized_df)-1}\n")

        return {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset,
            'scalers': self.feature_scalers
        }
