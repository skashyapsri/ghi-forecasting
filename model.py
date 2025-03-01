import tensorflow as tf
import numpy as np
import math

# Import custom layers instead of tensorflow_addons
from custom_layers import GroupNormalization, entmax15


class Dense(tf.Module):
    """Ultra-minimal Dense layer implementation without any shape checking."""

    def __init__(self, input_size, output_size, activation=None, stddev=1.0, name=''):
        super(Dense, self).__init__()
        self.w = tf.Variable(
            tf.random.truncated_normal([input_size, output_size], stddev=stddev), name=name + '_w')
        self.b = tf.Variable(tf.zeros([output_size]), name=name+'_b')
        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size

    def __call__(self, x):
        """
        Forward pass supporting any input shape by reshaping automatically.

        Parameters:
        - x: Input tensor of arbitrary shape

        Returns:
        - Output tensor
        """
        # Get dynamic shape
        x_shape = tf.shape(x)

        # Calculate total size except the last dimension
        batch_dims = x_shape[:-1]
        batch_size = tf.reduce_prod(batch_dims)

        # Reshape to 2D: [batch_size, features]
        x_reshaped = tf.reshape(x, [batch_size, x_shape[-1]])

        # Apply linear transformation
        y_flat = tf.matmul(x_reshaped, self.w) + self.b

        # Apply activation if specified
        if self.activation is not None:
            y_flat = self.activation(y_flat)

        # Reshape back with output feature dimension
        new_shape = tf.concat([batch_dims, [self.output_size]], axis=0)
        y = tf.reshape(y_flat, new_shape)

        return y


class SparseAttentionLayer(tf.Module):
    """Attention layer with α-entmax sparse attention mechanism."""

    def __init__(self,
                 d_model,
                 num_attention_heads,
                 head_size,
                 attention_mask=None,
                 activation_fn=None,
                 initializer_range=1.0):
        super(SparseAttentionLayer, self).__init__()

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.attention_mask = attention_mask
        self.last_attention_weights = None
        self.query_layer = Dense(d_model,
                                 num_attention_heads * head_size,
                                 activation=activation_fn,
                                 stddev=initializer_range,
                                 name='query_layer')

        self.key_layer = Dense(d_model,
                               num_attention_heads * head_size,
                               activation=activation_fn,
                               stddev=initializer_range,
                               name='key_layer')

        self.value_layer = Dense(d_model,
                                 num_attention_heads * head_size,
                                 activation=activation_fn,
                                 stddev=initializer_range,
                                 name='value_layer')

    def __call__(self, to_sequence, from_sequence):
        # Get dynamic batch size and sequence lengths
        batch_size = tf.shape(to_sequence)[0]
        to_sequence_length = tf.shape(to_sequence)[1]
        from_sequence_length = tf.shape(from_sequence)[1]

        query = self.query_layer(to_sequence)
        # (b, t, d) --> (b, t, h, dh)
        Q = tf.reshape(query, [batch_size, to_sequence_length,
                       self.num_attention_heads, self.head_size])
        # (b, t, h, dh) --> (b, h, t, dh)
        Q = tf.transpose(Q, [0, 2, 1, 3])

        key = self.key_layer(from_sequence)
        # (b, f, d) --> (b, f, h, dh)
        K = tf.reshape(key, [batch_size, from_sequence_length,
                       self.num_attention_heads, self.head_size])
        # (b, f, h, dh) --> (b, h, f, dh)
        K = tf.transpose(K, [0, 2, 1, 3])

        value = self.value_layer(from_sequence)
        # (b, f, d) --> (b, f, h, dh)
        V = tf.reshape(value, [batch_size, from_sequence_length,
                       self.num_attention_heads, self.head_size])
        # (b, f, h, dh) --> (b, h, f, dh)
        V = tf.transpose(V, [0, 2, 1, 3])

        # (b, h, t, dh), (b, h, f, dh)T --> (b, h, t, f)
        output = tf.matmul(Q, K, transpose_b=True) / \
            tf.math.sqrt(float(self.head_size))

        if self.attention_mask is not None:
            # If mask is 2D, expand it to 4D for broadcasting
            if len(self.attention_mask.shape) == 2:
                # (t, f) --> (1, 1, t, f)
                attention_mask = tf.expand_dims(
                    tf.expand_dims(self.attention_mask, 0), 0)
            elif len(self.attention_mask.shape) == 3:
                # (b, t, f) --> (b, 1, t, f)
                attention_mask = tf.expand_dims(self.attention_mask, 1)

            # Apply mask and scale
            attention_mask = tf.cast(attention_mask, tf.float32) * -10000.0
            output = output + attention_mask

        # Apply α-entmax sparse attention
        self.last_attention_weights = entmax15(output)
        output = entmax15(output)
        output = tf.matmul(output, V)

        # (b, h, t, dh) --> (b, t, h, dh))
        output = tf.transpose(output, [0, 2, 1, 3])
        # (b, t, h, dh) --> (b, t, d)
        output_shape = tf.concat(
            [tf.shape(to_sequence)[:-1], [self.num_attention_heads * self.head_size]], axis=0)
        return tf.reshape(output, output_shape)


def dropout(input, dropout_prob):
    """Apply dropout if dropout_prob > 0."""
    if dropout_prob == 0.0:
        return input
    return tf.nn.dropout(input, dropout_prob)


class EncoderLayer(tf.Module):
    """Transformer encoder layer with sparse attention."""

    def __init__(self,
                 hidden_size,
                 feedforward_size,
                 num_attention_heads,
                 head_size,
                 activation_fn=None,
                 feedforward_activation_fn=tf.nn.relu,
                 dropout_prob=0.1,
                 initializer_range=1.0):
        super(EncoderLayer, self).__init__()

        self.dropout_prob = dropout_prob

        self.self_attention_layer = SparseAttentionLayer(
            d_model=hidden_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
            activation_fn=activation_fn,
            initializer_range=initializer_range)

        self.layer_normalization_1 = GroupNormalization(groups=1, axis=-1)

        self.feedforward_layer = Dense(
            hidden_size,
            feedforward_size,
            activation=feedforward_activation_fn,
            stddev=initializer_range,
            name="feedforward_layer")

        self.layer_normalization_2 = GroupNormalization(groups=1, axis=-1)

        self.project_back_layer = Dense(
            feedforward_size,
            hidden_size,
            activation=None,
            stddev=initializer_range,
            name="project_back_layer")

    def __call__(self, input):
        output = self.self_attention_layer(
            input,
            input)

        output = self.layer_normalization_1(output)

        attention_output = input + dropout(output, self.dropout_prob)

        output = self.feedforward_layer(attention_output)

        output = self.project_back_layer(output)

        output = self.layer_normalization_2(output)

        return attention_output + dropout(output, self.dropout_prob)


class DecoderLayer(tf.Module):
    """Transformer decoder layer with sparse attention."""

    def __init__(self,
                 hidden_size,
                 feedforward_size,
                 num_attention_heads,
                 head_size,
                 use_masking=False,
                 activation_fn=None,
                 feedforward_activation_fn=tf.nn.relu,
                 dropout_prob=0.1,
                 initializer_range=1.0):
        super(DecoderLayer, self).__init__()

        self.dropout_prob = dropout_prob
        self.use_masking = use_masking

        # We'll create the mask dynamically during the call
        self.masked_attention_layer = SparseAttentionLayer(
            d_model=hidden_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
            attention_mask=None,  # Will be set dynamically
            activation_fn=activation_fn,
            initializer_range=initializer_range)

        self.layer_normalization_1 = GroupNormalization(groups=1, axis=-1)

        self.encoder_attention_layer = SparseAttentionLayer(
            d_model=hidden_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
            attention_mask=None,
            activation_fn=activation_fn,
            initializer_range=initializer_range)

        self.layer_normalization_2 = GroupNormalization(groups=1, axis=-1)

        self.feedforward_layer = Dense(
            hidden_size,
            feedforward_size,
            activation=feedforward_activation_fn,
            stddev=initializer_range,
            name="feedforward_layer")

        self.layer_normalization_3 = GroupNormalization(groups=1, axis=-1)

        self.project_back_layer = Dense(
            feedforward_size,
            hidden_size,
            activation=None,
            stddev=initializer_range,
            name="project_back_layer")

    def __call__(self, input, encoder_attention):
        # Create causal mask dynamically if needed
        if self.use_masking:
            seq_len = tf.shape(input)[1]
            # Create causal mask (lower triangular)
            mask = tf.linalg.band_part(
                tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
            # Set attention_mask for this call
            self.masked_attention_layer.attention_mask = mask

        output = self.masked_attention_layer(
            input,
            input)

        output = self.layer_normalization_1(output)

        masked_attention_output = input + dropout(output, self.dropout_prob)

        output = self.encoder_attention_layer(
            masked_attention_output,
            encoder_attention)

        output = self.layer_normalization_2(output)

        attention_output = masked_attention_output + \
            dropout(output, self.dropout_prob)

        output = self.feedforward_layer(attention_output)

        output = self.project_back_layer(output)

        output = self.layer_normalization_3(output)

        return attention_output + dropout(output, self.dropout_prob)


class Encoder(tf.Module):
    """Encoder stack with multiple encoder layers."""

    def __init__(self,
                 hidden_size,
                 feedforward_size,
                 num_layers,
                 num_attention_heads,
                 head_size,
                 activation_fn=None,
                 feedforward_activation_fn=tf.nn.relu,
                 dropout_prob=0.1,
                 initializer_range=1.0):
        super(Encoder, self).__init__()

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(
                hidden_size,
                feedforward_size,
                num_attention_heads,
                head_size,
                activation_fn,
                feedforward_activation_fn,
                dropout_prob,
                initializer_range))

    def __call__(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        return output


class Decoder(tf.Module):
    """Decoder stack with multiple decoder layers."""

    def __init__(self,
                 hidden_size,
                 feedforward_size,
                 num_layers,
                 num_attention_heads,
                 head_size,
                 masking=True,
                 activation_fn=None,
                 feedforward_activation_fn=tf.nn.relu,
                 dropout_prob=0.1,
                 initializer_range=1.0):
        super(Decoder, self).__init__()

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(
                hidden_size,
                feedforward_size,
                num_attention_heads,
                head_size,
                use_masking=masking,
                activation_fn=activation_fn,
                feedforward_activation_fn=feedforward_activation_fn,
                dropout_prob=dropout_prob,
                initializer_range=initializer_range))

    def __call__(self, input, encoder_attentions):
        output = input

        for layer in self.layers:
            output = layer(output, encoder_attentions)

        return output


class AST(tf.Module):
    """
    Adversarial Sparse Transformer for GHI forecasting.
    Implements sparse attention using α-entmax transformation with α=1.5
    """

    def __init__(self,
                 hidden_size,
                 feedforward_size,
                 num_encoder_layers=3,  # As per thesis specification
                 num_decoder_layers=3,  # As per thesis specification
                 num_attention_heads=8,  # As per thesis specification
                 head_size=32,          # As per thesis specification
                 decoder_masking=False,
                 activation_fn=None,
                 feedforward_activation_fn=tf.nn.relu,
                 dropout_prob=0.1,
                 initializer_range=1.0):
        super(AST, self).__init__()

        self.encoder = Encoder(
            hidden_size,
            feedforward_size,
            num_encoder_layers,
            num_attention_heads,
            head_size,
            activation_fn=activation_fn,
            feedforward_activation_fn=feedforward_activation_fn,
            dropout_prob=dropout_prob,
            initializer_range=initializer_range)

        self.decoder = Decoder(
            hidden_size,
            feedforward_size,
            num_decoder_layers,
            num_attention_heads,
            head_size,
            masking=decoder_masking,
            activation_fn=activation_fn,
            feedforward_activation_fn=feedforward_activation_fn,
            dropout_prob=dropout_prob,
            initializer_range=initializer_range)

    def __call__(self, encoder_input, decoder_input):
        encoder_attentions = self.encoder(encoder_input)
        return self.decoder(decoder_input, encoder_attentions)


class Embedding(tf.Module):
    """Parameter embedding layer."""

    def __init__(self,
                 hidden_size,
                 num_classes,
                 dropout_prob=0.1,
                 initializer_range=1.0):
        super(Embedding, self).__init__()

        self.embedding_table = tf.Variable(tf.random.truncated_normal(
            [num_classes, hidden_size], stddev=initializer_range), name='embedding_table')

    def __call__(self, index):
        # Get dynamic batch size
        batch_size = tf.shape(index)[0]

        # [I, d] --> [1, I, d]
        embedding_expanded = tf.expand_dims(self.embedding_table, 0)
        # [1, I, d] --> [B, I, d]
        embedding_expanded = tf.tile(
            embedding_expanded, [batch_size, 1, 1])

        # [B, I, d] --> [B, t, d]
        return tf.gather(embedding_expanded, index, axis=1, batch_dims=1, name="embedding")


class Generator(tf.Module):
    """
    Generator network for GHI forecasting.
    Processes historical GHI data and meteorological parameters to predict future GHI values.

    Parameters:
    - lookback_history: Number of historical time steps (e.g., 168 hours)
    - estimate_length: Number of future time steps to predict (e.g., 24 hours)
    - num_features: Number of input features (GHI + meteorological parameters)
    - embedding_size: Size of parameter embeddings
    - hidden_size: Dimensionality of hidden representation
    - feedforward_size: Size of feed-forward network
    - num_hidden_layers: Number of encoder/decoder layers
    - num_attention_heads: Number of attention heads
    - head_size: Dimension of each attention head
    """

    def __init__(self,
                 lookback_history,
                 estimate_length,
                 num_features,
                 embedding_size,
                 hidden_size,
                 feedforward_size,
                 num_hidden_layers=3,
                 num_attention_heads=8,
                 head_size=32,
                 activation_fn=None,
                 dropout_prob=0.1,
                 initializer_range=1.0,
                 is_training=False):
        super(Generator, self).__init__()

        def sine_cosine_position_method(sequence_length, hidden_size):
            pos = tf.expand_dims(
                tf.range(sequence_length, delta=1, dtype=tf.float32), -1)
            position_table = tf.Variable(
                tf.zeros([sequence_length, hidden_size], dtype=tf.float32))

            # Apply sinusoidal position encoding
            position_table[:, 0::2].assign(tf.math.sin(pos / tf.pow(10000.0, tf.expand_dims(
                tf.range(0, hidden_size, delta=2, dtype=tf.float32), 0) / hidden_size)))
            position_table[:, 1::2].assign(tf.math.cos(pos / tf.pow(10000.0, tf.expand_dims(
                tf.range(1, hidden_size, delta=2, dtype=tf.float32), 0) / hidden_size)))

            return position_table

        if not is_training:
            dropout_prob = 0.0

        self.lookback_history = lookback_history
        self.estimate_length = estimate_length
        self.dropout_prob = dropout_prob
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range

        # Initialize alignment layers with input_size=None, will be created dynamically
        self.encoder_alignment_layer = None
        self.decoder_alignment_layer = None

        # Position encodings
        self.encoder_position_table = sine_cosine_position_method(
            lookback_history, hidden_size)
        self.decoder_position_table = sine_cosine_position_method(
            estimate_length, hidden_size)

        # Initialize the AST with smaller components
        self.transformer_layer = AST(
            hidden_size=hidden_size,
            feedforward_size=feedforward_size,
            num_encoder_layers=num_hidden_layers,
            num_decoder_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
            decoder_masking=False,
            activation_fn=activation_fn,
            feedforward_activation_fn=tf.nn.relu,
            dropout_prob=dropout_prob,
            initializer_range=initializer_range
        )

        # Output projection layer
        self.output_layer = Dense(hidden_size,
                                  1,
                                  activation=activation_fn,
                                  stddev=initializer_range,
                                  name='output_layer')

        # Layer normalization
        self.group_normalization_1 = GroupNormalization(
            groups=1, axis=-1)
        self.group_normalization_2 = GroupNormalization(
            groups=1, axis=-1)

    def __call__(self, historical_data, future_covariates):
        """
        Forward pass through the generator with minimal conditionals to avoid gradient issues.

        Parameters:
        - historical_data: Input tensor of shape [batch_size, lookback_history, num_features]
        - future_covariates: Input tensor of shape [batch_size, estimate_length, num_features-1]

        Returns:
        - Predicted GHI values of shape [batch_size, estimate_length, 1]
        """
        # Ensure inputs are float32
        historical_data = tf.cast(historical_data, tf.float32)
        future_covariates = tf.cast(future_covariates, tf.float32)

        # Directly create alignment layers if needed
        if self.encoder_alignment_layer is None:
            print(
                f"Creating encoder alignment layer with input size {historical_data.shape[-1]}")
            self.encoder_alignment_layer = Dense(
                historical_data.shape[-1],
                self.hidden_size,
                stddev=self.initializer_range,
                name='encoder_alignment_layer')

        if self.decoder_alignment_layer is None:
            print(
                f"Creating decoder alignment layer with input size {future_covariates.shape[-1]}")
            self.decoder_alignment_layer = Dense(
                future_covariates.shape[-1],
                self.hidden_size,
                stddev=self.initializer_range,
                name='decoder_alignment_layer')

        # Process encoder input - straightforward transformation
        encoder_input = self.encoder_alignment_layer(historical_data)
        # Add position embeddings - broadcast across batch dimension
        encoder_input = encoder_input + \
            tf.expand_dims(self.encoder_position_table, 0)
        encoder_input = self.group_normalization_1(encoder_input)
        encoder_input = tf.nn.dropout(encoder_input, self.dropout_prob)

        # Process decoder input - straightforward transformation
        decoder_input = self.decoder_alignment_layer(future_covariates)
        # Add position embeddings - broadcast across batch dimension
        decoder_input = decoder_input + \
            tf.expand_dims(self.decoder_position_table, 0)
        decoder_input = self.group_normalization_2(decoder_input)
        decoder_input = tf.nn.dropout(decoder_input, self.dropout_prob)

        # Generate GHI forecast
        output = self.transformer_layer(encoder_input, decoder_input)
        output = self.output_layer(output)

        # Ensure output has shape [batch_size, estimate_length, 1]
        if len(output.shape) < 3:
            output = tf.expand_dims(output, -1)

        return output


class Discriminator(tf.keras.Model):
    """
    A more robust discriminator that handles GHI time series data with proper shape handling.
    """

    def __init__(self, sequence_length, hidden_size=128, dropout_prob=0.2):
        super(Discriminator, self).__init__()

        self.sequence_length = sequence_length

        # Input normalization
        self.batch_norm = tf.keras.layers.BatchNormalization()

        # Ensure input has right shape with reshape layer
        self.reshape = tf.keras.layers.Reshape((sequence_length, 1))

        # Feature extraction with 1D convolutions
        self.conv1 = tf.keras.layers.Conv1D(
            32, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

        self.conv2 = tf.keras.layers.Conv1D(
            64, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)

        # Global pooling for sequence length invariance
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        # Dense classification layers
        self.dense1 = tf.keras.layers.Dense(
            hidden_size, activation=tf.keras.layers.LeakyReLU(0.2))
        self.dropout1 = tf.keras.layers.Dropout(dropout_prob)

        self.dense2 = tf.keras.layers.Dense(
            hidden_size // 2, activation=tf.keras.layers.LeakyReLU(0.2))
        self.dropout2 = tf.keras.layers.Dropout(dropout_prob)

        # Output layer - single value for GAN discrimination
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        # Store intermediate features for feature matching
        self.conv_features = None
        self.dense_features = None

    def call(self, inputs, training=True):
        # First, ensure we have the right shape - expecting [batch_size, sequence_length]
        # We'll add the channel dimension if needed

        # If rank is 2 (batch_size, sequence_length), add channel dimension
        if len(tf.shape(inputs)) == 2:
            x = self.reshape(inputs)
        else:
            # Already has channel dimension
            x = inputs

        # Apply batch normalization
        x = self.batch_norm(x, training=training)

        # Apply convolutional feature extraction
        x = self.conv1(x)
        x = self.pool1(x)

        # Store conv features for feature matching
        self.conv_features = x

        x = self.conv2(x)
        x = self.pool2(x)

        # Global pooling to handle variable sequence lengths
        x = self.global_avg_pool(x)

        # Dense layers
        x = self.dense1(x)
        if training:
            x = self.dropout1(x)

        # Store dense features
        self.dense_features = x

        x = self.dense2(x)
        if training:
            x = self.dropout2(x)

        # Output layer returns shape [batch_size, 1]
        x = self.output_layer(x)

        # Return scalar outputs of shape [batch_size]
        return tf.squeeze(x)

    def feature_matching(self, x, training=False):
        """Extract intermediate features for feature matching loss."""
        # First run forward pass to get features
        _ = self.call(x, training=training)

        # Return stored features
        return {
            'conv_features': self.conv_features,
            'dense_features': self.dense_features
        }
