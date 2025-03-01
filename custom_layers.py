import tensorflow as tf


class GroupNormalization(tf.keras.layers.Layer):
    """Replacement for tfa.layers.GroupNormalization with improved numerical stability"""

    def __init__(self, groups=32, axis=-1, epsilon=1e-5, center=True, scale=True, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) +
                             ' of input tensor should have a defined dimension')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) +
                             ') cannot be larger than the number of channels (' + str(dim) + ')')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) +
                             ') must be a multiple of the number of channels (' + str(dim) + ')')

        self.gamma = self.add_weight(
            name='gamma',
            shape=(dim,),
            initializer='ones',
            trainable=True if self.scale else False)

        self.beta = self.add_weight(
            name='beta',
            shape=(dim,),
            initializer='zeros',
            trainable=True if self.center else False)

    def call(self, inputs):
        input_shape = tf.shape(inputs)

        # Handle 3D inputs (batch, seq_len, channels)
        if len(inputs.shape) == 3:
            batch_size = input_shape[0]
            seq_len = input_shape[1]
            channels = inputs.shape[2]

            channels_per_group = channels // self.groups

            # Reshape for group normalization
            reshaped = tf.reshape(
                inputs, [batch_size, seq_len, self.groups, channels_per_group])

            # Compute mean and variance along sequence and channels per group dimensions
            mean, variance = tf.nn.moments(
                reshaped, axes=[1, 3], keepdims=True)

            # Add epsilon for numerical stability
            normalized = (reshaped - mean) / tf.sqrt(variance + self.epsilon)

            # Reshape gamma and beta
            gamma_reshape = tf.reshape(
                self.gamma, [1, 1, self.groups, channels_per_group])
            beta_reshape = tf.reshape(
                self.beta, [1, 1, self.groups, channels_per_group])

            # Apply scale and shift
            normalized = normalized * gamma_reshape + beta_reshape

            # Reshape back to original dimensions
            return tf.reshape(normalized, [batch_size, seq_len, channels])

        # Handle other input shapes
        return tf.keras.layers.GroupNormalization(
            groups=self.groups,
            axis=self.axis,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale)(inputs)


def entmax15(inputs, axis=-1):
    """
    α-entmax transformation with α=1.5 with improved numerical stability.
    This is a sparse attention mechanism that produces sparser distributions than softmax.
    """
    # Apply max subtraction for numerical stability (similar to softmax)
    inputs_max = tf.reduce_max(inputs, axis=axis, keepdims=True)
    inputs_shifted = inputs - inputs_max

    # Compute threshold for trimming
    threshold = tf.reduce_mean(inputs_shifted, axis=axis, keepdims=True)

    # Compute sparse attention weights with α=1.5
    weights = tf.maximum(0.0, inputs_shifted - threshold) ** 2

    # Normalize weights to sum to 1 with higher epsilon
    weights_sum = tf.reduce_sum(weights, axis=axis, keepdims=True)
    weights = weights / (weights_sum + 1e-5)  # Increased epsilon for stability

    # Handle edge case where all weights are zero
    is_all_zero = tf.cast(tf.equal(weights_sum, 0), tf.float32)
    uniform_weights = tf.ones_like(
        weights) / tf.cast(tf.shape(weights)[axis], tf.float32)
    weights = weights * (1 - is_all_zero) + uniform_weights * is_all_zero

    return weights
