import os
import platform
import logging
import numpy as np

logger = logging.getLogger("gpu_utils")


def setup_gpu(gpu_id=0):
    """Configure GPU for TensorFlow based on platform."""
    # Check if we're on macOS
    if platform.system() == 'Darwin':
        return setup_mac_gpu()
    else:
        return setup_standard_gpu(gpu_id)


def setup_mac_gpu():
    """Setup GPU on macOS using PlaidML for AMD GPUs."""
    try:
        # Configure PlaidML for AMD GPU
        os.environ["PLAIDML_NATIVE_PATH"] = os.path.join(os.path.expanduser("~"),
                                                         ".local/lib/libplaidml.dylib")
        os.environ["PLAIDML_DEVICE_IDS"] = "metal_amd_radeon_pro_5300m.0"

        # Import necessary PlaidML components
        import plaidml.keras
        plaidml.keras.install_backend()

        # Set Keras backend to PlaidML
        os.environ["KERAS_BACKEND"] = "plaidml"

        logger.info("PlaidML backend configured for AMD Radeon Pro 5300M")
        return True
    except Exception as e:
        logger.error(f"Error configuring PlaidML: {str(e)}")
        logger.warning("Falling back to CPU")
        return False


def setup_standard_gpu(gpu_id=0):
    """Setup GPU on non-macOS platforms."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            logger.warning("No GPU found. Running on CPU.")
            return False

        if gpu_id >= 0 and gpu_id < len(gpus):
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            logger.info(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
            return True
        else:
            logger.warning(f"GPU ID {gpu_id} out of range. Running on CPU.")
            return False
    except Exception as e:
        logger.error(f"Error setting up GPU: {str(e)}")
        return False


def get_gpu_info():
    """Get GPU information."""
    if platform.system() == 'Darwin':
        try:
            import subprocess
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                    capture_output=True, text=True)
            info = {'system_profiler': result.stdout}

            # Check for AMD GPU
            if "AMD Radeon" in result.stdout:
                info['gpu_type'] = 'AMD Radeon'
                info['plaidml_support'] = 'Configured'

            return info
        except Exception as e:
            return {'error': f'Failed to get macOS GPU info: {str(e)}'}
    else:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return {'num_gpus': len(gpus), 'gpus': str(gpus)}
        except:
            return {'error': 'Failed to get GPU info'}
