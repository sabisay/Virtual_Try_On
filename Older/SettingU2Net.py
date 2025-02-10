import os
import keras
import jax
import jax.numpy as jnp

# Ensure JAX uses GPU
#assert jax.devices()[0].device_kind == "NVIDIA GPU", "JAX is not using a GPU!"

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# Set Keras backend
os.environ["KERAS_BACKEND"] = "jax"

# Load the U²-Net model from Hugging Face
model = keras.saving.load_model("hf://reidn3r/u2net-image-rembg")

print("Model loaded successfully on GPU:", jax.devices()[0])