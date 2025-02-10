import jax.numpy as jnp
import numpy as np
import cv2
import os
from PIL import Image
from SettingU2Net import model
import datetime

class BodySegmenter:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        self.model = model
        
    def preprocess_image(self, image_path):
        """Preprocess image with error handling and validation."""
        try:
            image = Image.open(image_path)
            
            # Check if image is RGBA and convert if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Preserve aspect ratio while resizing
            image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            # Create new image with padding to reach target size
            new_image = Image.new('RGB', self.target_size, (0, 0, 0))
            new_image.paste(image, ((self.target_size[0] - image.size[0]) // 2,
                                  (self.target_size[1] - image.size[1]) // 2))
            
            # Convert to numpy array and normalize
            image_array = np.array(new_image) / 255.0
            return image_array, new_image.size
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def segment_body(self, image_array):
        """Run model inference with input validation and proper output handling."""
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        image = jnp.array(image_array)
        image = jnp.expand_dims(image, axis=0)
        
        try:
            mask = self.model(image)
            # Handle different possible output types
            if isinstance(mask, list):
                # If model returns a list of outputs, take the last one
                # (usually the final prediction in U2Net)
                mask = mask[-1]
            
            # Convert to numpy array if it's not already
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            # Handle different dimensions
            if mask.ndim > 2:
                mask = np.squeeze(mask)  # Remove singleton dimensions
                
                # If we have multiple channels, take the first one
                if mask.ndim > 2:
                    mask = mask[..., 0]
            
            return mask
            
        except Exception as e:
            raise RuntimeError(f"Segmentation failed: {str(e)}")

    def post_process_mask(self, mask, kernel_size=5):
        """Enhanced post-processing with morphological operations."""
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        
        # Convert to 8-bit grayscale
        mask = (mask * 255).astype(np.uint8)
        
        # Apply Gaussian blur to smooth edges
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Binary thresholding with Otsu's method
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask

    def extract_body(self, binary_mask, original_image):
        """Extract body from original image using the mask."""
        # Ensure mask and image have same dimensions
        if binary_mask.shape[:2] != original_image.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))
        
        # Create 3-channel mask if needed
        if len(binary_mask.shape) == 2:
            binary_mask = np.stack([binary_mask] * 3, axis=-1)
        
        # Extract body
        body = cv2.bitwise_and(original_image, binary_mask)
        return body

    def save_results(self, mask, body, output_dir="Older/Masks", base_filename="result"):
        """Save both mask and segmented body with proper naming."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_path = os.path.join(output_dir, f"{base_filename}_mask_{timestamp}.png")
        body_path = os.path.join(output_dir, f"{base_filename}_body_{timestamp}.png")
        
        # Save files
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(body_path, body)
        
        return mask_path, body_path

    def process_image(self, image_path, output_dir="Older/Masks"):
        """Complete pipeline with error handling."""
        try:
            # Preprocess
            image_array, original_size = self.preprocess_image(image_path)
            
            # Segment
            raw_mask = self.segment_body(image_array)
            
            # Post-process
            processed_mask = self.post_process_mask(raw_mask)
            
            # Extract body
            original_image = cv2.imread(image_path)
            body = self.extract_body(processed_mask, original_image)
            
            # Save results
            mask_path, body_path = self.save_results(processed_mask, body, output_dir)
            
            return {
                "status": "success",
                "mask_path": mask_path,
                "body_path": body_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    segmenter = BodySegmenter()
    result = segmenter.process_image("person.jpg")
    print(result)