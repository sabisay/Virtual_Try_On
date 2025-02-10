import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import datetime
import torch

class ClothingWarper:
    def __init__(self, model_size='n'):
        """Initialize YOLOv8 segmentation model."""
        self.model = YOLO(f'yolov8{model_size}-seg.pt')
        
    def get_body_segmentation(self, image_path):
        """Get segmentation mask using YOLOv8."""
        # Run inference
        results = self.model(image_path, stream=True)
        
        # Get the first result (assuming single image)
        for result in results:
            # Get the masks
            if result.masks is not None:
                # Get person mask (class 0 in COCO dataset is person)
                person_masks = result.masks.data[result.boxes.cls == 0]
                if len(person_masks) > 0:
                    # Take the largest person mask if multiple detected
                    areas = torch.sum(person_masks.bool(), dim=(1,2))
                    largest_mask = person_masks[torch.argmax(areas)]
                    return largest_mask.cpu().numpy()
            
        raise ValueError("No person detected in the image")
    
    def extract_upper_body(self, mask, height_ratio=0.4):
        """Extract upper body region from full body mask."""
        h, w = mask.shape
        upper_height = int(h * height_ratio)
        
        # Create upper body mask
        upper_mask = np.zeros_like(mask)
        upper_mask[:upper_height, :] = mask[:upper_height, :]
        
        return upper_mask
    
    def _sort_points(self, points):
        """Sort points in top-left, top-right, bottom-left, bottom-right order."""
        # Sort by Y coordinate (top-bottom)
        sorted_by_y = points[points[:, 1].argsort()]
        
        # Get top and bottom points
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]
        
        # Sort top and bottom points by X coordinate
        top_points = top_points[top_points[:, 0].argsort()]
        bottom_points = bottom_points[bottom_points[:, 0].argsort()]
        
        return np.vstack((top_points, bottom_points))

    def get_control_points(self, mask):
        """Get control points for warping from the mask."""
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No valid contours found in the mask")
            
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Replace np.int0 with np.int32
        
        # Sort points to match source points order (top-left, top-right, bottom-left, bottom-right)
        box = self._sort_points(box)
        
        return box
    
    def warp_clothing(self, clothing_path, body_image_path, output_folder="Warped"):
        """Main method to warp clothing onto body image."""
        try:
            # Load images
            body_image = cv2.imread(body_image_path)
            clothing = cv2.imread(clothing_path)
            
            if body_image is None or clothing is None:
                raise ValueError("Could not load input images")
            
            # Get body segmentation
            body_mask = self.get_body_segmentation(body_image_path)
            
            # Extract upper body region
            upper_mask = self.extract_upper_body(body_mask)
            
            # Get control points
            target_points = self.get_control_points(upper_mask)
            
            # Get source points from clothing image
            h, w = clothing.shape[:2]
            source_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            
            # Calculate perspective transform
            matrix = cv2.getPerspectiveTransform(source_points, target_points.astype(np.float32))
            
            # Warp clothing
            warped_clothing = cv2.warpPerspective(
                clothing, 
                matrix, 
                (body_image.shape[1], body_image.shape[0])
            )
            
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)
            
            # Save results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            warped_path = os.path.join(output_folder, f"warped_{timestamp}.png")
            mask_path = os.path.join(output_folder, f"mask_{timestamp}.png")
            
            cv2.imwrite(warped_path, warped_clothing)
            cv2.imwrite(mask_path, (upper_mask * 255).astype(np.uint8))
            
            # Blend images
            alpha = 0.7
            blended = cv2.addWeighted(body_image, 1-alpha, warped_clothing, alpha, 0)
            blended_path = os.path.join(output_folder, f"blended_{timestamp}.png")
            cv2.imwrite(blended_path, blended)
            
            return {
                'status': 'success',
                'warped_path': warped_path,
                'mask_path': mask_path,
                'blended_path': blended_path
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    warper = ClothingWarper()  # Use 'n' for nano, 's' for small model
    result = warper.warp_clothing(
        clothing_path="Shirts/shirt2.png",
        body_image_path="person.jpg"
    )
    print(result)