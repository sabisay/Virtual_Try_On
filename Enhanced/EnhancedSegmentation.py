import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import torch
import datetime

class BodyPartSegmenter:
    def __init__(self, model_size='n'):
        self.model = YOLO(f'yolov8{model_size}-seg.pt')
        self.colors = {
            'head': (255, 0, 0),    # Blue
            'upper_body': (0, 255, 0),  # Green
            'arms': (0, 0, 255),    # Red
            'lower_body': (255, 255, 0),  # Cyan
            'legs': (255, 0, 255)    # Magenta
        }
        
    def get_body_segmentation(self, image_path):
        """Get segmentation mask using YOLOv8."""
        results = self.model(image_path, stream=True)
        
        for result in results:
            if result.masks is not None:
                person_masks = result.masks.data[result.boxes.cls == 0]
                if len(person_masks) > 0:
                    areas = torch.sum(person_masks.bool(), dim=(1,2))
                    largest_mask = person_masks[torch.argmax(areas)]
                    return largest_mask.cpu().numpy()
            
        raise ValueError("No person detected in the image")

    def segment_body_parts(self, mask):
        """Segment different body parts with specific ratios."""
        h, w = mask.shape
        body_parts = {}
        
        # Head (top 15% of body)
        head_height = int(h * 0.15)
        body_parts['head'] = np.zeros_like(mask)
        body_parts['head'][:head_height, :] = mask[:head_height, :]
        
        # Upper body (15-40% of height)
        upper_start = head_height
        upper_end = int(h * 0.4)
        body_parts['upper_body'] = np.zeros_like(mask)
        body_parts['upper_body'][upper_start:upper_end, :] = mask[upper_start:upper_end, :]
        
        # Arms (detect from sides of upper body)
        arms_start = head_height
        arms_end = int(h * 0.5)
        body_parts['arms'] = np.zeros_like(mask)
        
        # Find contours for arms
        mask_section = (mask[arms_start:arms_end, :] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, _, w, _ = cv2.boundingRect(contour)
            arm_width = int(w * 0.2)
            temp_mask = np.zeros_like(mask_section)
            temp_mask[:, :x+arm_width] = mask_section[:, :x+arm_width]  # Left arm
            temp_mask[:, x+w-arm_width:] = mask_section[:, x+w-arm_width:]  # Right arm
            body_parts['arms'][arms_start:arms_end, :] = temp_mask
        
        # Lower body (40-65% of height)
        lower_start = upper_end
        lower_end = int(h * 0.65)
        body_parts['lower_body'] = np.zeros_like(mask)
        body_parts['lower_body'][lower_start:lower_end, :] = mask[lower_start:lower_end, :]
        
        # Legs (65-100% of height)
        legs_start = lower_end
        body_parts['legs'] = np.zeros_like(mask)
        body_parts['legs'][legs_start:, :] = mask[legs_start:, :]
        
        return body_parts

    def create_colored_mask(self, body_parts):
        """Create a colored mask where each body part has a different color."""
        # Create a 3-channel image
        colored_mask = np.zeros((body_parts['head'].shape[0], body_parts['head'].shape[1], 3), dtype=np.uint8)
        
        # Add each body part with its respective color
        for part_name, mask in body_parts.items():
            color = self.colors[part_name]
            colored_part = (mask > 0).astype(np.uint8) * 255
            for i in range(3):
                colored_mask[:, :, i] = np.where(colored_part > 0, 
                                               color[i], 
                                               colored_mask[:, :, i])
        
        return colored_mask

    def save_results(self, body_parts, colored_mask, output_dir="Enhanced/Segmentation"):
        """Save individual masks and colored combination."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}
        
        # Save individual masks
        for part_name, mask in body_parts.items():
            path = os.path.join(output_dir, f"{part_name}_mask_{timestamp}.png")
            cv2.imwrite(path, (mask * 255).astype(np.uint8))
            saved_paths[part_name] = path
        
        # Save colored mask
        colored_path = os.path.join(output_dir, f"colored_mask_{timestamp}.png")
        cv2.imwrite(colored_path, colored_mask)
        saved_paths['colored'] = colored_path
        
        return saved_paths

    def process_image(self, image_path, output_dir="Enhanced/Segmentation"):
        """Complete segmentation pipeline."""
        try:
            # Get full body mask
            full_mask = self.get_body_segmentation(image_path)
            
            # Segment body parts
            body_parts = self.segment_body_parts(full_mask)
            
            # Create colored visualization
            colored_mask = self.create_colored_mask(body_parts)
            
            # Save results
            saved_paths = self.save_results(body_parts, colored_mask, output_dir)
            
            return {
                "status": "success",
                "paths": saved_paths
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

if __name__ == "__main__":
    segmenter = BodyPartSegmenter()
    result = segmenter.process_image("person.jpg")
    print(result)