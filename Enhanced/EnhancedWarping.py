import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import Rbf

class AdvancedClothingWarper:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.3,  # Lower threshold
            model_complexity=1  # Medium complexity
        )

    def detect_landmarks(self, image_path):
        """Enhanced landmark detection with fallback"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print("MediaPipe landmark detection failed. Using alternative method.")
            return self.fallback_landmark_detection(image)
        
        landmarks = results.pose_landmarks.landmark
        height, width, _ = image.shape
        
        key_indices = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]
        
        landmark_points = [
            [int(landmarks[idx].x * width), int(landmarks[idx].y * height)] 
            for idx in key_indices
        ]
        
        return np.array(landmark_points, dtype=np.float32)

    def fallback_landmark_detection(self, image):
        """Manual landmark estimation using image contours"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(body_contour)
        
        return np.array([
            [x + int(0.2*w), y + int(0.2*h)],   # Left shoulder
            [x + int(0.8*w), y + int(0.2*h)],   # Right shoulder
            [x + int(0.2*w), y + int(0.8*h)],   # Left hip
            [x + int(0.8*w), y + int(0.8*h)]    # Right hip
        ], dtype=np.float32)

    def thin_plate_spline_warp(self, src_pts, dst_pts, image):
        """Thin Plate Spline Warping for better non-rigid transformation"""
        h, w, c = image.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        rbf_x = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 0], function='thin_plate')
        rbf_y = Rbf(src_pts[:, 0], src_pts[:, 1], dst_pts[:, 1], function='thin_plate')
        
        map_x = rbf_x(grid_x, grid_y).astype(np.float32)
        map_y = rbf_y(grid_x, grid_y).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def advanced_warp(self, clothing_path, body_image_path, mask_path):
        clothing_image = cv2.imread(clothing_path)
        body_image = cv2.imread(body_image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize mask to match body image
        mask = cv2.resize(mask, (body_image.shape[1], body_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        source_landmarks = self.detect_landmarks(clothing_path)
        target_landmarks = self.detect_landmarks(body_image_path)
        
        if source_landmarks is None or target_landmarks is None:
            print("Landmark detection completely failed!")
            return None
        
        warped_clothing = self.thin_plate_spline_warp(source_landmarks, target_landmarks, clothing_image)
        
        # Resize warped clothing to match body image
        warped_clothing = cv2.resize(warped_clothing, (body_image.shape[1], body_image.shape[0]))
        
        # Improve blending with alpha blending
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha = mask_blurred[:, :, None] / 255.0
        final_result = (warped_clothing * alpha + body_image * (1 - alpha)).astype(np.uint8)
        
        return final_result

def main():
    warper = AdvancedClothingWarper()
    
    clothing_path = 'Shirts/shirt3.png'
    body_image_path = 'person.jpg'
    mask_path = 'Enhanced/Segmentation/upperBody.png'
    output_path = 'Enhanced/Outputs/warped_shirt8.png'
    
    result = warper.advanced_warp(clothing_path, body_image_path, mask_path)
    
    if result is not None:
        cv2.imwrite(output_path, result)
        print(f"Warped clothing saved to {output_path}")

if __name__ == "__main__":
    main()
