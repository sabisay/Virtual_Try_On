import cv2
import os

def blend_images(body_path, clothing_path, mask_path, output_folder="Outputs"):
    body = cv2.imread(body_path)
    clothing = cv2.imread(clothing_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if body is None or clothing is None or mask is None:
        raise ValueError("Error: Could not load one of the images. Check file paths.")

    # Ensure the mask size matches body size
    mask = cv2.resize(mask, (body.shape[1], body.shape[0]))
    
    # Convert mask to 3 channels
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Find a valid center for blending
    center_x = body.shape[1] // 2
    center_y = body.shape[0] // 2
    center = (center_x, center_y)

    # Blend using seamless cloning
    blended = cv2.seamlessClone(clothing, body, mask, center, cv2.MIXED_CLONE)
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Determine the filename for the new image
    output_filename = "blended_output.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # Check if the file already exists, if so, append a number to the filename
    count = 1
    while os.path.exists(output_path):
        output_filename = f"blended_output({count}).png"
        output_path = os.path.join(output_folder, output_filename)
        count += 1

    # Save the blended image to the output folder
    cv2.imwrite(output_path, blended)

    return output_path

blended_output_path = blend_images("person.jpg", "Clothets/warped_clothing.png", "Masks/mask.png")
print(f"Blended image saved as: {blended_output_path}")
