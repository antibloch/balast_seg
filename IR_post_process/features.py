import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_rectangular_enhancement(image_path, output_path=None):
    """
    Process an image to enhance rectangular structures and visualize each step.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the visualization (optional)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Basic preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred)
    
    # Step 2: Directional edge detection
    sobelx = cv2.Sobel(contrast_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(contrast_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    
    # Step 3: Morphological operations for line enhancement
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horizontal_lines = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_OPEN, vertical_kernel)
    
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    
    structural_elements = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Step 4: Edge detection
    edges = cv2.Canny(contrast_enhanced, 50, 150)
    
    # Step 5: Morphological gradient for structure boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_gradient = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_GRADIENT, kernel)
    
    # Step 6: Distance transform
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Step 7: Final enhanced image
    final_enhanced = cv2.bitwise_or(morph_gradient, structural_elements)
    
    # Cleaning up with morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_enhanced = cv2.morphologyEx(final_enhanced, cv2.MORPH_CLOSE, kernel)
    
    # Create visualization of all steps
    plt.figure(figsize=(25, 12))
    
    images = [
        ("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        ("Grayscale", gray),
        ("Contrast Enhanced", contrast_enhanced),
        ("Horizontal Edge Detection", sobelx),
        ("Vertical Edge Detection", sobely),
        ("Combined Sobel Edges", sobel_combined),
        ("Horizontal Line Enhancement", horizontal_lines),
        ("Vertical Line Enhancement", vertical_lines),
        ("Structural Elements", structural_elements),
        ("Edge Detection", edges),
        ("Morphological Gradient", morph_gradient),
        ("Distance Transform", dist_transform),
        ("Final Enhanced Rectangles", final_enhanced)
    ]
    
    # Plot all images
    for i, (title, img) in enumerate(images):
        plt.subplot(4, 4, i+1)
        if len(img.shape) == 3:  # Color image
            plt.imshow(img)
        else:  # Grayscale image
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize rectangular structure enhancement steps")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save visualization (optional)")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Visualize the enhancement process
    visualize_rectangular_enhancement(args.image, args.output)

if __name__ == "__main__":
    main()