import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, exposure, feature
from skimage.filters import sato
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import disk, dilation, erosion, opening, closing
from skimage.draw import polygon, line
from scipy import ndimage as ndi
import cv2
from skimage.morphology import closing, disk

def detect_rail_stuff(image_path, 
                      otsu_thr= 0.3, 
                      min_sato = 50,
                      min_rail_line = 50, 
                      debug=True, 
                      do_adaptive=True):
    """
    Rail bed detection using Sato filter for tubular structures and Hough transform
    for line detection, then fitting rectangles between parallel lines
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.copy()
    
    # Step 1: Enhance contrast
    p2, p98 = np.percentile(gray, (2, 98))
    img_contrast = exposure.rescale_intensity(gray, in_range=(p2, p98))

    ref_img_contrast = img_contrast.copy()

    # Step 2: Apply Sato filter to detect tubular structures (rails)
    sato_result = sato(img_contrast, sigmas=(0.5, 1.0, 1.5), black_ridges=False)
    sato_normalized = (exposure.rescale_intensity(sato_result)*255).astype(np.uint8)

    ref_sato_normalized = sato_normalized.copy()

    # Step 3: Otsu Binarization of Normalized Sato
    p=otsu_thr
    _, greyscale_otsu = cv2.threshold(sato_normalized, int(p*255), 255, cv2.THRESH_BINARY_INV)
    greyscale_otsu = 255-greyscale_otsu

    ref_otsu = greyscale_otsu.copy()

    # Step 4: Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(greyscale_otsu, connectivity=8)
    
    # Step 4.1: Aaptively finding the minimum area of the connected components
    if do_adaptive:
        areas = [stats[label, cv2.CC_STAT_AREA] for label in range(1, num_labels)]

        if areas:  # Check if there are any components
            # Use a percentile of the area distribution
            areas.sort()
            min_area = np.percentile(areas, 90)  # Filter out the smallest 90%
        else:
            min_area = min_sato  # Default if no components
    else:
        min_area = min_sato

    filtered = np.zeros_like(greyscale_otsu)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    greyscale_otsu = filtered

    pruned_sato_otsu = np.copy(greyscale_otsu)


    # Step 5: Morphological operations to dilate the rail lines
    ker = np.ones((5,5),np.uint8)
    greyscale_otsu = cv2.dilate(greyscale_otsu, ker, iterations=1)

    # Step 6: Convert the otsu image to binary to get rail lines
    greyscale_otsu = greyscale_otsu.astype(np.uint8)

    # Step 6.1: Further prune small connected componets of raw rail lines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(greyscale_otsu, connectivity=8)
    areas = [stats[label, cv2.CC_STAT_AREA] for label in range(1, num_labels)]

    if do_adaptive:
        if areas:  # Check if there are any components
            # Use a percentile of the area distribution
            areas.sort()
            min_area = np.percentile(areas, 90)  # Filter out the smallest 90%
        else:
            min_area = min_rail_line
    else:
        min_area = min_rail_line

    filtered = np.zeros_like(greyscale_otsu)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    greyscale_otsu = filtered

    secondary_pruned_sato_otsu = np.copy(greyscale_otsu)

    # Save the rail lines
    rail_lines = np.copy(greyscale_otsu)
    # Assign rail lines as pixel value of 1
    rail_lines [greyscale_otsu == 255] = 1

    greyscale_otsu_r = greyscale_otsu.copy()


    # Step 7: Use dilation and erosion to close the rail lines to get raw rail bed between rail lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    dilated = cv2.dilate(greyscale_otsu, kernel, iterations=5)
    connected = cv2.erode(dilated, kernel, iterations=5)

    ref_raw_rail_bed = connected.copy()


    # Step 8: Identify the filled gaps: where closing resulted in 255 but the original was 0
    gap_fill = (connected == 255) & (greyscale_otsu_r == 0)
    # Step 9: Identify the union of the rail bed and the filled gaps
    union_fill = (connected == 255) | (greyscale_otsu_r == 255)


    # Step 10: Remove small connected components from the rail bed (these are areas that go beyond the rail line)
    rail_closed = np.zeros_like(greyscale_otsu)
    rail_closed[gap_fill] = 255   
    greyscale_otsu = np.copy(rail_closed)
    num_labels_1, labels_1 = cv2.connectedComponents(rail_closed, connectivity=8)
    for i in range(num_labels_1):
        # find IOU score of raw rail_bed and connected components
        intersection = np.sum(union_fill[labels_1 == i])
        union = np.sum(union_fill)
        iou = intersection / union
        if iou < 0.5:
            greyscale_otsu[labels_1 == i] = 0

    # Step 11: Final rail bed
    rail_bed = greyscale_otsu
    refined_raw_rail_bed = np.copy(rail_bed)

    # Assign rail bed as pixel value of 2
    rail_bed [rail_bed == 255] = 2

    # Step 12: Get both rail lines and rail bed as total rail
    total_rail = cv2.add(rail_lines, rail_bed)

    overlay = img.copy() if len(img.shape) == 3 else color.gray2rgb(gray)
    overlay[ total_rail == 1 ] = [ 255, 0, 0]   # rail line
    overlay[ total_rail == 2 ] = [ 0, 255, 0]    # rail bed


    if debug:
        # plot: ref_img_contrast, ref_sato_normalized , ref_otsu, pruned_sato_otsu, 
        # secondary_pruned_sato_otsu, ref_raw_rail_bed, refined_raw_rail_bed, overlay
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))

        # Row 1
        ax[0, 0].imshow(ref_img_contrast, cmap='gray')
        ax[0, 0].set_title("Contrast Enhanced Image")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(ref_sato_normalized, cmap='gray')
        ax[0, 1].set_title("Sato Filtered Image")
        ax[0, 1].axis('off')

        ax[0, 2].imshow(ref_otsu, cmap='gray')
        ax[0, 2].set_title("Otsu Binarization of Sato")
        ax[0, 2].axis('off')

        ax[0, 3].imshow(pruned_sato_otsu, cmap='gray')
        ax[0, 3].set_title("Pruned Sato Otsu")
        ax[0, 3].axis('off')

        # Row 2
        ax[1, 0].imshow(secondary_pruned_sato_otsu, cmap='gray')
        ax[1, 0].set_title("Secondary Pruned Sato Otsu")
        ax[1, 0].axis('off')

        ax[1, 1].imshow(ref_raw_rail_bed, cmap='gray')
        ax[1, 1].set_title("Raw Rail Bed")
        ax[1, 1].axis('off')

        ax[1, 2].imshow(refined_raw_rail_bed, cmap='gray')
        ax[1, 2].set_title("Refined Raw Rail Bed")
        ax[1, 2].axis('off')

        ax[1, 3].imshow(overlay)
        ax[1, 3].set_title("Total Rail")
        ax[1, 3].axis('off')

        plt.tight_layout()
        plt.show()

        cv2.imwrite("segmented.png", overlay)

    


if __name__ == "__main__":
    # image_path = "2.png"
    # image_path = "4.png"
    # image_path = "preprocessed_ir_orthoimage.png"
    image_path = "preprocessed_orthoimage.png"
    detect_rail_stuff(image_path, 
                      otsu_thr= 0.3, 
                      min_sato = 50, 
                      min_rail_line = 50,
                      debug=True, 
                      do_adaptive=False
                      )