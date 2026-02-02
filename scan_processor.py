import cv2
import numpy as np
import click
import math
from PIL import Image
from pdf2image import convert_from_path

# --- Constants & Geometry ---

def get_dimensions_mm(format_name):
    if format_name == 'a4':
        return 210, 297
    elif format_name == 'a3':
        return 297, 420
    return 0, 0

def mm_to_px(mm_w, mm_h, dpi):
    return (int(mm_w / 25.4 * dpi), int(mm_h / 25.4 * dpi))

# --- Image Processing Functions ---

def crop_margins_opencv(pil_img, padding=10):
    """
    Detects the content area using OpenCV and crops the image.
    Adds a small padding to avoid cutting off thin text edges.
    """
    # Convert PIL to OpenCV format (RGB)
    img = np.array(pil_img)
    
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Binary threshold (assuming white background, darker text/artifacts)
    # Otsu's thresholding automatically finds the best threshold value
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate slightly to merge nearby noise/text into a block
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return pil_img # No content found, return original

    # Get the bounding box of all content combined
    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Apply padding
    h_img, w_img = img.shape[:2]
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w_img, x_max + padding)
    y_max = min(h_img, y_max + padding)
    
    # Crop using PIL
    cropped = pil_img.crop((x_min, y_min, x_max, y_max))
    return cropped

def rotate_90_ccw(pil_img):
    return pil_img.transpose(Image.ROTATE_90)

def rescale_to_a4(pil_img, dpi=300):
    """
    Rescales the image to fit within A4 dimensions, maintaining aspect ratio.
    """
    a4_w_mm, a4_h_mm = get_dimensions_mm('a4')
    a4_w_px, a4_h_px = mm_to_px(a4_w_mm, a4_h_mm, dpi)
    
    img_w, img_h = pil_img.size
    
    # Calculate ratios
    ratio_w = a4_w_px / img_w
    ratio_h = a4_h_px / img_h
    
    # Use the smaller ratio to ensure the image fits inside (contain)
    scale_factor = min(ratio_w, ratio_h)
    
    # If image is smaller than A4, we usually don't upscale for scans to avoid blur,
    # but the prompt implies "fit onto", which can mean upscaling. 
    # We will upscale if scale_factor > 1 based on strict interpretation.
    if scale_factor != 1.0:
        new_size = (int(img_w * scale_factor), int(img_h * scale_factor))
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
    return pil_img

def create_blank_a4(dpi=300):
    w_mm, h_mm = get_dimensions_mm('a4')
    w_px, h_px = mm_to_px(w_mm, h_mm, dpi)
    return Image.new('RGB', (w_px, h_px), 'white')

def collate_to_a3(images, start_right=False, dpi=300):
    """
    Collates a list of A4 images onto A3 pages.
    logic:
    - Default: 1,2 on pg1; 3,4 on pg2
    - Start Right: 4,1 on pg1; 2,3 on pg2 (Booklet style)
    """
    a3_w_mm, a3_h_mm = get_dimensions_mm('a3')
    a3_w_px, a3_h_px = mm_to_px(a3_w_mm, a3_h_mm, dpi)
    
    a4_w_px, a4_h_px = mm_to_px(*get_dimensions_mm('a4'), dpi)
    
    result_pages = []
    
    # Pad input list with blanks if necessary to complete the signature (groups of 4)
    # The logic specifically implies a 4-page cycle.
    total_inputs = len(images)
    
    # Calculate how many blank pages we need to make the length a multiple of 4
    remainder = total_inputs % 4
    if remainder != 0:
        padding_needed = 4 - remainder
        for _ in range(padding_needed):
            images.append(create_blank_a4(dpi))
            
    # Process in chunks of 4
    for i in range(0, len(images), 4):
        chunk = images[i:i+4]
        # Ensure we have exactly 4 items (fill missing with blank if input logic was weird)
        while len(chunk) < 4:
            chunk.append(create_blank_a4(dpi))
            
        p1, p2, p3, p4 = chunk
        
        # Determine Layout
        # A3 Landscape mode is usually: [Left A4] [Right A4]
        
        if start_right:
            # Page 4, 1 on page 1
            # Page 2, 3 on page 2
            sheet1_pairs = [(p4, p1)]
            sheet2_pairs = [(p2, p3)]
        else:
            # 1, 2 on page 1
            # 3, 4 on page 2
            sheet1_pairs = [(p1, p2)]
            sheet2_pairs = [(p3, p4)]
            
        all_pairs = sheet1_pairs + sheet2_pairs
        
        for left_img, right_img in all_pairs:
            a3_canvas = Image.new('RGB', (a3_w_px, a3_h_px), 'white')
            
            # Paste Left
            # Note: A3 width is double A4 width at same DPI
            a3_canvas.paste(left_img, (0, 0))
            
            # Paste Right
            a3_canvas.paste(right_img, (a4_w_px, 0))
            
            result_pages.append(a3_canvas)
            
    return result_pages

# --- CLI Interface ---

@click.command()
@click.argument('input_pdf', type=click.Path(exists=True))
@click.argument('output_pdf', type=click.Path())
@click.option('--dpi', default=300, help='DPI for processing and output (default: 300)')
@click.option('--crop', is_flag=True, help='Auto-crop margins using OpenCV')
@click.option('--rotate', is_flag=True, help='Rotate 90 degrees clockwise')
@click.option('--rescale-a4', is_flag=True, help='Rescale pages to fit A4 optimally')
@click.option('--collate-a3', is_flag=True, help='Collate pages onto A3 sheets')
@click.option('--start-right', is_flag=True, help='If collating, use booklet order (4,1 then 2,3)')
def main(input_pdf, output_pdf, dpi, crop, rotate, rescale_a4, collate_a3, start_right):
    """
    Process a scanned PDF with various image manipulation options.
    """
    if collate_a3 and not start_right:
        pass # Default order handled in logic
    if start_right and not collate_a3:
        click.echo("Warning: --start-right specified without --collate-a3. Ignoring --start-right.")

    click.echo(f"Processing {input_pdf}...")
    
    # 1. Load PDF
    try:
        images = convert_from_path(input_pdf, dpi=dpi)
    except Exception as e:
        click.echo(f"Error loading PDF. Is Poppler installed? {e}")
        return

    processed_images = []

    for i, img in enumerate(images):
        current_img = img
        
        # 2. Crop
        if crop:
            click.echo(f"Page {i+1}: Cropping...")
            current_img = crop_margins_opencv(current_img)
        
        # 3. Rotate
        if rotate:
            click.echo(f"Page {i+1}: Rotating...")
            current_img = rotate_90_ccw(current_img)
            
        # 4. Rescale A4
        if rescale_a4:
            click.echo(f"Page {i+1}: Rescaling to A4...")
            current_img = rescale_to_a4(current_img, dpi=dpi)
            
        processed_images.append(current_img)

    # 5. Collate A3
    final_output_images = processed_images
    if collate_a3:
        click.echo("Collating onto A3...")
        if start_right:
            click.echo("Using Start-Right layout...")
        final_output_images = collate_to_a3(processed_images, start_right=start_right, dpi=dpi)

    # 6. Save
    click.echo(f"Saving to {output_pdf}...")
    if final_output_images:
        # Save first page to init the PDF
        final_output_images[0].save(
            output_pdf, 
            save_all=True, 
            append_images=final_output_images[1:],
            resolution=dpi
        )
    else:
        click.echo("No pages to save.")

if __name__ == '__main__':
    main()
