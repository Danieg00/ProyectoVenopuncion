"""
ChArUco board generator for printing.
Generates a printable ChArUco (Checkerboard ARuco) board image.
Supports standard paper sizes: A4, A3, Letter, Tabloid
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
from utils.config import BOARD_CONFIG


# Paper size definitions (width x height in pixels at 300 DPI)
PAPER_SIZES = {
    "A4": (2480, 3508),      # 210x297mm at 300 DPI - Standard worldwide
    "A3": (3508, 4961),      # 297x420mm at 300 DPI - Larger, easier to detect
}

DEFAULT_PAPER_SIZE = "A3"  # Recommended for easy detection


def generate_charuco_board(output_path: Union[str, Path] = "charuco_board.png", 
                          paper_size: str = "A3",
                          display: bool = True):
    """
    Generate a printable ChArUco board image.
    
    Args:
        output_path: Where to save the board image (PNG or PDF)
        paper_size: Paper size ('A4', 'A3', 'Letter', 'Tabloid')
        display: If True, display the board in a window
    
    RECOMMENDATION:
        - Use A3 (297x420mm) for best detection at distance
        - A4 (210x297mm) works but board must be closer to camera
        - Print on glossy cardstock and mount on rigid surface
    """
    
    # Validate paper size
    if paper_size not in PAPER_SIZES:
        print(f"ERROR: Unknown paper size '{paper_size}'")
        print(f"Available sizes: {', '.join(PAPER_SIZES.keys())}")
        return
    
    image_size = PAPER_SIZES[paper_size]
    
    # Get board configuration
    squares_x = BOARD_CONFIG["squares_x"]
    squares_y = BOARD_CONFIG["squares_y"]
    square_length = BOARD_CONFIG["square_length"]  # meters
    marker_length = BOARD_CONFIG["marker_length"]  # meters
    
    # Get dictionary
    dict_name = BOARD_CONFIG["dictionary"]
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dict_name)
    )
    
    # Create ChArUco board object
    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=dictionary
    )
    
    # Generate board image
    print(f"Generating ChArUco board: {squares_x}x{squares_y} squares")
    print(f"  Square length: {square_length*100:.1f} cm")
    print(f"  Marker length: {marker_length*100:.1f} cm")
    print(f"  Paper size: {paper_size} {image_size}")
    
    board_image = board.generateImage(
        outSize=image_size,
        marginSize=50,  # Margin in pixels
        borderBits=1    # Border thickness in bits (markers use 1 bit)
    )
    
    # Save board image
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(str(output_file), board_image)
    
    if success:
        print(f"\n✓ Board saved to: {output_file}")
        print(f"  Resolution: {board_image.shape[1]} x {board_image.shape[0]} pixels")
        print(f"  Print at 100% scale (NO SCALING)")
        print(f"\nPRINTING INSTRUCTIONS:")
        print(f"  1. Open {output_path}")
        print(f"  2. Print at 100% scale - CRITICAL for size accuracy")
        print(f"  3. Paper: Glossy cardstock or photo paper (NOT matte)")
        print(f"  4. Mount firmly on wood/plastic/cardboard (must be rigid)")
        print(f"  5. Keep clean and dry (avoid fingerprints/reflections)")
        print(f"  6. If printing from browser:")
        print(f"     - Disable 'Fit to page'")
        print(f"     - Disable 'Shrink to fit'")
        print(f"     - Set margins to 0")
    else:
        print(f"ERROR: Failed to save board image")
        return
    
    # Display if requested
    if display:
        # Resize for display if too large
        display_size = (1280, int(1280 * board_image.shape[0] / board_image.shape[1]))
        display_image = cv2.resize(board_image, display_size)
        
        cv2.imshow(f"ChArUco Board - {paper_size} (Press any key to close)", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return board_image


def generate_charuco_pdf(output_path: Union[str, Path] = "charuco_board.pdf") -> bool:
    """
    Generate ChArUco board as PDF for printing.
    (Requires reportlab: pip install reportlab)
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import Image as RLImage
        from PIL import Image
        import io
        
        # First generate the PNG image
        board_image = generate_charuco_board(display=False)
        
        if board_image is None:
            print("ERROR: Failed to generate board image")
            return False
        
        # Convert to PIL Image
        pil_image = Image.fromarray(board_image)
        
        # Create PDF
        import tempfile
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        c = canvas.Canvas(str(output_file), pagesize=A4)
        width, height = A4
        
        # Add image centered on page
        img_width = width * 0.95
        img_height = (pil_image.size[1] / pil_image.size[0]) * img_width
        
        x = (width - img_width) / 2
        y = (height - img_height) / 2
        
        # Create temporary file for image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pil_image.save(tmp.name)
            
            # Draw image on PDF
            c.drawImage(tmp.name, x, y, width=img_width, height=img_height)
            c.save()
            
            # Clean up temp file
            import os
            os.unlink(tmp.name)
        
        print(f"\n✓ PDF saved to: {output_file}")
        return True
        
    except ImportError as e:
        print(f"\nERROR: Missing dependencies for PDF generation: {e}")
        print("  pip install reportlab pillow")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to generate PDF: {e}")
        return False


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate printable ChArUco board")
    parser.add_argument(
        "--output",
        type=str,
        default="charuco_board.png",
        help="Output file path (PNG)"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=list(PAPER_SIZES.keys()),
        default=DEFAULT_PAPER_SIZE,
        help=f"Paper size (default: {DEFAULT_PAPER_SIZE})"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display preview"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ChArUco Board Generator")
    print("="*60 + "\n")
    
    generate_charuco_board(
        output_path=args.output,
        paper_size=args.size,
        display=not args.no_display
    )
