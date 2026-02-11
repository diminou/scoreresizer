# Scoreresizer - AI Agent Guide

## Project Overview

Scoreresizer is a Python CLI tool for processing scanned PDF documents (primarily sheet music scores). It provides image manipulation capabilities including auto-cropping, rotation, rescaling, and collation onto larger paper formats.

**Project Name:** Scoreresizer  
**License:** GNU General Public License v3.0 (GPL-3.0)  
**Copyright:** 2026 Dmitry Ivanov  

## Technology Stack

- **Language:** Python 3
- **Core Dependencies:**
  - `opencv-python-headless` (4.13.0.90) - Computer vision for margin cropping
  - `Pillow` (12.1.0) - Image processing and PDF generation
  - `pdf2image` (1.17.0) - PDF to image conversion
  - `numpy` (2.4.2) - Array operations for image data
  - `click` - CLI interface framework

## External System Dependencies

The project requires **Poppler** to be installed on the system for PDF processing:
- **Ubuntu/Debian:** `apt-get install poppler-utils`
- **macOS:** `brew install poppler`
- **Windows:** Download from https://github.com/oschwartz10612/poppler-windows/releases/

## Project Structure

```
/mnt/c/Users/divanov/projects/scoreresizer/
├── scan_processor.py    # Main application code (264 lines)
├── requirements.txt     # Python dependencies
├── LICENSE             # GPL-3.0 license file
├── __init__.py         # Package marker (empty)
└── .gitignore          # Git ignore rules (*.pyc, __pycache__)
```

**Note:** This is a single-file application with no test suite currently implemented.

## Code Organization

The `scan_processor.py` file is organized into three main sections:

1. **Constants & Geometry** (lines 8-18)
   - `get_dimensions_mm()` - Paper size dimensions (A4, A3)
   - `mm_to_px()` - Millimeter to pixel conversion

2. **Image Processing Functions** (lines 21-185)
   - `crop_margins_opencv()` - Auto-detect content and crop margins
   - `rotate_90_ccw()` - Rotate images 90 degrees
   - `rescale_to_a4()` - Fit images to A4 dimensions
   - `create_blank_a4()` - Generate blank A4 pages
   - `collate_to_a3()` - Arrange A4 pages onto A3 sheets

3. **CLI Interface** (lines 189-264)
   - `main()` - Click-based command-line interface

## Build and Installation

No build process is required. To set up:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependency (Poppler) - example for Ubuntu
sudo apt-get install poppler-utils
```

## Usage

```bash
python scan_processor.py <input_pdf> <output_pdf> [OPTIONS]
```

### Available Options

| Option | Description |
|--------|-------------|
| `--dpi` | DPI for processing and output (default: 300) |
| `--crop` | Auto-crop margins using OpenCV |
| `--aggressive-crop` | Stronger filtering for edge artifacts (requires `--crop`) |
| `--rotate` | Rotate 90 degrees clockwise |
| `--rescale-a4` | Rescale pages to fit A4 optimally |
| `--collate-a3` | Collate pages onto A3 sheets |
| `--start-right` | Booklet-style collation (4,1 then 2,3) |

### Usage Examples

```bash
# Basic cropping
python scan_processor.py input.pdf output.pdf --crop

# Full processing pipeline
python scan_processor.py input.pdf output.pdf --crop --rescale-a4 --collate-a3

# Booklet-style collation
python scan_processor.py input.pdf output.pdf --crop --collate-a3 --start-right

# Aggressive crop for noisy scans
python scan_processor.py input.pdf output.pdf --crop --aggressive-crop
```

## Code Style Guidelines

- **Function naming:** snake_case (e.g., `crop_margins_opencv`)
- **Constants:** UPPER_CASE in function defaults
- **Comments:** Use inline comments for algorithm explanations
- **Docstrings:** Google-style docstrings for function documentation
- **Imports:** Standard library first, then third-party (numpy, cv2, PIL, click)

## Key Implementation Details

### Margin Cropping Algorithm
1. Convert to grayscale and apply Otsu thresholding
2. Apply morphological operations (opening for aggressive mode, dilation for standard)
3. Find contours and filter by area (0.02% for aggressive, 0.01% for standard)
4. Calculate bounding box with configurable padding

### A3 Collation Logic
- Processes pages in groups of 4
- Default layout: (1,2) on sheet 1, (3,4) on sheet 2
- Booklet layout (`--start-right`): (4,1) on sheet 1, (2,3) on sheet 2
- Automatically pads with blank pages if input count is not divisible by 4

### Image Dimensions
- **A4:** 210mm x 297mm
- **A3:** 297mm x 420mm
- Conversion: `px = mm / 25.4 * dpi`

## Testing

**No automated tests are currently implemented.** Testing is done manually by running the CLI with sample PDF files.

## Security Considerations

- Input PDF paths are validated using Click's `Path(exists=True)` type
- Output paths are not validated (allows creating new files)
- No sandboxing of PDF processing - relies on pdf2image/Pillow security
- Temporary images are handled in memory, not written to disk

## Development Notes

- The project uses headless OpenCV (`opencv-python-headless`) for server environments
- Image processing is done entirely in memory; large PDFs may require significant RAM
- No logging framework is used; status messages are printed via `click.echo()`
