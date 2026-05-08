#Self‑Calibrated OMR
A flexible, self‑calibrating Optical Mark Recognition (OMR) system for detecting bubble grids in answer sheets.
The pipeline estimates bubble size, grid spacing, and grid rotation directly from the image — making it robust to noise, scale variation, and misalignment.

#Approaches
1. Using Expected Bubble Size
Contours are filtered using geometric features and a known bubble size range. This works well for clean scans or controlled layouts.

2. Without Expected Bubble Size (Isolation Forest)
For more challenging images, bubble candidates are selected using Isolation Forest and clustering.
This approach handles noisy, low‑quality, or unpredictable scans more effectively.

#Pipeline Overview
Regardless of the chosen contour‑selection method, the core processing steps are:

1. Preprocess the image and extract contours

2. Convert to grayscale, blur, threshold, and find contours suitable for analysis.

3. Estimate initial grid spacing

4. Use the distribution and dimensions of bubble‑like contours to estimate horizontal and vertical spacing.

5. Estimate grid deviation angle

6. A randomized search (“randomized game”) is used to find the rotation angle of the bubble grid with respect to the image axes.

7. Detect columns and rows

8. Using the updated grid spacing and rotation, bubbles are grouped into coherent rows and columns.

9. Identify blocks of bubbles

10. Rows and columns are organized into blocks that correspond to answer regions or question groups on the sheet.

#Key Features
-Self‑calibrating grid (size + angle)
-Works with or without known bubble size
-Robust to noisy scans and image distortions
-Template‑independent bubble grouping
-Modular components for preprocessing, contour filtering, spacing estimation, and block detection

#Use Cases
-Multiple‑choice exam sheets
-Template‑free or loosely formatted answer forms
-Scanned or photographed sheets with rotation or scale variation
-Research on robust, data‑driven OMR pipelines

#License
MIT 

#Installation
Clone the repository and install locally:

git clone https://github.com/yazdannejat/selfcal-omr.git
cd selfcal-omr
pip install -e .

#Dependencies:

pip install opencv-python numpy Pillow 