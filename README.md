# CanopyLens

**AI-powered tree crown detection and approximate canopy-area analysis from georeferenced forest imagery.**

CanopyLens is a Streamlit-based geospatial analysis tool that detects individual tree crowns from high-resolution GeoTIFF imagery using **DeepForest**, optionally restricts the analysis to a **KML forest boundary**, and estimates the approximate canopy area covered by the detected trees.


> **Important:** CanopyLens reports canopy area as an approximation derived from tree-detection bounding boxes. It does not perform pixel-level crown segmentation and should not be treated as ground-truth canopy measurement.

---

## Live Demo

**Streamlit Demo:** (https://saswataa04-canopylens-main-app-gizouf.streamlit.app/)


<img width="1919" height="925" alt="Screenshot 2026-09-13 173957" src="https://github.com/user-attachments/assets/806c69b5-d1f7-435b-8222-22ca33232bd9" />


---

## Problem

Forest monitoring often involves multiple disconnected steps:

* viewing satellite or aerial imagery,
* identifying individual trees,
* defining forest/project boundaries,
* calculating spatial statistics,
* and manually interpreting model outputs.

CanopyLens combines these steps into a simple workflow that another user can operate directly from a browser.

---

##  Features

###  Individual Tree Crown Detection

CanopyLens uses the pretrained **DeepForest tree detection model** to identify individual tree crowns in high-resolution imagery.

Each detection includes:

* bounding-box coordinates,
* confidence score,
* tree ID,
* AOI inclusion status,
* and approximate ground area.

---

### GeoTIFF Support

Users upload georeferenced `.tif` or `.tiff` imagery.

CanopyLens reads:

* image dimensions,
* raster bands,
* coordinate reference system (CRS),
* affine transform,
* raster bounds,
* and ground coordinates.

The geospatial metadata is preserved for spatial calculations.

---

###  Optional KML Forest Boundary

A `.kml` polygon can be uploaded to define the forest or project Area of Interest (AOI).

CanopyLens:

1. parses the KML polygon,
2. interprets the KML coordinates as WGS84,
3. reprojects the boundary into the GeoTIFF CRS,
4. checks whether it overlaps the imagery,
5. clips the analysis to the overlapping area.

If no KML is supplied, the entire GeoTIFF is analysed.

---

###  Adjustable Confidence Threshold

Users can interactively change the detection confidence threshold.

A lower threshold may identify more possible trees but can also introduce additional false positives.

A higher threshold produces fewer, more confident detections.

This allows users to inspect how sensitive the final tree count and canopy estimate are to model confidence.

---

###  Confidence-Aware Visualization

Detected trees are displayed using confidence-aware bounding boxes.

Higher and lower confidence detections can be visually distinguished, while detections excluded from the AOI can also be shown separately.

Optional confidence labels make individual predictions easier to inspect.

---

###  Threshold Sensitivity Analysis

CanopyLens recalculates tree count and canopy estimates across different confidence thresholds without rerunning the DeepForest model.

This makes it possible to understand how stable the reported results are rather than presenting a single number without context.

---

###  Detection Analytics

The application provides additional analysis of the model predictions, including:

* threshold sensitivity,
* confidence distribution,
* detection-area distribution,
* included vs excluded detections,
* and model prediction statistics.

These views help users understand **why the reported result changes**, not just what the final number is.

---

###  Analysis Audit Trail

CanopyLens exposes the steps used to transform model predictions into the final result.

Conceptually:

```text
Raw DeepForest predictions
        ↓
Confidence threshold filtering
        ↓
AOI inclusion filtering
        ↓
Bounding boxes clipped to AOI
        ↓
Overlapping detection areas merged
        ↓
Final tree count + approximate canopy area
```

This makes the analysis more transparent and helps avoid presenting AI-generated measurements as unquestionable ground truth.

---

##  How Tree Inclusion Works

For each detected tree, CanopyLens calculates the geographic position of the **centre of the detection box**.

A tree is counted inside the analysis region when:

```text
Detection centre ∈ AOI
```

If its centre lies outside the AOI, that detection is excluded from the final tree count.

This rule provides a deterministic way of handling trees whose bounding boxes intersect the forest boundary.

---

##  How Canopy Area Is Estimated

DeepForest produces rectangular tree-crown detections rather than exact crown segmentation masks.

Therefore, CanopyLens uses the geographic footprint of each detection box as a **canopy-area proxy**.

For every included detection:

```text
Pixel bounding box
      ↓
GeoTIFF affine transform
      ↓
Ground-coordinate polygon
      ↓
Clip polygon to AOI
      ↓
Merge overlapping polygons
      ↓
Approximate canopy area
```

Overlapping bounding boxes are geometrically merged before calculating the final area, preventing overlapping regions from being counted multiple times.

For a metric projected CRS:

```text
Canopy Coverage (%) =
Approximate Canopy Area / AOI Area × 100
```

---

##  Architecture

```text
GeoTIFF
   │
   ├── Rasterio → imagery + geospatial metadata
   │
   └── RGB image
           │
           ▼
      DeepForest
           │
           ▼
    Tree detections
           │
           ├── Confidence filtering
           │
KML ───────┼── AOI filtering
           │
           ▼
    Geospatial conversion
           │
           ▼
      Shapely geometry
           │
           ├── AOI clipping
           ├── overlap removal
           └── area calculation
           │
           ▼
       CanopyLens UI
           │
           ├── Tree count
           ├── Canopy estimate
           ├── Detection map
           └── Analytics
```

---

##  Tech Stack

| Technology      | Purpose                     |
| --------------- | --------------------------- |
| Python          | Core application            |
| Streamlit       | Interactive web application |
| DeepForest      | Tree crown detection        |
| Rasterio        | GeoTIFF processing          |
| Shapely         | Spatial geometry operations |
| PyProj          | CRS transformations         |
| Pandas          | Detection data processing   |
| NumPy           | Numerical operations        |
| Matplotlib      | Analytics and visualization |
| Pillow / OpenCV | Image processing            |

---

##  Project Structure

```text
canopylens/
│
├── main_app.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── threshold_test.py
│   └── Confidence-threshold experimentation
│
├── generate_demo_kml.py
│   └── Utility for generating demo boundaries
│
├── demo_forest_boundary.kml
├── demo_forest_boundary_1_16.kml
├── forest_boundary.kml
│   └── Example KML boundaries
│
└── README.md
```

---

##  Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd canopylens
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start CanopyLens

```bash
streamlit run main_app.py
```

Streamlit will open the application in your browser.

---

##  Using CanopyLens

1. Open the application.
2. Upload a georeferenced GeoTIFF.
3. Optionally upload a KML forest boundary.
4. Select the desired confidence threshold.
5. Allow DeepForest to detect tree crowns.
6. Inspect the detected trees and AOI.
7. Review tree count and approximate canopy coverage.
8. Explore the analytics to understand confidence and threshold sensitivity.

---

##  Limitations

CanopyLens intentionally exposes its limitations rather than presenting its estimates as exact measurements.

### Bounding boxes are not true crown segmentation

DeepForest predicts rectangular bounding boxes around tree crowns.

Therefore:

**CanopyLens estimates canopy area from bounding-box footprints, not the exact biological crown shape.**

This means canopy area may be overestimated in some situations.

### Model accuracy depends on imagery

Detection performance can vary with:

* image resolution,
* forest type,
* tree density,
* crown overlap,
* illumination,
* seasonal conditions,
* and similarity to DeepForest's training data.

### Confidence threshold affects results

There is no universally correct confidence threshold.

Lower thresholds may increase false positives, while higher thresholds may miss valid trees.

This is why CanopyLens includes threshold sensitivity analysis.

### Dense forests remain challenging

When neighbouring tree crowns strongly overlap, an object detector may:

* merge multiple trees,
* miss partially visible crowns,
* or incorrectly split a crown.

### Area calculations require appropriate geospatial information

Reliable physical-area calculations require valid georeferencing and an appropriate projected CRS with metric units.

### Not a carbon-credit measurement system

CanopyLens provides exploratory tree-crown and canopy estimates.

It is **not intended to independently certify carbon stocks, biomass, forest inventories, or carbon credits**.

---



---

##  Design Philosophy

The goal of CanopyLens is not simply to produce a tree count.

It is designed to make the path from:

```text
imagery → model prediction → spatial filtering → measurement
```

visible to the user.

For environmental and carbon-market applications, knowing **how uncertain a measurement is and how it was produced** can be as important as the measurement itself.

---

##  Author

** Created by Saswata Acharya**



---

##  CanopyLens

**Detect. Measure. Inspect.**

AI-assisted forest canopy analysis with transparent assumptions and explicit limitations.
