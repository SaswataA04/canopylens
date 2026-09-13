from deepforest import main
import rasterio
import numpy as np
from PIL import Image
import os

GEOTIFF_PATH = "data/tile_0_0.tif"

print("Loading model...")

model = main.deepforest()

model.load_model(
    model_name="weecology/deepforest-tree",
    revision="main"
)

print("Reading GeoTIFF...")

with rasterio.open(GEOTIFF_PATH) as src:

    rgb_array = src.read([1, 2, 3])

    pixel_width = abs(src.transform.a)
    pixel_height = abs(src.transform.e)

rgb_array = np.transpose(rgb_array, (1, 2, 0))

image = Image.fromarray(
    rgb_array.astype(np.uint8),
    mode="RGB"
)

temp_path = "threshold_temp.png"
image.save(temp_path)

print("Running prediction...")

predictions = model.predict_image(
    path=temp_path
)

print("\n========== THRESHOLD ANALYSIS ==========")

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:

    filtered = predictions[
        predictions["score"] >= threshold
    ].copy()

    filtered["width_px"] = (
        filtered["xmax"] - filtered["xmin"]
    )

    filtered["height_px"] = (
        filtered["ymax"] - filtered["ymin"]
    )

    filtered["area_px"] = (
        filtered["width_px"] * filtered["height_px"]
    )

    area_m2 = (
        filtered["area_px"].sum()
        * pixel_width
        * pixel_height
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Trees: {len(filtered):2d} | "
        f"Estimated crown bounding-box area: {area_m2:8.2f} m²"
    )

print("========================================")

if os.path.exists(temp_path):
    os.remove(temp_path)