from pathlib import Path
import rasterio
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer
import xml.etree.ElementTree as ET


# ==========================================================
# INPUT / OUTPUT
# ==========================================================

geotiff_path = Path("data") / "tile_1_16.tif"

output_path = Path("demo_forest_boundary_1_16.kml")


# ==========================================================
# OPEN GEOTIFF
# ==========================================================

with rasterio.open(geotiff_path) as src:

    crs = src.crs
    raster_transform = src.transform

    width = src.width
    height = src.height

    print("GeoTIFF size:", width, "x", height)
    print("GeoTIFF CRS:", crs)


if crs is None:
    raise ValueError("GeoTIFF has no CRS.")


# ==========================================================
# DEFINE DEMO AOI USING PIXEL COORDINATES
# ==========================================================
#
# These points create an irregular polygon around
# the more wooded left-hand portion of the image.
#
# Pixel coordinate format:
#
# (column, row)
#
# Top-left = (0, 0)
# Bottom-right = (500, 500)
#
# You can adjust these later.
# ==========================================================

pixel_points = [

    (10, 10),

    (180, 10),

    (220, 55),

    (205, 115),

    (170, 160),

    (155, 230),

    (135, 310),

    (150, 400),

    (125, 490),

    (10, 490),

]


# ==========================================================
# PIXEL -> MAP COORDINATES
# ==========================================================

map_points = []

for col, row in pixel_points:

    x, y = raster_transform * (
        col,
        row
    )

    map_points.append(
        (
            x,
            y,
        )
    )


forest_polygon = Polygon(
    map_points
)


if not forest_polygon.is_valid:

    forest_polygon = (
        forest_polygon.buffer(0)
    )


print(
    "Demo AOI area:",
    forest_polygon.area,
    "map units²",
)


# ==========================================================
# CONVERT TO WGS84 FOR KML
# ==========================================================

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True,
)


polygon_wgs84 = transform(
    transformer.transform,
    forest_polygon,
)


# ==========================================================
# KML COORDINATES
# ==========================================================

coordinates = []

for longitude, latitude in (
    polygon_wgs84.exterior.coords
):

    coordinates.append(
        f"{longitude},{latitude},0"
    )


coordinate_string = " ".join(
    coordinates
)


# ==========================================================
# CREATE KML
# ==========================================================

namespace = (
    "http://www.opengis.net/kml/2.2"
)

ET.register_namespace(
    "",
    namespace,
)


kml = ET.Element(
    f"{{{namespace}}}kml"
)


document = ET.SubElement(
    kml,
    f"{{{namespace}}}Document"
)


document_name = ET.SubElement(
    document,
    f"{{{namespace}}}name"
)

document_name.text = (
    "CanopyLens Demo Forest Boundary"
)


placemark = ET.SubElement(
    document,
    f"{{{namespace}}}Placemark"
)


placemark_name = ET.SubElement(
    placemark,
    f"{{{namespace}}}name"
)

placemark_name.text = (
    "Demo Forest AOI"
)


description = ET.SubElement(
    placemark,
    f"{{{namespace}}}description"
)

description.text = (
    "Synthetic demonstration AOI created "
    "for validating CanopyLens boundary analysis. "
    "Not a surveyed forest boundary."
)


polygon = ET.SubElement(
    placemark,
    f"{{{namespace}}}Polygon"
)


outer = ET.SubElement(
    polygon,
    f"{{{namespace}}}outerBoundaryIs"
)


ring = ET.SubElement(
    outer,
    f"{{{namespace}}}LinearRing"
)


coords = ET.SubElement(
    ring,
    f"{{{namespace}}}coordinates"
)


coords.text = coordinate_string


# ==========================================================
# SAVE
# ==========================================================

tree = ET.ElementTree(
    kml
)


try:

    ET.indent(
        tree,
        space="    ",
    )

except AttributeError:
    pass


tree.write(
    output_path,
    encoding="utf-8",
    xml_declaration=True,
)


print()
print("====================================")
print("DEMO KML CREATED")
print("====================================")

print(
    "Input GeoTIFF:",
    geotiff_path,
)

print(
    "Output KML:",
    output_path,
)

print()
print(
    "This KML is a synthetic demo AOI, "
    "not a surveyed forest boundary."
)