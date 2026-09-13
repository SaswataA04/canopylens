import io
import os
import json
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st

import rasterio
from rasterio.io import MemoryFile

from PIL import Image, ImageDraw, ImageFont

from shapely.geometry import Polygon, Point
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

from pyproj import Transformer, CRS

from deepforest import main


# APP CONFIG

APP_NAME = "CanopyLens"
MODEL_NAME = "weecology/deepforest-tree"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM UI

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }

    .hero {
        padding: 24px 26px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background:
            linear-gradient(
                135deg,
                rgba(34, 197, 94, 0.12),
                rgba(20, 83, 45, 0.03)
            );
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.75;
    }

    .status-good {
        padding: 12px 16px;
        border-radius: 10px;
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.20);
    }

    .small-note {
        font-size: 0.84rem;
        opacity: 0.68;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# HERO

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🌳 CanopyLens</div>
        <div class="hero-subtitle">
            Detect individual tree crowns, analyse a forest boundary,
            and estimate approximate canopy coverage from georeferenced imagery.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "CanopyLens uses DeepForest crown detections. "
    "Area values are derived from detection bounding boxes and should "
    "be treated as estimates rather than true crown segmentation."
)


# MODEL

@st.cache_resource
def load_model():

    model = main.deepforest()

    model.load_model(
        model_name=MODEL_NAME,
        revision="main",
    )

    return model


# IMAGE HELPERS

def normalize_band(band):

    band = band.astype(np.float32)

    valid = band[np.isfinite(band)]

    if valid.size == 0:
        return np.zeros_like(
            band,
            dtype=np.uint8,
        )

    low = np.percentile(
        valid,
        2,
    )

    high = np.percentile(
        valid,
        98,
    )

    if high <= low:
        return np.zeros_like(
            band,
            dtype=np.uint8,
        )

    normalized = (
        band - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0,
        1,
    )

    return (
        normalized * 255
    ).astype(np.uint8)


def read_geotiff(tiff_bytes):

    with MemoryFile(
        tiff_bytes
    ) as memfile:

        with memfile.open() as src:

            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "dtype": str(src.dtypes[0]),
            }

            # -------------------------------
            # RGB
            # -------------------------------

            if src.count >= 3:

                data = src.read(
                    [1, 2, 3]
                )

            else:

                single_band = src.read(1)

                data = np.stack(
                    [
                        single_band,
                        single_band,
                        single_band,
                    ]
                )

            rgb = np.moveaxis(
                data,
                0,
                -1,
            )

            if rgb.dtype != np.uint8:

                output = np.zeros(
                    rgb.shape,
                    dtype=np.uint8,
                )

                for channel in range(3):

                    output[:, :, channel] = (
                        normalize_band(
                            rgb[:, :, channel]
                        )
                    )

                rgb = output

            return rgb, metadata


# KML PARSER

def parse_coordinate_string(text):

    if not text:
        return []

    coordinates = []

    for coordinate in text.strip().split():

        parts = coordinate.split(",")

        if len(parts) < 2:
            continue

        try:

            longitude = float(
                parts[0]
            )

            latitude = float(
                parts[1]
            )

            coordinates.append(
                (
                    longitude,
                    latitude,
                )
            )

        except ValueError:
            continue

    return coordinates


def parse_kml(kml_bytes):

    root = ET.fromstring(
        kml_bytes
    )

    polygons = []

    for polygon_element in root.findall(
        ".//{*}Polygon"
    ):

        outer_element = (
            polygon_element.find(
                ".//{*}outerBoundaryIs/"
                "{*}LinearRing/"
                "{*}coordinates"
            )
        )

        if outer_element is None:
            continue

        outer_coords = (
            parse_coordinate_string(
                outer_element.text
            )
        )

        if len(outer_coords) < 3:
            continue

        holes = []

        inner_elements = (
            polygon_element.findall(
                ".//{*}innerBoundaryIs/"
                "{*}LinearRing/"
                "{*}coordinates"
            )
        )

        for inner in inner_elements:

            coords = (
                parse_coordinate_string(
                    inner.text
                )
            )

            if len(coords) >= 3:
                holes.append(
                    coords
                )

        try:

            polygon = Polygon(
                outer_coords,
                holes=holes,
            )

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if not polygon.is_empty:
                polygons.append(
                    polygon
                )

        except Exception:
            continue

    if not polygons:

        raise ValueError(
            "No valid polygon geometry was found in this KML."
        )

    geometry = unary_union(
        polygons
    )

    if geometry.is_empty:

        raise ValueError(
            "The KML contains no usable polygon geometry."
        )

    return geometry


# GEOMETRY

def reproject_geometry(
    geometry,
    source_crs,
    destination_crs,
):

    transformer = Transformer.from_crs(
        source_crs,
        destination_crs,
        always_xy=True,
    )

    return shapely_transform(
        transformer.transform,
        geometry,
    )


def raster_footprint(
    transform,
    width,
    height,
):

    return Polygon(
        [
            transform * (0, 0),
            transform * (width, 0),
            transform * (width, height),
            transform * (0, height),
        ]
    )


def detection_box_geometry(
    xmin,
    ymin,
    xmax,
    ymax,
    transform,
):

    return Polygon(
        [
            transform * (xmin, ymin),
            transform * (xmax, ymin),
            transform * (xmax, ymax),
            transform * (xmin, ymax),
        ]
    )


def detection_center(
    xmin,
    ymin,
    xmax,
    ymax,
    transform,
):

    center_col = (
        xmin + xmax
    ) / 2

    center_row = (
        ymin + ymax
    ) / 2

    x, y = transform * (
        center_col,
        center_row,
    )

    return Point(
        x,
        y,
    )


def get_polygons(geometry):

    if geometry is None:
        return []

    if geometry.geom_type == "Polygon":
        return [geometry]

    if geometry.geom_type == "MultiPolygon":
        return list(
            geometry.geoms
        )

    if hasattr(
        geometry,
        "geoms",
    ):

        return [
            geom
            for geom in geometry.geoms
            if geom.geom_type == "Polygon"
        ]

    return []


# CRS CHECK

def uses_metre_units(crs):

    if crs is None:
        return False

    try:

        crs_obj = CRS.from_user_input(
            crs
        )

        if not crs_obj.is_projected:
            return False

        for axis in crs_obj.axis_info:

            unit = (
                axis.unit_name
                or ""
            ).lower()

            if (
                "metre" in unit
                or "meter" in unit
            ):
                return True

    except Exception:
        pass

    return False


# DEEPFOREST

@st.cache_data(
    show_spinner=False
)
def run_detection(
    png_bytes,
):

    model = load_model()

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as temp:

            temp.write(
                png_bytes
            )

            temp_path = (
                temp.name
            )

        predictions = (
            model.predict_image(
                path=temp_path
            )
        )

        return predictions

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


# VISUALIZATION

def draw_aoi_overlay(
    image,
    geometry,
    transform,
):

    overlay = Image.new(
        "RGBA",
        image.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        overlay
    )

    inverse = ~transform

    for polygon in get_polygons(
        geometry
    ):

        points = []

        for x, y in (
            polygon.exterior.coords
        ):

            col, row = (
                inverse * (x, y)
            )

            points.append(
                (
                    int(col),
                    int(row),
                )
            )

        if len(points) >= 3:

            draw.polygon(
                points,
                fill=(
                    255,
                    215,
                    0,
                    25,
                ),
            )

            draw.line(
                points,
                fill=(
                    255,
                    235,
                    59,
                    255,
                ),
                width=4,
                joint="curve",
            )

    return Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    )


def draw_detections(
    image,
    results_df,
    show_excluded,
    show_labels,
):

    output = (
        image
        .convert("RGB")
        .copy()
    )

    draw = ImageDraw.Draw(
        output
    )

    for _, row in (
        results_df.iterrows()
    ):

        inside = bool(
            row["inside_aoi"]
        )

        if (
            not inside
            and not show_excluded
        ):
            continue

        xmin = int(
            row["xmin_px"]
        )

        ymin = int(
            row["ymin_px"]
        )

        xmax = int(
            row["xmax_px"]
        )

        ymax = int(
            row["ymax_px"]
        )

        color = (
            "lime"
            if inside
            else "red"
        )

        draw.rectangle(
            [
                xmin,
                ymin,
                xmax,
                ymax,
            ],
            outline=color,
            width=3,
        )

        if show_labels:

            label = (
                f"#{int(row['tree_id'])} "
                f"{row['confidence']:.2f}"
            )

            text_y = max(
                0,
                ymin - 14,
            )

            draw.text(
                (
                    xmin + 2,
                    text_y,
                ),
                label,
                fill=color,
            )

    return output


# SIDEBAR

with st.sidebar:

    st.header(
        "Analysis settings"
    )

    confidence_threshold = (
        st.slider(
            "Confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help=(
                "Lower thresholds detect more possible "
                "tree crowns but may increase false positives."
            ),
        )
    )

    st.divider()

    show_excluded = st.toggle(
        "Show excluded detections",
        value=True,
        help=(
            "Show detections whose centres "
            "fall outside the uploaded AOI."
        ),
    )

    show_labels = st.toggle(
        "Show confidence labels",
        value=False,
    )

    st.divider()

    st.caption(
        "Model"
    )

    st.write(
        "**DeepForest**"
    )

    st.caption(
        MODEL_NAME
    )

    st.caption(
        "Tree inclusion rule: detection centre "
        "must fall inside the analysis boundary."
    )


# UPLOAD


st.subheader(
    "Upload imagery"
)

upload_col1, upload_col2 = (
    st.columns(
        [1, 1]
    )
)

with upload_col1:

    geotiff_file = (
        st.file_uploader(
            "GeoTIFF imagery",
            type=[
                "tif",
                "tiff",
            ],
            help=(
                "Upload georeferenced forest imagery."
            ),
        )
    )

with upload_col2:

    kml_file = (
        st.file_uploader(
            "Forest boundary (optional)",
            type=["kml"],
            help=(
                "Upload a KML polygon to restrict "
                "analysis to a project or forest boundary."
            ),
        )
    )


# EMPTY STATE


if geotiff_file is None:

    st.info(
        "Upload a GeoTIFF to start the analysis. "
        "A KML boundary is optional."
    )

    st.stop()


# READ TIFF

try:

    tiff_bytes = (
        geotiff_file.getvalue()
    )

    rgb_array, metadata = (
        read_geotiff(
            tiff_bytes
        )
    )

except Exception as error:

    st.error(
        "The GeoTIFF could not be read."
    )

    st.exception(
        error
    )

    st.stop()


width = metadata["width"]
height = metadata["height"]
crs = metadata["crs"]
transform = metadata["transform"]
bounds = metadata["bounds"]

metric_crs = uses_metre_units(
    crs
)


# IMAGE METADATA

with st.expander(
    "GeoTIFF details",
    expanded=False,
):

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "Width",
        f"{width}px",
    )

    c2.metric(
        "Height",
        f"{height}px",
    )

    c3.metric(
        "Bands",
        metadata["count"],
    )

    c4.metric(
        "CRS",
        str(crs)
        if crs
        else "Missing",
    )

    c5.metric(
        "Data type",
        metadata["dtype"],
    )

    st.caption(
        f"Bounds: "
        f"{bounds.left:.2f}, "
        f"{bounds.bottom:.2f}, "
        f"{bounds.right:.2f}, "
        f"{bounds.top:.2f}"
    )


# RASTER FOOTPRINT

raster_polygon = (
    raster_footprint(
        transform,
        width,
        height,
    )
)

analysis_aoi = (
    raster_polygon
)

uploaded_kml_geometry = None

aoi_name = (
    "Entire GeoTIFF"
)


# KML PROCESSING


if kml_file is not None:

    if crs is None:

        st.error(
            "This GeoTIFF has no CRS, so the KML "
            "cannot be aligned reliably."
        )

        st.stop()

    try:

        kml_wgs84 = (
            parse_kml(
                kml_file.getvalue()
            )
        )

        uploaded_kml_geometry = (
            reproject_geometry(
                kml_wgs84,
                "EPSG:4326",
                crs,
            )
        )

        overlap = (
            raster_polygon
            .intersection(
                uploaded_kml_geometry
            )
        )

        if overlap.is_empty:

            st.error(
                "The uploaded KML boundary does not "
                "overlap this GeoTIFF."
            )

            st.stop()

        analysis_aoi = (
            overlap
        )

        aoi_name = (
            kml_file.name
        )

        st.success(
            "✓ KML boundary loaded, reprojected "
            "and matched with the GeoTIFF."
        )

    except Exception as error:

        st.error(
            "The KML could not be processed."
        )

        st.exception(
            error
        )

        st.stop()

else:

    st.info(
        "No KML boundary supplied — analysis "
        "will use the full GeoTIFF extent."
    )


# PREPARE MODEL IMAGE

pil_image = Image.fromarray(
    rgb_array
).convert("RGB")

png_buffer = io.BytesIO()

pil_image.save(
    png_buffer,
    format="PNG",
)

png_bytes = (
    png_buffer.getvalue()
)



# RUN MODEL

with st.spinner(
    "Detecting tree crowns..."
):

    try:

        raw_predictions = (
            run_detection(
                png_bytes
            )
        )

    except Exception as error:

        st.error(
            "DeepForest could not complete the analysis."
        )

        st.exception(
            error
        )

        st.stop()


if raw_predictions is None:

    raw_predictions = (
        pd.DataFrame()
    )


raw_detection_count = (
    len(raw_predictions)
)


if raw_predictions.empty:

    predictions = (
        raw_predictions.copy()
    )

else:

    predictions = (
        raw_predictions[
            raw_predictions[
                "score"
            ]
            >= confidence_threshold
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


threshold_detection_count = (
    len(predictions)
)



# DETECTION ANALYSIS

records = []
included_geometries = []

inside_count = 0
outside_count = 0


for index, row in (
    predictions.iterrows()
):

    xmin = float(
        row["xmin"]
    )

    ymin = float(
        row["ymin"]
    )

    xmax = float(
        row["xmax"]
    )

    ymax = float(
        row["ymax"]
    )

    score = float(
        row["score"]
    )

    center = (
        detection_center(
            xmin,
            ymin,
            xmax,
            ymax,
            transform,
        )
    )

    bbox_geometry = (
        detection_box_geometry(
            xmin,
            ymin,
            xmax,
            ymax,
            transform,
        )
    )

    inside_aoi = (
        analysis_aoi.covers(
            center
        )
    )

    clipped_area = 0.0

    if inside_aoi:

        inside_count += 1

        clipped = (
            bbox_geometry
            .intersection(
                analysis_aoi
            )
        )

        if not clipped.is_empty:

            included_geometries.append(
                clipped
            )

            clipped_area = (
                clipped.area
            )

    else:

        outside_count += 1

    record = {

        "tree_id":
            index + 1,

        "confidence":
            score,

        "inside_aoi":
            inside_aoi,

        "xmin_px":
            xmin,

        "ymin_px":
            ymin,

        "xmax_px":
            xmax,

        "ymax_px":
            ymax,

        "box_width_px":
            xmax - xmin,

        "box_height_px":
            ymax - ymin,

        "bbox_area_map_units":
            bbox_geometry.area,

        "area_inside_aoi_map_units":
            clipped_area,
    }

    if "label" in row:

        record[
            "model_label"
        ] = row["label"]

    records.append(
        record
    )


result_columns = [

    "tree_id",
    "confidence",
    "inside_aoi",

    "xmin_px",
    "ymin_px",
    "xmax_px",
    "ymax_px",

    "box_width_px",
    "box_height_px",

    "bbox_area_map_units",
    "area_inside_aoi_map_units",
]


results_df = pd.DataFrame(
    records
)


if results_df.empty:

    results_df = pd.DataFrame(
        columns=result_columns
    )


# CANOPY AREA

if included_geometries:

    canopy_union = unary_union(
        included_geometries
    )

    canopy_area = (
        canopy_union.area
    )

else:

    canopy_area = 0.0


aoi_area = (
    analysis_aoi.area
)


if (
    metric_crs
    and aoi_area > 0
):

    canopy_coverage = (
        canopy_area
        / aoi_area
        * 100
    )

else:

    canopy_coverage = None


# STATUS HEADER

st.divider()

st.subheader(
    "Analysis results"
)

st.markdown(
    f"""
    <div class="status-good">
        <strong>Analysis area:</strong> {aoi_name}
        &nbsp;&nbsp;•&nbsp;&nbsp;
        <strong>Confidence threshold:</strong> {confidence_threshold:.2f}
        &nbsp;&nbsp;•&nbsp;&nbsp;
        <strong>Model:</strong> DeepForest
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# MAIN METRICS

m1, m2, m3, m4 = (
    st.columns(4)
)

m1.metric(
    "Raw detections",
    raw_detection_count,
)

m2.metric(
    "At current threshold",
    threshold_detection_count,
)

m3.metric(
    "Trees inside AOI",
    inside_count,
)

m4.metric(
    "Excluded from AOI",
    outside_count,
)


if metric_crs:

    a1, a2, a3, a4 = (
        st.columns(4)
    )

    a1.metric(
        "AOI area",
        f"{aoi_area:,.2f} m²",
    )

    a2.metric(
        "AOI area",
        f"{aoi_area / 10000:,.4f} ha",
    )

    a3.metric(
        "Approx. crown-box area",
        f"{canopy_area:,.2f} m²",
    )

    a4.metric(
        "Approx. canopy coverage",
        (
            f"{canopy_coverage:.2f}%"
            if canopy_coverage
            is not None
            else "N/A"
        ),
    )

else:

    st.warning(
        "The GeoTIFF CRS is not confirmed to use metres. "
        "Tree detection can continue, but CanopyLens will "
        "not present area values as m² or hectares."
    )


# PREPARE VISUALIZATION

visual = pil_image

if kml_file is not None:

    visual = (
        draw_aoi_overlay(
            visual,
            analysis_aoi,
            transform,
        )
    )


visual = draw_detections(
    visual,
    results_df,
    show_excluded,
    show_labels,
)


# TABS

overview_tab, map_tab, data_tab = (
    st.tabs(
        [
            "Overview",
            "Detection map",
            "Detection data",
        ]
    )
)


# OVERVIEW TAB

with overview_tab:

    left, right = (
        st.columns(
            [1.5, 1]
        )
    )

    with left:

        st.image(
            visual,
            caption=(
                "Green = included tree crown detection • "
                "Red = excluded detection • "
                "Yellow = AOI boundary"
            ),
            use_container_width=True,
        )

    with right:

        st.markdown(
            "### Analysis summary"
        )

        st.write(
            f"""
            **{threshold_detection_count}** crown candidates
            passed the selected confidence threshold.

            **{inside_count}** detections have their centre
            inside the selected analysis area.
            """
        )

        if kml_file is not None:

            st.write(
                f"""
                **{outside_count}** detections were excluded
                because their centres fall outside the
                uploaded forest boundary.
                """
            )

        if metric_crs:

            st.write(
                f"""
                The AOI covers **{aoi_area:,.2f} m²**
                (**{aoi_area / 10000:.4f} ha**).

                The union of included detection boxes covers
                approximately **{canopy_area:,.2f} m²**,
                corresponding to **{canopy_coverage:.2f}%**
                of the AOI.
                """
            )

        st.info(
            "The area estimate uses the union of DeepForest "
            "bounding boxes. Overlapping boxes are therefore "
            "not counted twice."
        )


# MAP TAB

with map_tab:

    st.image(
        visual,
        use_container_width=True,
    )

    legend1, legend2, legend3 = (
        st.columns(3)
    )

    legend1.success(
        "Green — included detection"
    )

    legend2.error(
        "Red — outside analysis boundary"
    )

    if kml_file is not None:

        legend3.warning(
            "Yellow — uploaded KML AOI"
        )

    else:

        legend3.info(
            "AOI — full GeoTIFF"
        )


#data tab

with data_tab:

    if results_df.empty:

        st.warning(
            "No detections passed the selected threshold."
        )

    else:

        display_df = (
            results_df.copy()
        )

        display_df[
            "confidence"
        ] = (
            display_df[
                "confidence"
            ]
            .astype(float)
            .round(3)
        )

        for column in [
            "xmin_px",
            "ymin_px",
            "xmax_px",
            "ymax_px",
            "box_width_px",
            "box_height_px",
        ]:

            if column in display_df:

                display_df[
                    column
                ] = (
                    display_df[
                        column
                    ]
                    .astype(float)
                    .round(1)
                )

        if metric_crs:

            display_df[
                "approx_box_area_m2"
            ] = (
                display_df[
                    "bbox_area_map_units"
                ]
                .astype(float)
                .round(2)
            )

            display_df[
                "area_inside_aoi_m2"
            ] = (
                display_df[
                    "area_inside_aoi_map_units"
                ]
                .astype(float)
                .round(2)
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = (
            display_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        summary = {

            "application":
                APP_NAME,

            "model":
                MODEL_NAME,

            "confidence_threshold":
                confidence_threshold,

            "raw_detections":
                raw_detection_count,

            "detections_above_threshold":
                threshold_detection_count,

            "trees_inside_aoi":
                inside_count,

            "trees_outside_aoi":
                outside_count,

            "crs":
                str(crs),

            "aoi_source":
                aoi_name,
        }

        if metric_crs:

            summary.update(
                {
                    "aoi_area_m2":
                        round(
                            aoi_area,
                            2,
                        ),

                    "aoi_area_ha":
                        round(
                            aoi_area
                            / 10000,
                            5,
                        ),

                    "approx_crown_box_area_m2":
                        round(
                            canopy_area,
                            2,
                        ),

                    "approx_canopy_coverage_percent":
                        round(
                            canopy_coverage,
                            2,
                        ),
                }
            )

        summary_json = json.dumps(
            summary,
            indent=2,
        )

        download1, download2, download3 = (
            st.columns(3)
        )

        with download1:

            st.download_button(
                "⬇ Download detections CSV",
                data=csv_bytes,
                file_name=(
                    "canopylens_detections.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with download2:

            st.download_button(
                "⬇ Download analysis summary",
                data=summary_json,
                file_name=(
                    "canopylens_summary.json"
                ),
                mime="application/json",
                use_container_width=True,
            )

        with download3:

            output_buffer = (
                io.BytesIO()
            )

            visual.save(
                output_buffer,
                format="PNG",
            )

            st.download_button(
                "⬇ Download annotated image",
                data=(
                    output_buffer
                    .getvalue()
                ),
                file_name=(
                    "canopylens_detection_map.png"
                ),
                mime="image/png",
                use_container_width=True,
            )


