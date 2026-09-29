from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import shapefile
from pyproj import Transformer
from shapely.geometry import shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory_csv", type=Path)
    parser.add_argument("corridor_shp", type=Path)
    parser.add_argument("mosaic_1", type=Path)
    parser.add_argument("mosaic_2", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.inventory_csv, parse_dates=["metadata_date"])
    data = data[data["region_class"] == "Pinglu Canal candidate"].copy()
    data["year"] = data["metadata_date"].dt.year
    # PAN/MSS metadata may describe the same acquisition package; use one row.
    data = data.sort_values(["scene_key", "image_gsd_m"]).drop_duplicates("scene_key")

    figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    styles = {2021: ("#d95f02", "2021 source-scene footprint"), 2025: ("#1b9e77", "2025 source-scene footprint")}
    for year, (color, label) in styles.items():
        subset = data[data["year"] == year]
        for row_index, (_, row) in enumerate(subset.iterrows()):
            x = [row.min_lon, row.max_lon, row.max_lon, row.min_lon, row.min_lon]
            y = [row.min_lat, row.min_lat, row.max_lat, row.max_lat, row.min_lat]
            axis.plot(x, y, color=color, linewidth=0.8, alpha=0.60, label=label if row_index == 0 else None)

    reader = shapefile.Reader(str(args.corridor_shp))
    for index, record_shape in enumerate(reader.shapes()):
        geometry = shape(record_shape.__geo_interface__)
        geoms = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in geoms:
            x, y = polygon.exterior.xy
            axis.plot(x, y, color="black", linewidth=1.25, label="200-m project corridor" if index == 0 else None)

    for mosaic_path, color, label in (
        (args.mosaic_1, "#7570b3", "Mosaic A extent"),
        (args.mosaic_2, "#e7298a", "Mosaic B extent"),
    ):
        with rasterio.open(mosaic_path) as dataset:
            transformer = Transformer.from_crs(dataset.crs, 4326, always_xy=True)
            left, bottom = transformer.transform(dataset.bounds.left, dataset.bounds.bottom)
            right, top = transformer.transform(dataset.bounds.right, dataset.bounds.top)
        axis.plot(
            [left, right, right, left, left],
            [bottom, bottom, top, top, bottom],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=label,
        )

    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title("Metadata-derived acquisition footprints and processed mosaic extents")
    axis.grid(True, color="#dddddd", linewidth=0.5)
    axis.legend(loc="best", fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=450)
    figure.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
