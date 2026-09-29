from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import shapefile
from pyproj import CRS, Geod
from shapely.geometry import shape


def geodesic_area_m2(geometry, geod: Geod) -> float:
    area, _ = geod.geometry_area_perimeter(geometry)
    return abs(float(area))


def audit(path: Path, geod: Geod) -> tuple[dict, list[dict]]:
    cpg = path.with_suffix(".cpg")
    encoding = cpg.read_text(encoding="ascii", errors="ignore").strip() if cpg.exists() else "gb18030"
    reader = shapefile.Reader(str(path), encoding=encoding, encodingErrors="replace")
    fields = [field[0] for field in reader.fields[1:]]
    rows = []
    total_area = 0.0
    for index, shape_record in enumerate(reader.iterShapeRecords(), 1):
        geometry = shape(shape_record.shape.__geo_interface__)
        area = geodesic_area_m2(geometry, geod)
        total_area += area
        attributes = dict(zip(fields, list(shape_record.record)))
        rows.append(
            {
                "source_file": str(path.resolve()),
                "feature_index": index,
                "geometry_type": geometry.geom_type,
                "geodesic_area_m2": area,
                "geodesic_area_ha": area / 10000,
                "attributes_json": json.dumps(attributes, ensure_ascii=False, default=str),
            }
        )
    prj = path.with_suffix(".prj")
    crs_wkt = prj.read_text(encoding="utf-8", errors="replace") if prj.exists() else None
    crs = CRS.from_wkt(crs_wkt) if crs_wkt else None
    return (
        {
            "source_file": str(path.resolve()),
            "record_count": len(reader),
            "shape_type": reader.shapeTypeName,
            "dbf_encoding": encoding,
            "fields": fields,
            "bbox": list(reader.bbox),
            "crs": crs.to_string() if crs else None,
            "geodesic_area_m2": total_area,
            "geodesic_area_ha": total_area / 10000,
        },
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("shapefiles", type=Path, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geod = Geod(ellps="WGS84")
    summaries = []
    features = []
    for path in args.shapefiles:
        summary, rows = audit(path, geod)
        summaries.append(summary)
        features.extend(rows)
    totals = {
        "record_count": sum(item["record_count"] for item in summaries),
        "geodesic_area_m2": sum(item["geodesic_area_m2"] for item in summaries),
        "geodesic_area_ha": sum(item["geodesic_area_ha"] for item in summaries),
    }
    payload = {"files": summaries, "totals": totals}
    (args.output_dir / "project_vector_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "project_vector_features.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
