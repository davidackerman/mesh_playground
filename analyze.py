"""Batch mesh analysis: compute geometric + watertight metrics for every segment
in a precomputed multiresolution mesh source and write ``mesh_metrics.csv``.

Computes the same properties as the per-mesh analyzer (``MeshProcessor.compute_metrics``):
volume, surface area, principal-inertia components, oriented-bounding-box dims,
discrete curvature (mean/gaussian/rms/abs) and shape-diameter thickness, plus
watertightness diagnostics (is_watertight, winding consistency, euler number,
boundary edges). Work is fanned out across processes with dask.

Usage:
    pixi run python analyze.py <mesh_dir> <out_dir> [options]

Example:
    pixi run python analyze.py /path/to/mesh/multires ./out --lod 0 --workers 10
"""
import argparse
import os

import dask
import pandas as pd
from dask.distributed import Client, LocalCluster

from util.mesh import MeshProcessor


def _analyze_chunk(cfg, ids):
    """Analyze a chunk of segment ids with a single MeshProcessor (built in the
    worker, so nothing unpicklable crosses the process boundary). Per-mesh
    failures are recorded as an ``error`` field rather than aborting the run."""
    mp = MeshProcessor(**cfg)
    rows = []
    for seg in ids:
        try:
            rows.append(mp.compute_metrics(int(seg)))
        except Exception as e:  # noqa: BLE001 — keep going; flag the bad mesh
            rows.append({"id": int(seg), "error": repr(e)})
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mesh_dir", help="precomputed mesh directory (local path or URL)")
    ap.add_argument("out_dir", help="directory to write mesh_metrics.csv into")
    ap.add_argument("--lod", type=int, default=0, help="level of detail (0 = finest)")
    ap.add_argument("--workers", type=int, default=10, help="dask worker processes")
    ap.add_argument("--skeletons", action="store_true", help="also compute skeleton metrics")
    ap.add_argument("--min-branch-length", type=int, default=500)
    ap.add_argument("--ids", type=int, nargs="*", help="explicit segment ids (default: all)")
    args = ap.parse_args()

    cfg = dict(
        path=args.mesh_dir,
        lod=args.lod,
        use_skeletons=args.skeletons,
        min_branch_length=args.min_branch_length,
    )

    ids = args.ids if args.ids else MeshProcessor(**cfg).reader.list_segment_ids()
    if not ids:
        raise SystemExit(f"no segment ids found at {args.mesh_dir}")
    print(f"analyzing {len(ids)} meshes (lod={args.lod}, workers={args.workers})")

    # split into ~4 chunks per worker for load balancing
    nchunks = max(1, min(len(ids), args.workers * 4))
    chunks = [ids[i::nchunks] for i in range(nchunks)]

    cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
    client = Client(cluster)
    print("dask dashboard:", client.dashboard_link)
    try:
        results = dask.compute(*[dask.delayed(_analyze_chunk)(cfg, ch) for ch in chunks])
    finally:
        client.close()
        cluster.close()

    rows = [row for chunk in results for row in chunk]
    df = pd.DataFrame.from_records(rows).sort_values("id").reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "mesh_metrics.csv")
    df.to_csv(out, index=False)

    n_err = int(df["error"].notna().sum()) if "error" in df.columns else 0
    print(f"wrote {out}: {len(df)} rows ({n_err} errored)")


if __name__ == "__main__":
    main()
