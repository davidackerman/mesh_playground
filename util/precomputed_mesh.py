"""Loader for neuroglancer precomputed multiresolution meshes.

Uses CloudVolume for assembly so both the unsharded and sharded
(``neuroglancer_uint64_sharded_v1``) layouts work transparently, over local
(``file://``) paths and remote http(s)/gs/s3 servers. Vertices are returned in
physical nm (CloudVolume applies the mesh ``transform``).

Adapted from CineMap's ``MeshLoader`` (src/cinemap/data/mesh_loader.py): a bare
neuroglancer *mesh* dir has a mesh-info (no ``scales``), so we point CloudVolume
at the parent volume with a fabricated info that names the mesh subdir. A legacy
meshifier variant baked Draco's own dequantization into the points, which makes
CloudVolume double-scale; we detect that and decode those ourselves.

References:
  https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md
"""
import os
import struct

import numpy as np
import requests
import DracoPy
from cloudvolume import CloudVolume
from requests.adapters import HTTPAdapter
from urllib3.response import HTTPResponse


class FileAdapter(HTTPAdapter):
    """Lets ``requests`` serve ``file://`` URLs (used for local mesh data)."""

    def send(self, request, *args, **kwargs):
        resp = HTTPResponse(
            body=open(request.url[7:], "rb"), status=200, preload_content=False
        )
        return self.build_response(request, resp)


def _http(url):
    """urllib/CloudVolume read https directly; map cloud URLs to https."""
    if url.startswith("gs://"):
        return "https://storage.googleapis.com/" + url[len("gs://"):]
    if url.startswith("s3://"):
        return "https://s3.amazonaws.com/" + url[len("s3://"):]
    return url


class PrecomputedMeshReader:
    """Fetches and decodes precomputed multiresolution meshes via CloudVolume."""

    def __init__(self, path, lod):
        if not (path.startswith(("http", "file", "gs://", "s3://"))):
            path = "file://" + path
        self.is_file_path = path.startswith("file://")
        self.path = path.rstrip("/")
        self.lod = lod
        self.parent, self.subdir = (
            self.path.rsplit("/", 1) if "/" in self.path else ("", self.path)
        )
        self._cv = None
        self._manual = None  # does this source need our model-space draco decode?

        self.session = requests.Session()
        self.session.mount("file://", FileAdapter())

    @property
    def cv(self):
        """A CloudVolume exposing this source's meshes.

        Either the source is a precomputed *segmentation* volume (its info has
        ``scales`` and a ``mesh`` key) and we open it directly, or it's a bare
        *mesh* dir, in which case we point CloudVolume at the parent with a
        fabricated volume info naming this subdir as ``mesh``.
        """
        if self._cv is None:
            use_https = not self.is_file_path
            try:
                direct = CloudVolume(
                    f"precomputed://{self.path}", use_https=use_https, progress=False
                )
                if "scales" in direct.info and direct.info.get("mesh"):
                    self._cv = direct
            except Exception:
                pass
            if self._cv is None:
                info = {
                    "@type": "neuroglancer_multiscale_volume",
                    "type": "segmentation",
                    "data_type": "uint64",
                    "num_channels": 1,
                    "mesh": self.subdir,
                    "scales": [{
                        "key": "s0", "size": [1, 1, 1], "resolution": [8, 8, 8],
                        "chunk_sizes": [[64, 64, 64]], "encoding": "raw",
                        "voxel_offset": [0, 0, 0],
                    }],
                }
                self._cv = CloudVolume(
                    f"precomputed://{self.parent}", info=info,
                    use_https=use_https, progress=False,
                )
        return self._cv

    @property
    def info(self):
        return self.cv.mesh.meta.info

    @property
    def is_sharded(self):
        return bool(self.info.get("sharding"))

    def list_segment_ids(self):
        """Segment ids for this mesh source.

        Prefers the ``segment_properties`` inline id list; for a local unsharded
        directory, falls back to listing ``{id}.index`` manifest files.
        """
        try:
            d = self.session.get(f"{self.path}/segment_properties/info").json()
            ids = [int(x) for x in d.get("inline", {}).get("ids", [])]
            if ids:
                return sorted(ids)
        except Exception:
            pass
        if self.is_file_path and not self.is_sharded:
            local = self.path[len("file://"):]
            return sorted(
                int(f[: -len(".index")])
                for f in os.listdir(local)
                if f.endswith(".index")
            )
        return []

    def _read(self, rel_path):
        return self.session.get(f"{self.path}/{rel_path}").content

    def _manifest_bits(self):
        try:
            return int(self.info.get("vertex_quantization_bits", 16))
        except Exception:
            return 16

    def _needs_manual_decode(self, sample_id):
        """True for meshes from the legacy meshifier (see module docstring).

        That pipeline baked Draco's position dequantization in, so the decoded
        points come out in fractional chunk model space rather than the integer
        grid indices [0, 2^bits) the current meshifier (and CloudVolume) assume;
        CloudVolume then double-scales them. We sample one fragment to detect it.
        Sharded / unreachable sources fall through to CloudVolume.
        """
        if self._manual is not None:
            return self._manual
        result = False
        try:
            bits = self._manifest_bits()
            idx = self._read(f"{sample_id}.index")
            data = self._read(f"{sample_id}")
            o = 24                                       # skip chunk_shape + grid_origin
            nl = struct.unpack("<I", idx[o:o + 4])[0]; o += 4
            o += 4 * nl + 12 * nl                        # lod_scales + vertex_offsets
            nfrag = struct.unpack(f"<{nl}I", idx[o:o + 4 * nl]); o += 4 * nl
            n0 = nfrag[0]
            o += 12 * n0                                 # skip lod0 fragment_positions
            fsz = struct.unpack(f"<{n0}I", idx[o:o + 4 * n0])
            dp = 0
            for sz in fsz:                               # first non-empty lod0 fragment
                if sz:
                    pts = np.asarray(DracoPy.decode(data[dp:dp + sz]).points, float)
                    fractional = not np.allclose(pts, np.round(pts), atol=1e-3)
                    result = bool(fractional or pts.max() > 1.5 * (2 ** bits))
                    break
                dp += sz
        except Exception:  # sharded / unreachable -> trust CloudVolume
            result = False
        self._manual = result
        return self._manual

    def _draco_manual(self, id):
        """Decode legacy-meshifier unsharded meshes (Draco already dequantized
        the points to chunk model space, so vertex = grid_origin + points), then
        apply the mesh transform to return physical nm."""
        idx = self._read(f"{id}.index")
        data = self._read(f"{id}")
        o = 12                                           # skip chunk_shape
        go = np.array(struct.unpack("<3f", idx[o:o + 12])); o += 12
        nl = struct.unpack("<I", idx[o:o + 4])[0]; o += 4
        o += 4 * nl + 12 * nl                            # lod_scales + vertex_offsets
        nfrag = struct.unpack(f"<{nl}I", idx[o:o + 4 * nl]); o += 4 * nl
        want = min(max(int(self.lod), 0), nl - 1)
        dp = 0
        verts = []
        faces = []
        nv = 0
        for cur in range(nl):                            # fragments stored lod0..lodN
            n = nfrag[cur]
            o += 12 * n                                  # skip fragment_positions
            fsz = struct.unpack(f"<{n}I", idx[o:o + 4 * n]); o += 4 * n
            for i in range(n):
                b = data[dp:dp + fsz[i]]; dp += fsz[i]
                if cur == want and fsz[i]:
                    mm = DracoPy.decode(b)
                    verts.append(go + np.asarray(mm.points, float))
                    faces.append(np.asarray(mm.faces, np.int64) + nv)
                    nv += len(mm.points)
        model = np.vstack(verts)
        all_faces = np.vstack(faces)
        T = np.asarray(self.cv.mesh.transform, float)    # 4x4, resolution baked in
        nm = model @ T[:3, :3].T + T[:3, 3]
        return nm, all_faces

    def load_vertices_faces(self, id):
        """Download and decode mesh ``id`` at ``self.lod``.

        Returns ``(vertices, faces)`` with vertices in physical nm and fragments
        concatenated. Works for both sharded and unsharded meshes.
        """
        id = int(id)
        if not self.is_sharded and self._needs_manual_decode(id):
            return self._draco_manual(id)

        try:
            mesh = self.cv.mesh.get(id, lod=self.lod)
        except TypeError:  # source has no LOD support
            mesh = self.cv.mesh.get(id)
        if isinstance(mesh, dict):
            mesh = mesh[id]
        return (
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
        )
