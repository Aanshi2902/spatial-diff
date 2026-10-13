"""Interactive 3D HTML viewer: aligned scenes + colour-coded change boxes + move arrows."""
import numpy as np
import plotly.graph_objects as go

COL = {"ADDED": "#1faa59", "REMOVED": "#e03131", "MOVED": "#f59f00"}


def _box_lines(lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    c = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [c[a][0], c[b][0], None]; ys += [c[a][1], c[b][1], None]; zs += [c[a][2], c[b][2], None]
    return xs, ys, zs


def _sub(p, c, n, seed=0):
    if len(p) <= n:
        return p, c
    i = np.random.default_rng(seed).choice(len(p), n, replace=False)
    return p[i], (None if c is None else c[i])


def export_html(A_pts, A_cols, B_pts, B_cols, changes, path, unit="m", max_pts=25000):
    fig = go.Figure()
    for name, p, c, vis in (("Scene A (before)", A_pts, A_cols, True),
                            ("Scene B (after)", B_pts, B_cols, "legendonly")):
        p, c = _sub(p, c, max_pts)
        color = ([f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in c]
                 if c is not None else "#888")
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                                   visible=vis, marker=dict(size=1.6, color=color, opacity=0.8)))
    seen = set()
    for ch in changes:
        k = ch["kind"]
        if k == "MOVED":
            boxes = [(ch["from_bbox"][0], ch["from_bbox"][1], "dash"), (ch["bbox_min"], ch["bbox_max"], "solid")]
        else:
            boxes = [(ch["bbox_min"], ch["bbox_max"], "solid")]
        for lo, hi, dash in boxes:
            xs, ys, zs = _box_lines(lo, hi)
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=k, legendgroup=k,
                                       showlegend=k not in seen,
                                       line=dict(color=COL[k], width=6, dash=dash)))
            seen.add(k)
        pos = np.array(ch.get("center", ch.get("to_center")))
        text = f"{k}: {ch['label']}" + (f" ({ch['distance']:.2f} {unit})" if k == "MOVED" else "")
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2] + (ch['bbox_max'][2] - ch['bbox_min'][2]) / 2 + 0.1],
                                   mode="text", text=[text], textfont=dict(color=COL[k], size=14),
                                   showlegend=False))
        if k == "MOVED":
            a, b = ch["from_center"], ch["to_center"]
            fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines+markers",
                                       line=dict(color=COL[k], width=5), marker=dict(size=[2, 6]),
                                       showlegend=False))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                      title="Spatial change map: green = added, red = removed, yellow = moved (dashed = old position)",
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"))
    fig.write_html(path, include_plotlyjs="cdn")
    return path
