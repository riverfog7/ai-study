import base64
from io import BytesIO

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, no_update


def _pil_to_data_uri(img, max_side=160, fmt="JPEG", q=85):
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, format=fmt, quality=q)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{b64}"

def create_3d_plot_with_image_hover_colored(
    img_xyz,                 # np.ndarray (N_img, 3)
    txt_xyz,                 # np.ndarray (N_txt, 3)
    images,                  # list[PIL.Image], len == N_img
    text_labels,             # list[str], len == N_txt (labels to display next to text points)
    img_classes,             # list[str|int], len == N_img
    text_classes=None,       # optional list[str|int], len == N_txt; if None, infer from text_labels
    title="Embeddings (3D)",
):
    """
    Returns a Dash app that:
      - Plots 3D image and text points with class-consistent colors.
      - Shows image thumbnails on hover for image points only via dcc.Tooltip.
    """
    img_xyz = np.asarray(img_xyz)
    txt_xyz = np.asarray(txt_xyz)
    if text_classes is None:
        text_classes = list(text_labels)

    # Build a consistent discrete color map across all classes. [web:583]
    all_classes = list(dict.fromkeys(list(img_classes) + list(text_classes)))
    palette = px.colors.qualitative.Prism
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(all_classes)}  # class -> color [web:583]

    # Precompute image data URIs for hover previews. [web:588]
    img_uris = [_pil_to_data_uri(im) for im in images]

    fig = go.Figure()

    # Add one image trace per class for clean legends and per-class coloring. [web:572]
    for cls in all_classes:
        idx = [i for i, c in enumerate(img_classes) if c == cls]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=img_xyz[idx, 0],
                y=img_xyz[idx, 1],
                z=img_xyz[idx, 2],
                mode="markers",
                name=str(cls),
                legendgroup=str(cls),
                showlegend=True,
                marker=dict(size=4, color=color_map[cls], opacity=0.85),
                # Put the data URI directly per-point; Tooltip callback reads this. [web:588]
                customdata=[img_uris[i] for i in idx],
                hoverinfo="none",     # suppress default tooltip for image traces [web:588]
                hovertemplate=None,   # required for dcc.Tooltip to control content [web:588]
                meta=dict(kind="images", cls=str(cls)),
            )
        )

    # Add one text trace per class with the same color and diamond symbol. [web:572]
    for cls in all_classes:
        idx = [i for i, c in enumerate(text_classes) if c == cls]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=txt_xyz[idx, 0],
                y=txt_xyz[idx, 1],
                z=txt_xyz[idx, 2],
                mode="markers+text",
                name=str(cls) + " (text)",
                legendgroup=str(cls),
                showlegend=False,  # keep legend single-entry per class (from images) [web:572]
                marker=dict(size=8, color=color_map[cls], symbol="diamond"),
                text=[text_labels[i] for i in idx],
                textposition="top center",
                hovertemplate="%{text}<extra></extra>",  # textual hover for text points [web:566]
                meta=dict(kind="text", cls=str(cls)),
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(aspectmode="data"),
        hovermode="closest",
    )

    # Dash app with dcc.Tooltip displaying the hovered image for image traces. [web:588]
    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="graph-3d", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip"),
        ]
    )

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-3d", "hoverData"),
    )
    def display_hover(hoverData):
        # Only show for image points; rely on customdata carrying data URI. [web:588]
        if hoverData is None:
            return False, no_update, no_update
        pt = hoverData["points"][0]
        bbox = pt.get("bbox")
        cd = pt.get("customdata")
        if isinstance(cd, str):  # data URI string for image traces
            children = [html.Img(src=cd, style={"width": "160px", "height": "auto"})]
            return True, bbox, children
        return False, no_update, no_update

    return app

def save_3d_hover_images_html(
    out_path,
    img_xyz,                 # (N_img, 3) numpy array
    txt_xyz,                 # (N_txt, 3) numpy array
    images,                  # list[PIL.Image], len == N_img
    image_classes,           # list[str|int], len == N_img
    text_labels,             # list[str], len == N_txt
    text_classes=None,       # optional list[str|int], len == N_txt (defaults to text_labels)
    title="Embeddings (3D)",
):
    img_xyz = np.asarray(img_xyz)
    txt_xyz = np.asarray(txt_xyz)
    if text_classes is None:
        text_classes = list(text_labels)

    # consistent discrete colors across image/text classes
    all_classes = list(dict.fromkeys(list(image_classes) + list(text_classes)))
    palette = px.colors.qualitative.Plotly
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(all_classes)}  # class -> color [web:583]

    img_uris = [_pil_to_data_uri(im, max_side=160, fmt="JPEG", q=85) for im in images]

    fig = go.Figure()

    # image traces (one per class)
    for cls in all_classes:
        idx = [i for i, c in enumerate(image_classes) if c == cls]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=img_xyz[idx, 0], y=img_xyz[idx, 1], z=img_xyz[idx, 2],
                mode="markers",
                name=str(cls),
                legendgroup=str(cls),
                showlegend=True,
                marker=dict(size=4, color=color_map[cls], opacity=0.85),
                customdata=[img_uris[i] for i in idx],   # per-point data URI
                hoverinfo="none", hovertemplate=None,    # disable default tooltip
            )
        )

    # text traces (one per class) with same colors
    for cls in all_classes:
        idx = [i for i, c in enumerate(text_classes) if c == cls]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=txt_xyz[idx, 0], y=txt_xyz[idx, 1], z=txt_xyz[idx, 2],
                mode="markers+text",
                name=str(cls) + " (text)",
                legendgroup=str(cls),
                showlegend=False,  # keep single legend entry per class
                marker=dict(size=8, color=color_map[cls], symbol="diamond"),
                text=[text_labels[i] for i in idx],
                textposition="top center",
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title, scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=30), hovermode="closest",
    )

    # client-side hover overlay (no server required)
    post_script = r"""
    (function(){
  var gd = document.getElementById('{plot_id}');
  // make the plot div a positioning context
  if (getComputedStyle(gd).position === 'static') gd.style.position = 'relative';

  // floating image
  var holder = document.createElement('img');
  Object.assign(holder.style, {
    position: 'absolute',
    display: 'none',
    pointerEvents: 'none',
    border: '1px solid #888',
    borderRadius: '4px',
    background: '#fff',
    padding: '2px',
    maxWidth: '160px',
    maxHeight: '160px',
    zIndex: 1000
  });
  gd.appendChild(holder);

  function place(ev){
    var rect = gd.getBoundingClientRect();
    var cx = (ev && ev.clientX != null) ? ev.clientX : 0;
    var cy = (ev && ev.clientY != null) ? ev.clientY : 0;
    // position relative to the plot div, with a small offset to the right/below the cursor
    holder.style.left = (cx - rect.left + 12) + 'px';
    holder.style.top  = (cy - rect.top  + 12) + 'px';
  }

  gd.on('plotly_hover', function(e){
    if (!e || !e.points || !e.points.length) return;
    var pt = e.points[0];
    // image traces store a data URI string in customdata
    if (typeof pt.customdata === 'string') {
      holder.src = pt.customdata;
      place(e.event);              // initial placement; may be undefined in some cases
      holder.style.display = 'block';
    } else {
      holder.style.display = 'none';
    }
  });

  gd.on('plotly_unhover', function(){ holder.style.display = 'none'; });
  // keep the preview next to the cursor even when plotly_hover lacked mouse coords
  window.addEventListener('mousemove', function(ev){
    if (holder.style.display === 'block') place(ev);
  }, { passive: true });
})();
    """

    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True, post_script=post_script)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
