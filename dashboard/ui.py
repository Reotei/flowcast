"""FlowCast visual system: palette, type, page furniture, the map and chart styling.

What the design is trying to do:
- answer "what needs me right now" first, then show the detail;
- grey by default, colour only where traffic is slow or a patrol is needed;
- one typeface (IBM Plex Sans), numbers in tabular figures, short plain sentences;
- one strong element per page: the status band on Now, the map everywhere it appears.
"""

import html
import math

import plotly.graph_objects as go
import streamlit as st

from store import STATIC

FONT = "'IBM Plex Sans', system-ui, sans-serif"
LEVELS = ["Free-flow", "Moderate", "Heavy", "Severe"]

PALETTE = {
    "light": {
        "bg": "#ffffff", "subtle": "#f4f5f6", "hover": "#f6f7f8", "ink": "#1d2125", "text": "#3d444b",
        "muted": "#6b737b", "rule": "#e6e8eb", "grid": "#eef0f2", "accent": "#2f6db5", "alert": "#c8372d",
        "band": "#17202a", "band_text": "#c9d1da", "map_bg": "#f2f3f4", "ward": "#d6dade",
        "series": ["#2f6db5", "#c9731c", "#8657b5", "#3f9b5a"], "recorded": "#9aa1a8",
        "levels": {"Free-flow": "#b9bfc6", "Moderate": "#e8b04a", "Heavy": "#e07a2e", "Severe": "#c8372d"},
        "ramp": ["#f4f5f6", "#dde6f2", "#b9cde6", "#8eb0d8", "#5f8fc6", "#2f6db5", "#24548c", "#183a61"],
    },
    "dark": {
        "bg": "#15181c", "subtle": "#1e2227", "hover": "#1b1f24", "ink": "#e4e7ea", "text": "#c3c8cd",
        "muted": "#8d949b", "rule": "#2c3137", "grid": "#23282d", "accent": "#5591d8", "alert": "#e0574b",
        "band": "#222a33", "band_text": "#aeb7c1", "map_bg": "#1b1f24", "ward": "#30363d",
        "series": ["#5591d8", "#c07630", "#9a76cf", "#4fa66b"], "recorded": "#6a7178",
        "levels": {"Free-flow": "#5a616a", "Moderate": "#d9a23f", "Heavy": "#d9732c", "Severe": "#d2463b"},
        "ramp": ["#1e2227", "#1d2d40", "#22405f", "#2a5680", "#3a70a6", "#5591d8", "#86b2e6", "#bcd5f2"],
    },
}


def mode():
    """'light' or 'dark', following the viewer's theme."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def colors():
    """Palette for the active mode."""
    return PALETTE[mode()]


def esc(value):
    """HTML-escape any value."""
    return html.escape(str(value))


def html_block(markup):
    """Render trusted HTML built in this module."""
    st.markdown(markup, unsafe_allow_html=True)


# ====================================================================== page chrome
def setup():
    """Logo and page CSS. Call once per run."""
    c = colors()
    logo = str(STATIC / f"logo-{mode()}.svg")
    st.logo(logo, icon_image=logo, size="large")
    st.html(f"""<style>
  .block-container {{ max-width: 1440px; padding: 4.3rem 2rem 3rem; }}
  @media (max-width: 760px) {{ .block-container {{ padding: 4rem .85rem 2rem; }} }}
  h1 {{ font-size: 22px !important; padding: 0 !important; margin: 0 !important; letter-spacing: -.01em; }}
  [data-testid="stCaptionContainer"] p {{ color: {c['muted']}; font-size: 12px; }}
  [data-testid="stTabs"] button p {{ font-size: 14px; }}
  :focus-visible {{ outline: 2px solid {c['accent']} !important; outline-offset: 2px; }}

  /* status band: the one loud element on the Now page */
  .fc-band {{ background:{c['band']}; color:#fff; border-radius:10px; padding:1.25rem 1.5rem 1.2rem;
             display:flex; justify-content:space-between; gap:1.5rem; align-items:flex-end; flex-wrap:wrap; margin-bottom:.9rem; }}
  .fc-band .h {{ font-size:30px; font-weight:600; letter-spacing:-.02em; line-height:1.15; }}
  .fc-band .p {{ color:{c['band_text']}; font-size:15px; margin-top:.35rem; max-width:70ch; line-height:1.45; }}
  .fc-band .t {{ color:{c['band_text']}; font-size:12px; text-align:right; white-space:nowrap; }}
  .fc-band .t b {{ display:block; color:#fff; font-size:22px; font-weight:600; }}
  @media (max-width: 760px) {{ .fc-band .h {{ font-size:24px; }} .fc-band .t {{ text-align:left; }} }}

  .fc-title {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap; }}
  .fc-title > span {{ color:{c['muted']}; font-size:12px; }}
  .fc-lede {{ color:{c['text']}; font-size:14px; margin:.3rem 0 .9rem; max-width:80ch; line-height:1.5; }}
  .fc-lede b {{ color:{c['ink']}; font-weight:600; }}
  .fc-sec {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; margin:1.2rem 0 .45rem; }}
  .fc-sec b {{ font-size:15px; font-weight:600; color:{c['ink']}; }}
  .fc-sec span {{ font-size:12px; color:{c['muted']}; }}

  /* three numbers, the first one leads */
  .fc-nums {{ display:grid; grid-template-columns:1.3fr 1fr 1fr; gap:1rem; margin:.2rem 0 .6rem; max-width:880px; }}
  .fc-num .v {{ font-size:24px; font-weight:600; color:{c['ink']}; line-height:1.1; font-variant-numeric:tabular-nums; }}
  .fc-num:first-child .v {{ font-size:32px; }}
  .fc-num .v small {{ font-size:13px; font-weight:400; color:{c['muted']}; margin-left:.25rem; }}
  .fc-num .l {{ font-size:13px; color:{c['text']}; margin-top:.2rem; }}
  .fc-num .d {{ font-size:12px; color:{c['muted']}; margin-top:.1rem; }}
  .fc-up {{ color:{c['alert']} !important; }}

  /* attention list */
  .fc-att {{ display:grid; grid-template-columns:4px minmax(0,1fr) auto; gap:.25rem .8rem; padding:.55rem .2rem .55rem 0;
            border-top:1px solid {c['grid']}; align-items:center; }}
  .fc-att:first-of-type {{ border-top:none; }}
  .fc-att i {{ align-self:stretch; border-radius:2px; }}
  .fc-att b {{ font-size:14px; font-weight:600; color:{c['ink']}; display:block; }}
  .fc-att span {{ font-size:13px; color:{c['text']}; }}
  .fc-att em {{ font-style:normal; font-size:15px; font-weight:600; color:{c['ink']}; font-variant-numeric:tabular-nums; }}
  .fc-calm {{ font-size:14px; color:{c['text']}; padding:.6rem 0; line-height:1.5; }}

  /* every road: a list that turns into two-line blocks on phones */
  .fc-roads {{ font-size:13px; color:{c['text']}; }}
  .fc-road {{ display:grid; grid-template-columns:4px minmax(150px,2.2fr) 6.5rem 4.5rem 5.5rem 7rem 6.5rem;
             gap:.6rem; align-items:center; padding:.32rem .2rem; border-bottom:1px solid {c['grid']}; }}
  .fc-road:hover {{ background:{c['hover']}; }}
  .fc-road.head {{ color:{c['muted']}; font-size:12px; border-bottom:1px solid {c['rule']}; }}
  .fc-road.head:hover {{ background:none; }}
  .fc-road i.line {{ height:20px; border-radius:2px; }}
  .fc-road .n {{ color:{c['ink']}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .fc-road .n span {{ color:{c['muted']}; margin-right:.4rem; }}
  .fc-road .r {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .fc-road .k {{ color:{c['ink']}; }}
  .fc-zone {{ font-size:12px; color:{c['muted']}; padding:.8rem 0 .3rem; border-bottom:1px solid {c['rule']}; }}
  .fc-zone b {{ color:{c['ink']}; }}
  .fc-sq {{ display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:.4rem; }}
  .fc-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:{c['alert']}; margin-left:.35rem; }}
  .fc-key {{ display:flex; flex-wrap:wrap; gap:.3rem 1rem; font-size:12px; color:{c['muted']}; margin:.5rem 0 0; }}
  @media (max-width: 760px) {{
    .fc-road {{ grid-template-columns:4px minmax(0,1fr) auto; grid-template-areas:"l n v" "l s r"; row-gap:.1rem; }}
    .fc-road i.line {{ grid-area:l; height:auto; align-self:stretch; }}
    .fc-road .n {{ grid-area:n; }} .fc-road .lv {{ grid-area:s; }} .fc-road .vol {{ grid-area:v; }}
    .fc-road .risk {{ grid-area:r; }} .fc-road .vs, .fc-road .bar {{ display:none; }}
    .fc-road.head {{ display:none; }}
    .fc-nums {{ grid-template-columns:1fr 1fr; }} .fc-num:first-child {{ grid-column:1 / -1; }}
  }}

  .fc-check {{ display:grid; grid-template-columns:5.5rem minmax(0,17rem) 1fr; gap:.8rem; padding:.45rem 0;
              border-bottom:1px solid {c['grid']}; font-size:13px; color:{c['text']}; }}
  .fc-check .ok {{ color:{c['series'][3]}; font-weight:600; }} .fc-check .no {{ color:{c['alert']}; font-weight:600; }}
  @media (max-width: 760px) {{ .fc-check {{ grid-template-columns:5.5rem 1fr; }} .fc-check span:last-child {{ grid-column:2; }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>""")


def band(headline, detail, label, value):
    """Status band: big plain headline, one line of detail, and the time it is about."""
    html_block(f'<div class="fc-band"><div><div class="h">{esc(headline)}</div><div class="p">{detail}</div></div>'
               f'<div class="t">{esc(label)}<b>{esc(value)}</b></div></div>')


def title(text, lede=None, note=None):
    """Page title for the quieter pages."""
    note_html = f"<span>{esc(note)}</span>" if note else ""
    html_block(f'<div class="fc-title"><h1>{esc(text)}</h1>{note_html}</div>')
    if lede:
        html_block(f'<p class="fc-lede">{lede}</p>')


def section(text, note=None):
    """Section heading."""
    note_html = f"<span>{esc(note)}</span>" if note else ""
    html_block(f'<div class="fc-sec"><b>{esc(text)}</b>{note_html}</div>')


def numbers(items):
    """Three headline numbers: dicts with v, l, optional unit, d (detail), up (detail is bad news)."""
    cells = []
    for item in items:
        unit = f"<small>{esc(item['unit'])}</small>" if item.get("unit") else ""
        css = ' class="d fc-up"' if item.get("up") else ' class="d"'
        detail = f'<div{css}>{esc(item["d"])}</div>' if item.get("d") else ""
        cells.append(f'<div class="fc-num"><div class="v">{esc(item["v"])}{unit}</div>'
                     f'<div class="l">{esc(item["l"])}</div>{detail}</div>')
    html_block(f'<div class="fc-nums">{"".join(cells)}</div>')


def attention(items, calm_text):
    """'Needs attention' list: dicts with colour, name, reason, value. Shows calm_text when empty."""
    if not items:
        html_block(f'<div class="fc-calm">{esc(calm_text)}</div>')
        return
    rows = "".join(f'<div class="fc-att"><i style="background:{item["colour"]}"></i><div><b>{esc(item["name"])}</b>'
                   f'<span>{esc(item["reason"])}</span></div><em>{esc(item["value"])}</em></div>' for item in items)
    html_block(rows)


def level_key(patrol=True):
    """Legend for the congestion colours."""
    c = colors()
    items = "".join(f'<span><i class="fc-sq" style="background:{c["levels"][level]}"></i>{level}</span>' for level in LEVELS)
    if patrol:
        items += '<span><i class="fc-dot" style="margin:0 .35rem 0 0"></i>patrol suggested</span>'
    return f'<div class="fc-key">{items}</div>'


def road_list(rows):
    """Every road in corridor order: rows are dicts with zone, road_id, name, level, volume, vs_usual,
    vc, risk, flag (risk/flag may be None)."""
    c = colors()
    parts = ['<div class="fc-roads"><div class="fc-road head"><span></span><span>Road</span><span>Level</span>'
             '<span class="r">Vehicles</span><span class="r">vs usual</span><span>Capacity used</span>'
             '<span class="r">Accident risk</span></div>']
    zone = None
    for row in rows:
        if row["zone"] != zone:
            zone = row["zone"]
            members = [r for r in rows if r["zone"] == zone]
            slow = sum(r["level"] in ("Heavy", "Severe") for r in members)
            parts.append(f'<div class="fc-zone"><b>{esc(zone)} stretch</b>, {len(members)} roads, {slow} slow</div>')
        colour = c["levels"][row["level"]]
        bar_colour = colour if row["level"] != "Free-flow" else c["recorded"]
        width = min(row["vc"] / 1.4, 1) * 100
        risk = "" if row["risk"] is None else f'{row["risk"]:.1%}'
        dot = '<i class="fc-dot" title="Patrol suggested"></i>' if row["flag"] else ""
        vs = "" if row["vs_usual"] is None else f'{row["vs_usual"]:+.0%}'
        parts.append(
            f'<div class="fc-road"><i class="line" style="background:{colour}"></i>'
            f'<span class="n"><span>{esc(row["road_id"][-3:])}</span>{esc(row["name"])}</span>'
            f'<span class="lv"><i class="fc-sq" style="background:{colour}"></i>{esc(row["level"])}</span>'
            f'<span class="r k vol">{row["volume"]:,.0f}</span><span class="r vs">{vs}</span>'
            f'<span class="bar"><svg width="100%" height="8" viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true">'
            f'<rect width="100" height="8" rx="1.5" fill="{c["grid"]}"/><rect width="{width:.1f}" height="8" rx="1.5" fill="{bar_colour}"/>'
            f'<line x1="71.4" x2="71.4" y1="0" y2="8" stroke="{c["muted"]}"/></svg></span>'
            f'<span class="r k risk">{risk}{dot}</span></div>')
    parts.append("</div>")
    html_block("".join(parts) + level_key())


def checks(items):
    """Success criteria: (label, result, passed)."""
    html_block("".join(f'<div class="fc-check"><span class="{"ok" if ok else "no"}">{"Met" if ok else "Not met"}</span>'
                       f'<span>{esc(label)}</span><span>{esc(result)}</span></div>' for label, result, ok in items))


# ====================================================================== map
def _map_layout(segments, wards, streets, height):
    """Shared map settings: fitted to the corridor, street tiles when reachable, ward outlines otherwise."""
    c = colors()
    lat_span = segments["latitude"].max() - segments["latitude"].min()
    lon_span = segments["longitude"].max() - segments["longitude"].min()
    # MapLibre shows 360 degrees of longitude across 512 * 2**zoom pixels; leave a 30% margin around the roads.
    zoom = min(math.log2(360 * 800 / 512 / max(lon_span * 1.3, 1e-3)),
               math.log2(300 * height / 512 / max(lat_span * 1.3, 1e-3)))
    if streets:
        style, layers = ("carto-darkmatter" if mode() == "dark" else "carto-positron"), []
    else:
        style = {"version": 8, "sources": {}, "layers": [{"id": "bg", "type": "background",
                                                           "paint": {"background-color": c["map_bg"]}}]}
        layers = [{"source": wards, "type": "line", "color": c["ward"], "line": {"width": 0.7}, "below": "traces"}]
    return {"map": {"style": style, "layers": layers, "zoom": zoom,
                    "center": {"lat": segments["latitude"].mean(), "lon": segments["longitude"].mean()}},
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0}, "height": height, "showlegend": False,
            "paper_bgcolor": "rgba(0,0,0,0)", "font": {"family": FONT},
            "hoverlabel": {"bgcolor": c["bg"], "bordercolor": c["rule"], "font": {"family": FONT, "color": c["ink"]}}}


def _marker_sizes(volumes):
    """Marker area grows with volume."""
    top = max(max(volumes), 1)
    return [9 + 15 * math.sqrt(max(v, 0) / top) for v in volumes]


def corridor_map(segments, rows, wards, streets, height=600):
    """Every road as a dot on the map, coloured by level and sized by vehicles. Red halo = patrol."""
    c = colors()
    lat = [segments.loc[r["road_id"], "latitude"] for r in rows]
    lon = [segments.loc[r["road_id"], "longitude"] for r in rows]
    hover = [f"<b>{esc(r['name'])}</b><br>{r['level']}, {r['volume']:,.0f} vehicles"
             + (f"<br>{r['vs_usual']:+.0%} on usual" if r["vs_usual"] is not None else "")
             + (f"<br>Accident risk {r['risk']:.1%}" if r["risk"] is not None else "") for r in rows]
    flagged = [i for i, r in enumerate(rows) if r["flag"]]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(lat=[lat[i] for i in flagged], lon=[lon[i] for i in flagged], mode="markers",
                                marker={"size": [s + 12 for s in (_marker_sizes([r["volume"] for r in rows])[i] for i in flagged)],
                                        "color": c["alert"], "opacity": 0.28}, hoverinfo="skip"))
    fig.add_trace(go.Scattermap(lat=lat, lon=lon, mode="markers", customdata=[r["road_id"] for r in rows],
                                marker={"size": _marker_sizes([r["volume"] for r in rows]),
                                        "color": [c["levels"][r["level"]] for r in rows], "opacity": 0.95},
                                hovertext=hover, hoverinfo="text"))
    fig.update_layout(**_map_layout(segments, wards, streets, height))
    return fig


def day_map(segments, times, levels, volumes, wards, streets, start_index, height=470):
    """Animated map through one day: press play or drag the slider (runs in the browser, no reloads)."""
    c = colors()
    order = list(segments.index)
    def marker(i):
        return {"size": _marker_sizes([volumes[i][r] for r in order]),
                "color": [c["levels"][levels[i][r]] for r in order], "opacity": 0.95}
    def text(i):
        return [f"<b>{esc(segments.loc[r, 'road_name'])}</b><br>{times[i]}: {levels[i][r]}, {volumes[i][r]:,.0f} vehicles"
                for r in order]
    fig = go.Figure(go.Scattermap(lat=segments["latitude"], lon=segments["longitude"], mode="markers",
                                  marker=marker(start_index), hovertext=text(start_index), hoverinfo="text"))
    fig.frames = [go.Frame(name=t, data=[go.Scattermap(marker=marker(i), hovertext=text(i))]) for i, t in enumerate(times)]
    fig.update_layout(**_map_layout(segments, wards, streets, height))
    button = {"bgcolor": c["bg"], "bordercolor": c["rule"], "font": {"color": c["ink"], "size": 12}}
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 70},
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0, "y": -0.04, "xanchor": "left", "yanchor": "top",
                      "showactive": False, "pad": {"r": 8, "t": 4}, **button,
                      "buttons": [{"label": "Play", "method": "animate",
                                   "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True,
                                                   "transition": {"duration": 0}}]},
                                  {"label": "Pause", "method": "animate",
                                   "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}]}],
        sliders=[{"active": start_index, "x": 0.13, "len": 0.87, "y": -0.02, "yanchor": "top", "pad": {"t": 4},
                  "currentvalue": {"prefix": "", "font": {"color": c["ink"], "size": 13}, "xanchor": "right"},
                  "font": {"color": "rgba(0,0,0,0)", "size": 1}, "bgcolor": c["grid"], "bordercolor": c["rule"],
                  "activebgcolor": c["accent"], "ticklen": 3, "tickcolor": c["rule"],
                  "steps": [{"label": t, "method": "animate",
                             "args": [[t], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                             "name": t} for t in times]}])
    return fig


def map_note(streets):
    """Attribution for the offline base layer."""
    if not streets:
        st.caption("Ward outlines: DataMeet, CC BY-SA 2.5 IN. Street maps load when the server has internet access.")


# ====================================================================== charts
def figure():
    """Empty Plotly figure."""
    return go.Figure()


def ramp():
    """Sequential colourscale for magnitude heatmaps."""
    values = colors()["ramp"]
    return [[i / (len(values) - 1), value] for i, value in enumerate(values)]


def level_scale():
    """Stepped colourscale for congestion codes 0 to 3."""
    scale = []
    for code, level in enumerate(LEVELS):
        scale += [[code / 4, colors()["levels"][level]], [(code + 1) / 4, colors()["levels"][level]]]
    return scale


def style(fig, height=280, y=None, x=None, legend=True, hover="x unified"):
    """Quiet chart style: no frame, faint horizontal grid, legend under the plot."""
    c = colors()
    fig.update_layout(
        template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "size": 12, "color": c["text"]}, margin={"l": 4, "r": 8, "t": 6, "b": 4},
        hovermode=hover, showlegend=legend, bargap=0.35,
        hoverlabel={"bgcolor": c["bg"], "bordercolor": c["rule"], "font": {"family": FONT, "color": c["ink"]}},
        legend={"orientation": "h", "x": 0, "y": -0.3 if x else -0.14, "yanchor": "top", "bgcolor": "rgba(0,0,0,0)"})
    fig.update_xaxes(showgrid=False, linecolor=c["rule"], ticks="outside", ticklen=3, tickcolor=c["rule"],
                     tickfont={"color": c["muted"]}, title={"text": x}, automargin=True, zeroline=False)
    fig.update_yaxes(gridcolor=c["grid"], linecolor="rgba(0,0,0,0)", tickfont={"color": c["muted"]},
                     title={"text": y}, automargin=True, zeroline=False)
    return fig


def show(fig, **kwargs):
    """Render a chart without the floating toolbar."""
    return st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": True}, **kwargs)


def bars(x, y, fmt, height=220, colour=None, y_title=None, horizontal=False):
    """Small labelled bar chart."""
    c = colors()
    fig = figure()
    fig.add_trace(go.Bar(x=y if horizontal else x, y=x if horizontal else y, orientation="h" if horizontal else "v",
                         marker_color=colour or c["accent"], text=[fmt.format(v) for v in y], textposition="outside",
                         cliponaxis=False, textfont={"size": 11, "color": c["text"]}, hovertemplate="%{text}<extra></extra>"))
    style(fig, height=height, y=y_title, legend=False, hover="closest")
    if horizontal:
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)").update_xaxes(showgrid=True, gridcolor=c["grid"], range=[0, max(y) * 1.18])
    show(fig)
