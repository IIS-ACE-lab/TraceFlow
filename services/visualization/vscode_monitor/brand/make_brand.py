# services/visualization/vscode_monitor/brand/make_brand.py
"""
Builds the TraceFlow brand files in this folder from one definition:

    traceflow-social.png         1280x640 GitHub social preview: lockup + the paper's subtitle
    traceflow-lockup-light.svg   lockup for light backgrounds (README title, GitHub light theme)
    traceflow-lockup-dark.svg    lockup for dark backgrounds (README title, GitHub dark theme)
    traceflow-icon.png           256x256 icon of the VS Code extension Shraddha-Mishra.traceflow-monitor
                                 (Extensions view): the mark on the social card's dark tile; ../pack.py puts
                                 it in the .vsix as media/icon.png, beside traceflow.svg (the Activity Bar icon)
    traceflow-share-{dark,light}.png the status page's button on the TraceFlow Monitor title line (dark, light themes):
                                 the share mark in VS Code's focus blue with a soft halo, as the planner's + glows on
                                 hover (text-shadow 0 0 4px, 0 0 10px); VS Code draws that line, so the glow is drawn in
    traceflow-web-{180,192,512}.png  the status page's home-screen icon (iPhone 180, Android 192 and 512): the
                                 Activity Bar icon (traceflow.svg: its 24-unit box, its stroke) on a full square
                                 of the dark tile, corners left square: the phone rounds them

The lockup is the mark (a power trace running into a forward chevron, the
same geometry as traceflow.svg and the VS Code icon) beside the wordmark:
"Trace" in Inter Display Thin, "Flow" in Inter Display ExtraBold. The SVGs
hold the letters as outlines, so they look the same where Inter is not
installed (GitHub).

    python services/visualization/vscode_monitor/brand/make_brand.py        (Inter installed: /usr/share/fonts/opentype/inter)
    python services/visualization/vscode_monitor/brand/make_brand.py icon   only the extension icon (no font needed)
    python services/visualization/vscode_monitor/brand/make_brand.py web    only the status page's home-screen icons
    python services/visualization/vscode_monitor/brand/make_brand.py share  only the share button's icons

Author: Shraddha Mishra
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

HERE = Path(__file__).resolve().parent
FONTS = Path("/usr/share/fonts/opentype/inter")
THIN = FontProperties(fname=str(FONTS / "InterDisplay-Thin.otf"))           # Trace: the faint measured signal
HEAVY = FontProperties(fname=str(FONTS / "InterDisplay-ExtraBold.otf"))     # Flow: the pipeline that carries it
BODY = FontProperties(fname=str(FONTS / "Inter-Regular.otf"))
SUBTITLE = ("An Extensible Framework for Reproducible", "Deep Learning Side-Channel Analysis")
INK = {"light": "#1F2328", "dark": "#E6EDF3"}                              # GitHub's text colours
CARD_BG, CARD_SUB = "#0D1117", "#8B949E"
TRACE = [(2, 12), (4.5, 12), (6, 8), (8, 16), (10, 5), (12, 15), (13.5, 11), (15, 11)]   # mark, 24-unit box
CHEVRON = [(17, 7), (22, 12), (17, 17)]
MARK_STROKE = 1.6                                                           # the icon's stroke, in mark units
EXTENSION_ICON = HERE / "traceflow-icon.png"


def lockup(size: float):
    """Geometry of the lockup with the wordmark baseline at y=0 (y up): mark polylines, stroke, word paths."""
    trace, flow = TextPath((0, 0), "Trace", size=size, prop=THIN), TextPath((0, 0), "Flow", size=size, prop=HEAVY)
    cap = TextPath((0, 0), "T", size=size, prop=HEAVY).get_extents().height
    scale = cap * 1.25 / 12.0                      # the mark spans 12 units vertically: 1.25 cap heights
    mid = cap / 2
    marks = [[((x - 2) * scale, mid - (y - 11) * scale) for x, y in pts] for pts in (TRACE, CHEVRON)]
    x_trace = 20 * scale + size * 0.28
    x_flow = x_trace + trace.get_extents().x1 + size * 0.02
    words = [Affine2D().translate(x_trace, 0).transform_path(trace), Affine2D().translate(x_flow, 0).transform_path(flow)]
    return marks, MARK_STROKE * scale, words


def social() -> None:
    w, h = 1280, 640
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=CARD_BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    marks, stroke, words = lockup(132)
    x1 = max(p.get_extents().x1 for p in words)
    dx, base = (w - x1) / 2, 330
    for pts in marks:
        xy = [(dx + x, base + y) for x, y in pts]
        ax.add_patch(PathPatch(MPath(xy, [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 1)), fill=False,
                               edgecolor=INK["dark"], lw=stroke * 72 / 100, capstyle="round", joinstyle="round"))
    for p in words:
        ax.add_patch(PathPatch(Affine2D().translate(dx, base).transform_path(p), facecolor=INK["dark"], edgecolor="none"))
    for i, line in enumerate(SUBTITLE):
        tp = TextPath((0, 0), line, size=34, prop=BODY)
        ax.add_patch(PathPatch(Affine2D().translate((w - tp.get_extents().x1) / 2, 222 - i * 52).transform_path(tp),
                               facecolor=CARD_SUB, edgecolor="none"))
    fig.savefig(HERE / "traceflow-social.png", facecolor=CARD_BG)
    plt.close(fig)


def extension_icon(px: int = 256) -> None:
    """The VS Code extension's icon: the mark in the dark ink on the social card's tile, rounded corners clear."""
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, px); ax.set_ylim(0, px); ax.axis("off")
    pad, r = px * 0.02, px * 0.18                    # the tile fills the canvas; corner radius like VS Code's own icons
    ax.add_patch(FancyBboxPatch((pad, pad), px - 2 * pad, px - 2 * pad, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=CARD_BG, edgecolor="none"))
    # the mark spans x 2..22 and y 5..17 of its 24-unit box: centre it, 70 % of the tile wide
    scale = 0.70 * px / 20
    cx, cy = (2 + 22) / 2, (5 + 17) / 2
    for pts in (TRACE, CHEVRON):
        xy = [(px / 2 + (x - cx) * scale, px / 2 - (y - cy) * scale) for x, y in pts]
        ax.add_patch(PathPatch(MPath(xy, [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 1)), fill=False,
                               edgecolor=INK["dark"], lw=MARK_STROKE * scale * 72 / 100,
                               capstyle="round", joinstyle="round"))
    fig.savefig(EXTENSION_ICON, transparent=True)
    plt.close(fig)


WEB_ICONS = {px: HERE / f"traceflow-web-{px}.png" for px in (180, 192, 512)}
SHARE_ICONS = {"dark": ((140, 200, 255), (55, 148, 255)), "light": ((0, 95, 184), (0, 120, 212))}   # core, glow: focusBorder


def share_icons() -> None:
    """The share button's icons, 32 px (16 CSS px at 2x): the share mark (three nodes, two links) at 62 % of the box, in
    a halo of two Gaussian blurs of its glow colour, strengthened as stacked text-shadows are (wide sigma 10, near 4,
    drawn at 4x), under the mark in its core colour."""
    from PIL import Image, ImageDraw, ImageFilter

    big, k = 128, 0.62

    def mark(color):
        im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d, u = ImageDraw.Draw(im), big / 16
        at = lambda x, y: ((8 + (x - 8) * k) * u, (8 + (y - 8) * k) * u)   # noqa: E731
        a, b, e = at(11.5, 3.8), at(4.5, 8), at(11.5, 12.2)
        d.line([b, a], fill=color, width=int(1.7 * k * u)); d.line([b, e], fill=color, width=int(1.7 * k * u))
        r = 2.1 * k * u
        for x, y in (a, b, e):
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
        return im

    def glow(im, sigma, gain):
        blur = im.filter(ImageFilter.GaussianBlur(sigma))
        blur.putalpha(blur.getchannel("A").point(lambda v: min(255, int(v * gain))))
        return blur

    for name, (core, halo) in SHARE_ICONS.items():
        halo_mark = mark(halo + (255,))
        out = Image.alpha_composite(Image.new("RGBA", (big, big), (0, 0, 0, 0)), glow(halo_mark, 10, 4.0))
        out = Image.alpha_composite(out, glow(halo_mark, 4, 3.0))
        Image.alpha_composite(out, mark(core + (255,))).resize((32, 32), Image.LANCZOS).save(HERE / f"traceflow-share-{name}.png")


def web_icons() -> None:
    """The status page's home-screen icons: traceflow.svg drawn as the Activity Bar draws it (its 24-unit box onto the
    whole square, stroke 1.6 units, round caps), the light ink on the dark tile, full bleed (phones round the corners)."""
    for px, out in WEB_ICONS.items():
        fig = plt.figure(figsize=(px / 100, px / 100), dpi=100, facecolor=CARD_BG)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 24); ax.set_ylim(24, 0); ax.axis("off")   # SVG's y-down box
        unit = px / 24                                       # pixels per mark unit
        for pts in (TRACE, CHEVRON):
            ax.add_patch(PathPatch(MPath(pts, [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)), fill=False,
                                   edgecolor=INK["dark"], lw=MARK_STROKE * unit * 72 / 100, capstyle="round", joinstyle="round"))
        fig.savefig(out, facecolor=CARD_BG)
        plt.close(fig)


def _svg_d(path: MPath) -> str:
    """SVG path data of a y-up matplotlib path, flipped to SVG's y-down."""
    out, v, codes, i = [], path.vertices, path.codes, 0
    pt = lambda k: "%.2f %.2f" % (v[k][0], -v[k][1])  # noqa: E731
    while i < len(v):
        c = codes[i]
        if c == MPath.MOVETO:
            out.append("M" + pt(i)); i += 1
        elif c == MPath.LINETO:
            out.append("L" + pt(i)); i += 1
        elif c == MPath.CURVE3:
            out.append("Q%s %s" % (pt(i), pt(i + 1))); i += 2
        elif c == MPath.CURVE4:
            out.append("C%s %s %s" % (pt(i), pt(i + 1), pt(i + 2))); i += 3
        elif c == MPath.CLOSEPOLY:
            out.append("Z"); i += 1
        else:
            i += 1
    return "".join(out)


def svgs() -> None:
    marks, stroke, words = lockup(100)
    xs = [x for pts in marks for x, _ in pts] + [x for p in words for x, _ in p.vertices]
    ys = [-y for pts in marks for _, y in pts] + [-y for p in words for _, y in p.vertices]
    m = stroke / 2 + 4                              # round caps and a small even margin
    x0, y0, w, h = min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m
    poly = "".join('<polyline points="%s"/>' % " ".join("%.2f,%.2f" % (x, -y) for x, y in pts) for pts in marks)
    d = "".join(_svg_d(p) for p in words)
    for name, ink in INK.items():
        (HERE / f"traceflow-lockup-{name}.svg").write_text(
            f"<!-- services/visualization/vscode_monitor/brand/traceflow-lockup-{name}.svg: the TraceFlow lockup (mark + wordmark, Trace in Inter "
            f"Display Thin, Flow in ExtraBold, as outlines) for {name} backgrounds; made by make_brand.py -->\n"
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {y0:.1f} {w:.1f} {h:.1f}" width="{w:.0f}" '
            f'height="{h:.0f}" role="img" aria-label="TraceFlow">\n<title>TraceFlow</title>\n'
            f'<g fill="none" stroke="{ink}" stroke-width="{stroke:.2f}" stroke-linecap="round" stroke-linejoin="round">'
            f"{poly}</g>\n<path fill=\"{ink}\" d=\"{d}\"/>\n</svg>\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["icon"]:
        extension_icon()
        print("written:", EXTENSION_ICON)
    elif sys.argv[1:] == ["share"]:
        share_icons()
        print("written: traceflow-share-dark.png, traceflow-share-light.png")
    elif sys.argv[1:] == ["web"]:
        web_icons()
        print("written:", ", ".join(p.name for p in WEB_ICONS.values()))
    else:
        social()
        svgs()
        extension_icon()
        web_icons()
        share_icons()
        print("written:", ", ".join(sorted(p.name for p in HERE.glob("traceflow-*") if p.suffix in (".png", ".svg"))))
