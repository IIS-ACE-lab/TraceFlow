# services/visualization/vscode_monitor/model_diagram.py
"""
The planner's Download: a model's diagram as a figure, saved as graph_manager saves its figures.

    python -m services.visualization.vscode_monitor.model_diagram --project P < diagram.json

diagram.json (the planner page sends it): {title, subtitle, facts: [[label, value]], layers: [{name, group, args,
out, trace}], max, name, tag, run, model_id, env, style, theme, arrows, traces, trace: {caption, input}, sparkline: {arguments, orientation,
ticks, legend}, text_rules: {target: {text, bold, italic, color, font: {family, size}}}, export: a graph job's
export: block}.

The figure (Matplotlib; theme: a theme of services/visualization/utils/fig_theme.py, traceflow by default, each layer
group in one palette slot, GROUP_SLOT): the title and subtitle, a line of facts (input, flatten, weights), then, by style:
  full       Full Version: a card per layer (#index name, every argument, the output shape) coloured by its group
             as the planner's cards, and its bar on the planner's log scale (length: output length or features;
             thickness: channels)
  cards      Layer cards: the cards alone
  sparkline  Scaled Sparkline: the bars alone, the index at each bar's start, the arguments outside the bars
             (`arguments` false: none), every layer's wrapped at one measure: the longest text wrapped into as few
             lines as the room allows, as evenly as its arguments split (Ink.even_width), the tick labels (`ticks`), portrait (layers down the page) or
             landscape (across it, the arguments reading upward from the bars' edge), and the layer legend: a solid
             box of each layer type's colour, its name and how many (×n), at `legend` (or none)
arrows: the data flowing from layer to layer. traces (full and cards): each layer's trace line under its card (the
planner's: values scaled to their range, the band of a range of traces, the range at the right), the trace itself
first, trace.caption over them. text_rules: graph_manager's text rule fields over each text's own look (TARGETS);
text replaces the words of title and subtitle only. Every text is measured in its font, so the columns are as wide
as their longest text and nothing is cut short.

Saved: export is a graph job's `export:` block, checked by graph_manager's normalize_export_config and written by
its save_figure (png at export.png.dpi; pdf; svg sanitised and validated; every page the figure plus
export.margin), as <paths.model_preview>/<name>.<format> (params.yaml paths.model_preview, {{env}} the env given;
name: the model, the trained run chosen and the trace settings drawn, e.g. ASCAD_CNNbase__tulip_hold__trace_ascad_t0-9).
Then copied, replacing what is there, into paths.model_out of every DONE run of the model_id whose snapshot layers
are the layers drawn ("auto" alike what it became), as <run>_diagram[__<tag>].<format> (tag: the trace settings
drawn); trace lines through a trained run's weights (run) go into that run's folder only. A run left out is named
(skipped), and so is a run whose folder this user may not write (denied: the container wrote it as root;
services/doc/03_file_permissions.md §3.1 gives it back).

Prints one JSON object: {written, copied, skipped, denied, error}.

Author: Shraddha Mishra
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from services.visualization.utils import fig_theme

DEFAULT_THEME = "traceflow"          # the planner's own light colours
# each layer group's palette slot (fig_theme.SLOTS): under traceflow, the planner cards' colours
GROUP_SLOT = {"convolution": "primary", "pooling": "secondary", "dense": "support", "normalization": "accent",
              "dropout": "neutral", "shape": "dark", "recurrent": "highlight", "attention": "tertiary",
              "merge": "deep", "other": "deep", "input": "key"}
C: dict[str, Any] = {}               # the active colours (set_theme): page, panel, ink, muted, faint, one per group


def set_theme(name: str) -> None:
    """The colours of one theme of fig_theme.py: its page, panel, ink, muted, a faint grid colour, each group's."""
    C.clear()
    C.update(fig_theme.resolve(name, GROUP_SLOT))
    C["faint"] = fig_theme.tint(C["muted"], 0.3, C["page"])


set_theme(DEFAULT_THEME)


MONO, SANS = "DejaVu Sans Mono", "DejaVu Sans"   # Matplotlib's own fonts: every machine draws them the same
BAR = 170.0                          # the bar column of the cards, pt (the planner's SCALE_W px)
SPARK = 220.0                        # the Scaled Sparkline's bar length, pt
ARGS_W = 260.0                       # the room for a layer's arguments beside a portrait sparkline, pt (landscape: SPARK)

STYLES = ("full", "cards", "sparkline")          # Full Version (cards and bars), Layer cards, Scaled Sparkline
ORIENTATIONS = ("portrait", "landscape")
LEGEND_POSITIONS = ("top-right", "bottom-right", "top-left", "none")
# the texts a text rule styles (graph_manager's text_rules fields: text, bold, italic, color, font.family, font.size)
# and how each is drawn when no rule is set; a rule's text replaces the words of title and subtitle only
TARGETS = {
    "title": {"family": SANS, "size": 13, "bold": True, "color": "ink"},      # color: the theme's ink or muted
    "subtitle": {"family": SANS, "size": 9, "color": "muted"},
    "facts": {"family": MONO, "size": 9, "color": "ink"},
    "layer_labels": {"family": MONO, "size": 9, "bold": True, "color": "ink"},
    "arguments": {"family": MONO, "size": 8, "color": "muted"},
    "shapes": {"family": MONO, "size": 9, "color": "ink"},
    "tick_labels": {"family": MONO, "size": 7, "color": "muted"},
    "legend_text": {"family": SANS, "size": 8, "color": "ink"},
    "trace_caption": {"family": SANS, "size": 7.5, "color": "muted"},
}
TEXT_TARGETS = ("title", "subtitle")



def args_line(args: Any) -> str:
    """Every argument, the activation last (the planner's cards show the first four)."""
    args = args if isinstance(args, dict) else {}
    parts = [f"{k} [{','.join(map(str, v))}]" if isinstance(v, list) else f"{k} {v}" for k, v in args.items() if k != "activation"]
    return " · ".join(parts + ([str(args["activation"])] if args.get("activation") else []))


def x_on_log(v: float, top: float, width: float) -> float:
    return width * math.log10(max(1.0, v)) / math.log10(top)


def scale_ticks(top: float, width: float, char: float = 6.0) -> list[float]:
    """Ticks at 1, 10, 100, ... and the axis end; powers of ten give way while their label would touch the end's
    (planner_core.scaleTicks)."""
    out = []
    t = 1
    while t < top:
        out.append(t)
        t *= 10
    while len(out) > 1 and width - x_on_log(out[-1], top, width) < char * (len(str(out[-1])) / 2 + len(_label(top))) + 4:
        out.pop()
    return out + [top]


def _label(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def options(d: dict[str, Any]) -> dict[str, Any]:
    """The drawing's options as the page sends them, checked: style, theme, arrows, traces, sparkline, text_rules."""
    style = str(d.get("style") or "full")
    spark = d.get("sparkline") or {}
    out = {"style": style, "theme": str(d.get("theme") or DEFAULT_THEME), "arrows": bool(d.get("arrows")), "traces": bool(d.get("traces")),
           "arguments": bool(spark.get("arguments", True)), "orientation": str(spark.get("orientation") or "portrait"),
           "ticks": bool(spark.get("ticks", True)), "legend": str(spark.get("legend") or "top-right"),
           "rules": d.get("text_rules") or {}}
    for key, value, allowed in (("style", style, STYLES), ("theme", out["theme"], tuple(fig_theme.names())),
                                ("orientation", out["orientation"], ORIENTATIONS),
                                ("legend", out["legend"], LEGEND_POSITIONS)):
        if value not in allowed:
            raise ValueError(f"diagram {key} {value!r}: one of {', '.join(allowed)}")
    for target, rule in out["rules"].items():
        if target not in TARGETS:
            raise ValueError(f"text_rules target {target!r}: one of {', '.join(TARGETS)}")
        if not isinstance(rule, dict):
            raise ValueError(f"text_rules {target}: a mapping of text, bold, italic, color, font")
    # the form's switch: a rule with enabled false is not applied (the diagram's own look); the key is the form's own
    out["rules"] = {t: {k: v for k, v in r.items() if k != "enabled"} for t, r in out["rules"].items() if r.get("enabled", True)}
    return out


class Ink:
    """One target's font: its default, with what a text rule sets over it; measures text in points."""

    def __init__(self, target: str, rules: dict[str, Any]) -> None:
        from matplotlib.font_manager import FontProperties

        base = TARGETS[target]
        rule = rules.get(target) or {}
        font = rule.get("font") or {}
        family = str(font.get("family") or base["family"])
        if family.startswith("Latin Modern"):
            from services.graph_manager.plot_style import register_tex_fonts

            register_tex_fonts()
        self.size = float(font.get("size") or base["size"])
        self.color = str(rule.get("color") or C[base["color"]])
        self.text = str(rule.get("text") or "") if target in TEXT_TARGETS else ""
        self.prop = FontProperties(family=family, size=self.size,
                                   weight="bold" if rule.get("bold", base.get("bold", False)) else "normal",
                                   style="italic" if rule.get("italic", False) else "normal")

    def width(self, s: str) -> float:
        from matplotlib.textpath import TextToPath

        return TextToPath().get_text_width_height_descent(s, self.prop, ismath=False)[0] if s else 0.0

    def wrap(self, s: str, width: float) -> list[str]:
        """The arguments in lines no wider than width, broken between arguments (one too long stays whole)."""
        lines: list[str] = []
        for part in [p for p in s.split(" · ") if p]:
            if lines and self.width(f"{lines[-1]} · {part}") <= width:
                lines[-1] = f"{lines[-1]} · {part}"
            else:
                lines.append(part)
        return lines

    def even_width(self, texts: list[str], room: float) -> float:
        """The one width every text wraps at: the longest text wrapped into as few lines as `room` allows, as evenly
        as its arguments split (the narrowest width that still needs no more lines); its widest line. The other
        texts wrap at it too, so every line ends at about the same measure."""
        texts = [t for t in texts if t]
        if not texts:
            return room
        longest = max(texts, key=self.width)
        parts = [self.width(p) for p in longest.split(" · ") if p]
        lines = len(self.wrap(longest, room))
        lo, hi = max(parts), max(room, max(parts))
        for _ in range(24):                                     # binary search: wrap is monotone in the width
            mid = (lo + hi) / 2
            if len(self.wrap(longest, mid)) <= lines:
                hi = mid
            else:
                lo = mid
        return max(self.width(s) for s in self.wrap(longest, hi))

    @property
    def line(self) -> float:
        return self.size * 1.25


class Block:
    """Drawing operations in points (y down) and the block's size; drawn later at an offset."""

    def __init__(self) -> None:
        self.ops: list[tuple] = []
        self.w = self.h = 0.0

    def text(self, x, y, s, ink: Ink, **kw):
        self.ops.append(("text", x, y, s, ink, kw))

    def rect(self, x, y, w, h, color, rounded=0.0):
        self.ops.append(("rect", x, y, w, h, color, rounded))

    def line(self, xs, ys, color, lw=0.7, dashed=False):
        self.ops.append(("line", xs, ys, color, lw, dashed))

    def arrow(self, x0, y0, x1, y1):
        self.ops.append(("arrow", x0, y0, x1, y1))

    def band(self, xs, lo, hi, color):
        self.ops.append(("band", xs, lo, hi, color))

    def draw(self, ax, dx: float, dy: float) -> None:
        from matplotlib.patches import FancyBboxPatch, Rectangle

        for op in self.ops:
            kind = op[0]
            if kind == "text":
                _, x, y, s, ink, kw = op
                ax.text(x + dx, y + dy, s, fontproperties=ink.prop, color=ink.color, **{"va": "center", **kw})
            elif kind == "rect":
                _, x, y, w, h, color, rounded = op
                ax.add_patch(FancyBboxPatch((x + dx, y + dy), w, h, boxstyle=f"round,pad=0,rounding_size={rounded}",
                                            facecolor=color, edgecolor="none") if rounded else
                             Rectangle((x + dx, y + dy), w, h, facecolor=color, edgecolor="none"))
            elif kind == "line":
                _, xs, ys, color, lw, dashed = op
                ax.plot([x + dx for x in xs], [y + dy for y in ys], color=color, lw=lw, ls=(0, (2, 3)) if dashed else "-",
                        solid_capstyle="round")
            elif kind == "arrow":
                _, x0, y0, x1, y1 = op
                ax.annotate("", xy=(x1 + dx, y1 + dy), xytext=(x0 + dx, y0 + dy),
                            arrowprops={"arrowstyle": "-|>", "color": C["muted"], "lw": 0.8, "mutation_scale": 7,
                                        "shrinkA": 0, "shrinkB": 0})
            elif kind == "band":
                _, xs, lo, hi, color = op
                ax.fill_between([x + dx for x in xs], [y + dy for y in lo], [y + dy for y in hi], color=color, alpha=0.18, lw=0)


def _rows(d: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, layer in enumerate(d.get("layers") or []):
        out = layer.get("out") if isinstance(layer.get("out"), list) and layer.get("out") else None
        rows.append({"index": i, "name": str(layer.get("name")), "head": f"#{i} {layer.get('name')}",
                     "args": args_line(layer.get("args")), "out": f"({' × '.join(map(str, out))})" if out else "",
                     "ink": C.get(str(layer.get("group") or "other"), C["other"]),
                     "size": (out[1] if len(out) == 2 else out[0]) if out else None,
                     "ch": out[0] if out and len(out) == 2 else 1, "trace": layer.get("trace")})
    return rows


def _thick(ch: float, most: float) -> float:
    """A bar's thickness: the channels, as the planner draws it (2 + 2 log2(ch + 1) px), at most `most`."""
    return max(1.5, min(most, 1.5 + 1.4 * math.log2(ch + 1)))


def _trace(b: Block, x: float, y: float, w: float, h: float, t: dict[str, Any], color: str, cap: Ink) -> None:
    """One trace line (the planner's): its values scaled to their own range across w, the band under a range's mean,
    a dashed zero line when it crosses 0, and its range at the right."""
    values = [v for v in (t.get("values") or []) if isinstance(v, (int, float)) and math.isfinite(v)]
    if not values:
        return
    band = t.get("band") if isinstance(t.get("band"), list) and len(t["band"]) == 2 and len(t["band"][0]) == len(values) else None
    every = values + (band[0] + band[1] if band else [])
    lo, hi = min(every), max(every)
    at_y = lambda v: y + h - 2 - ((v - lo) / (hi - lo) if hi > lo else 0.5) * (h - 4)          # noqa: E731
    xs = [x + (k * w / (len(values) - 1) if len(values) > 1 else w / 2) for k in range(len(values))]
    if lo < 0 < hi:
        b.line([x, x + w], [at_y(0), at_y(0)], C["faint"], 0.6, dashed=True)
    if band:
        b.band(xs, [at_y(v) for v in band[0]], [at_y(v) for v in band[1]], color)
    b.line(xs, [at_y(v) for v in values], color, 0.9)
    b.text(x + w + 6, y + h / 2, f"{_short(t.get('lo', lo))} … {_short(t.get('hi', hi))}", cap)


def _short(v: Any) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:.1e}" if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01) else f"{v:.3g}"


def _cards(d: dict[str, Any], o: dict[str, Any], ink: dict[str, Ink]) -> Block:
    """Layer cards (and, Full Version, each card's bar on the log scale): #index name, every argument, the shape;
    with trace lines, a line under each card (the trace above the first); with arrows, the data flowing down."""
    b, rows, full = Block(), _rows(d), o["style"] == "full"
    traces = o["traces"] and any(r["trace"] for r in rows)
    pad = 0.0
    row = max(17.0, max(ink[k].line for k in ("layer_labels", "arguments", "shapes")) + 5)
    gap = 12.0 if o["arrows"] else 3.0
    name_x = pad + 8
    args_x = name_x + max([ink["layer_labels"].width(r["head"]) for r in rows] + [0]) + 9
    out_end = args_x + max([ink["arguments"].width(r["args"]) for r in rows] + [0]) + 12 + \
        max([ink["shapes"].width(r["out"]) for r in rows] + [0])
    card_end = out_end + 6
    bar_x = card_end + 10
    hi = max(2.0, float(d.get("max") or 2))
    ticks = scale_ticks(hi, BAR, ink["tick_labels"].size * 0.6)
    at = lambda t: min(BAR - 0.5, max(0.5, x_on_log(t, hi, BAR)))                    # noqa: E731
    y = 0.0
    if full:
        b.text(bar_x - 6, y + 4, "output length (log scale) · thickness: channels", ink["tick_labels"], ha="right")
        y += 12
        b.line([bar_x, bar_x + BAR], [y, y], C["muted"])
        for k, t in enumerate(ticks):
            b.line([bar_x + at(t)] * 2, [y - 3, y], C["muted"])
            b.text(bar_x + at(t), y - 6, _label(t), ink["tick_labels"], va="baseline",
                   ha="left" if k == 0 else "right" if k == len(ticks) - 1 else "center")
        y += 6
    source = d.get("trace") or {}
    if traces and source.get("caption"):
        b.text(pad, y + 5, str(source["caption"]), ink["trace_caption"])
        y += 12
    if traces and source.get("input"):
        _trace(b, pad, y, card_end - pad, 20, source["input"], C["input"], ink["trace_caption"])
        y += 24
    th = 22.0
    for i, r in enumerate(rows):
        b.rect(pad, y, card_end - pad, row, fig_theme.tint(r["ink"], 0.2, C["page"]), 3)
        b.rect(pad, y, 3.5, row, r["ink"])
        mid = y + row / 2
        b.text(name_x, mid, r["head"], ink["layer_labels"])
        b.text(args_x, mid, r["args"], ink["arguments"])
        b.text(out_end, mid, r["out"], ink["shapes"], ha="right")
        if full:
            for t in ticks:
                b.line([bar_x + at(t)] * 2, [y, y + row], C["faint"], 0.6, dashed=True)
            if r["size"]:
                bh = _thick(r["ch"], 14)
                b.rect(bar_x, mid - bh / 2, max(2.0, min(BAR, x_on_log(r["size"], hi, BAR))), bh, r["ink"])
        y += row
        if traces and r["trace"]:
            _trace(b, pad, y + 1, card_end - pad, th - 2, r["trace"], r["ink"], ink["trace_caption"])
            y += th
        if o["arrows"] and i < len(rows) - 1:                  # down to the next card, in the gap under this one
            b.arrow(name_x + 10, y + 1, name_x + 10, y + gap - 1)
        y += gap
    b.w = (bar_x + BAR) if full else card_end
    if traces:
        b.w = max(b.w, card_end + 6 + max([ink["trace_caption"].width("-0.00e+00 … -0.00e+00")] + [0]))
    b.h = y - gap
    return b


def _sparkline(d: dict[str, Any], o: dict[str, Any], ink: dict[str, Ink]) -> Block:
    """Scaled Sparkline: one bar per layer on the log scale (length: output length or features; thickness: channels),
    its index at the bar's start; the arguments outside the bars, wrapped; portrait: layers down the page,
    landscape: across it, the arguments reading upward from the bars' edge."""
    b, rows = Block(), _rows(d)
    hi = max(2.0, float(d.get("max") or 2))
    lab, arg, tick = ink["layer_labels"], ink["arguments"], ink["tick_labels"]
    ticks = scale_ticks(hi, SPARK, tick.size * 0.6)
    at = lambda t: min(SPARK - 0.5, max(0.5, x_on_log(t, hi, SPARK)))                # noqa: E731
    room = ARGS_W if o["orientation"] == "portrait" else SPARK    # landscape: no taller than the bars
    even = arg.even_width([r["args"] for r in rows], room) if o["arguments"] else room
    wrapped = [arg.wrap(r["args"], even) if o["arguments"] else [] for r in rows]
    if o["orientation"] == "portrait":
        gutter = 14.0 if o["arrows"] else 0.0
        idx_w = max([lab.width(f"#{r['index']}") for r in rows] + [0]) + 8
        bar_x = gutter + idx_w
        y = 0.0
        if o["ticks"]:
            b.line([bar_x, bar_x + SPARK], [y + 10, y + 10], C["muted"])
            for k, t in enumerate(ticks):
                b.line([bar_x + at(t)] * 2, [y + 7, y + 10], C["muted"])
                b.text(bar_x + at(t), y + 4, _label(t), tick, va="baseline",
                       ha="left" if k == 0 else "right" if k == len(ticks) - 1 else "center")
            y += 14
        gap = 10.0 if o["arrows"] else 2.0
        mids = []
        for r, lines in zip(rows, wrapped):
            h = max(16.0, len(lines) * arg.line + 4)
            mid = y + h / 2
            mids.append((y, mid, h))
            b.text(bar_x - 6, mid, f"#{r['index']}", lab, ha="right")
            if o["ticks"]:
                for t in ticks:
                    b.line([bar_x + at(t)] * 2, [y, y + h], C["faint"], 0.6, dashed=True)
            if r["size"]:
                bh = _thick(r["ch"], 12)
                b.rect(bar_x, mid - bh / 2, max(2.0, min(SPARK, x_on_log(r["size"], hi, SPARK))), bh, r["ink"])
            for k, s in enumerate(lines):
                b.text(bar_x + SPARK + 10, mid + (k - (len(lines) - 1) / 2) * arg.line, s, arg)
            y += h + gap
        if o["arrows"]:                                        # down to the next layer, in the gap between them
            for (y0, _, h0), (y1, _, _) in zip(mids, mids[1:]):
                b.arrow(gutter / 2, y0 + h0 - 2, gutter / 2, y1 + 2)
        b.w = bar_x + SPARK + (10 + max([arg.width(s) for ls in wrapped for s in ls] + [0]) if any(wrapped) else 0)
        b.h = y - gap
        return b
    # landscape: a column per layer, the bars growing up from one baseline, the arguments above them, rotated
    tick_w = max([tick.width(_label(t)) for t in ticks] + [0]) + 8 if o["ticks"] else 0.0
    text_h = max([max([arg.width(s) for s in ls] + [0]) for ls in wrapped] + [0])
    cols = [max(16.0, len(ls) * arg.line + 6, lab.width(f"#{r['index']}") + 4) for r, ls in zip(rows, wrapped)]
    gap = 12.0 if o["arrows"] else 4.0
    base = text_h + (8 if text_h else 0) + SPARK                 # the baseline's y
    x = tick_w
    if o["ticks"]:
        b.line([tick_w - 2, tick_w - 2], [base - SPARK, base], C["muted"])
        for k, t in enumerate(ticks):
            yy = base - at(t)
            b.line([tick_w - 5, tick_w - 2], [yy, yy], C["muted"])
            b.text(tick_w - 7, yy, _label(t), tick, ha="right")
    centres = []
    for r, ls, w in zip(rows, wrapped, cols):
        cx = x + w / 2
        centres.append((x, cx, w))
        if o["ticks"]:
            for t in ticks:
                b.line([x, x + w], [base - at(t)] * 2, C["faint"], 0.6, dashed=True)
        if r["size"]:
            bw = _thick(r["ch"], max(2.0, w - 4))
            bh = max(2.0, min(SPARK, x_on_log(r["size"], hi, SPARK)))
            b.rect(cx - bw / 2, base - bh, bw, bh, r["ink"])
        for k, s in enumerate(ls):                                 # from the bars' top edge, reading upward
            b.text(cx + (k - (len(ls) - 1) / 2) * arg.line, base - SPARK - 6, s, arg, rotation=90,
                   rotation_mode="anchor", ha="left")
        b.text(cx, base + 9, f"#{r['index']}", lab, ha="center")
        x += w + gap
    if o["arrows"]:
        for (x0, c0, w0), (x1, c1, w1) in zip(centres, centres[1:]):
            b.arrow(x0 + w0 - 1, base + 20, x1 + 1, base + 20)
    b.w = x - gap
    b.h = base + 24 + (6 if o["arrows"] else 0)
    return b


def _pack(widths: list[float], sep: float, width: float) -> list[list[int]]:
    """The items, in order, in rows no wider than width (sep between two in a row; one too wide has a row alone)."""
    rows: list[list[int]] = []
    used = 0.0
    for k, w in enumerate(widths):
        if rows and used + sep + w <= width:
            rows[-1].append(k)
            used += sep + w
        else:
            rows.append([k])
            used = w
    return rows


def _legend(d: dict[str, Any], ink: Ink, width: float) -> Block:
    """The layer legend: a solid box of each layer type's colour, its name and how many (×n). In as few rows as the
    page's width allows, filled evenly: the narrowest width that needs no more rows (no long row over a short one)."""
    b, seen = Block(), {}
    for r in _rows(d):
        seen.setdefault(r["name"], [r["ink"], 0])[1] += 1
    items = [(c, f"{name} ×{n}") for name, (c, n) in seen.items()]
    box, sep, line = 7.0, 12.0, ink.line + 3
    widths = [box + 4 + ink.width(label) for _, label in items]
    if not items:
        return b
    count = len(_pack(widths, sep, width))
    lo, hi = max(widths), max(width, max(widths))
    for _ in range(24):
        mid = (lo + hi) / 2
        if len(_pack(widths, sep, mid)) <= count:
            hi = mid
        else:
            lo = mid
    for row, ks in enumerate(_pack(widths, sep, hi)):
        x = 0.0
        y = row * line
        for k in ks:
            colour, label = items[k]
            b.rect(x, y + line / 2 - box / 2, box, box, colour)
            b.text(x + box + 4, y + line / 2, label, ink)
            b.w = max(b.w, x + widths[k])
            x += widths[k] + sep
        b.h = y + line
    return b


def figure(d: dict[str, Any]) -> Any:
    """The diagram as a Matplotlib figure: the header (title, subtitle, facts), the body in its style, the layer
    legend (Scaled Sparkline); laid out in points (one data unit = 1 pt, y down)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = options(d)
    set_theme(o["theme"])
    ink = {t: Ink(t, o["rules"]) for t in TARGETS}
    body = _sparkline(d, o, ink) if o["style"] == "sparkline" else _cards(d, o, ink)
    title = ink["title"].text or str(d.get("title") or "")
    subtitle = ink["subtitle"].text or str(d.get("subtitle") or "")
    facts = [(str(k), str(v)) for k, v in (d.get("facts") or []) if v not in (None, "")]
    f = ink["facts"]
    facts_w = sum(f.width(f"{k} {v}") + 18 for k, v in facts)
    pad = 14.0
    W = max(body.w, ink["title"].width(title) + 24 + ink["subtitle"].width(subtitle), facts_w) + 2 * pad
    legend = _legend(d, ink["legend_text"], W - 2 * pad) if o["style"] == "sparkline" and o["legend"] != "none" else None
    head = pad + ink["title"].line + 4 + (f.line + 6 if facts else 0)
    top = head + (legend.h + 10 if legend and o["legend"].startswith("top") else 0) + 8
    H = top + body.h + (legend.h + 12 if legend and o["legend"] == "bottom-right" else 0) + pad
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor=C["page"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    page = Block()
    page.text(pad, pad + ink["title"].line / 2, title, ink["title"])
    page.text(W - pad, pad + ink["title"].line / 2, subtitle, ink["subtitle"], ha="right")
    fx = pad
    for k, v in facts:
        page.text(fx, pad + ink["title"].line + 4 + f.line / 2, f"{k} {v}", f)
        fx += f.width(f"{k} {v}") + 18
    page.draw(ax, 0, 0)
    body.draw(ax, pad, top)
    if legend:
        lx = pad if o["legend"] == "top-left" else W - pad - legend.w
        ly = head + 4 if o["legend"].startswith("top") else top + body.h + 12
        legend.draw(ax, lx, ly)
    return fig


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _preview_dir(project: str, env: str) -> Path:
    """params.yaml paths.model_preview under the project, {{env}} the env given."""
    from services.side_channel_core.utils.config_loader import get_project_config

    cfg = get_project_config(project, validate_data=False)
    template = str((cfg.get("paths") or {}).get("model_preview") or "")
    if not template:
        raise ValueError("params.yaml paths.model_preview is not set (e.g. \"artifacts/{{env}}/model_preview\")")
    rel = template.replace("{{env}}", env)
    if "{{" in rel:
        raise ValueError(f"paths.model_preview {template!r}: only {{{{env}}}} can be filled here")
    return Path(cfg["project"]["root"]) / rel


def _same(a: Any, b: Any) -> bool:
    """Layer lists alike, an "auto" value alike any (a run's snapshot holds what auto became, e.g. 256 classes)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b or "auto" in (a, b)


def _runs_drawn(project: str, model_id: str, layers: list) -> tuple[list[tuple[str, Path]], list[str]]:
    """The DONE runs of model_id whose snapshot layers are the layers drawn (their model_out folders), and the
    names of the others."""
    from sca.test_request import load_snapshot, resolve_models
    from services.visualization.vscode_monitor.trace_view import runs
    from utils.checkpoint_policy import checkpoint_paths

    drawn = [{str(l.get("name")): l.get("args") or {}} for l in layers]
    same, other = [], []
    for r in runs(project, model_id)["runs"]:
        if r.get("why") or not _same(r.get("layers"), drawn):
            other.append(r["name"])
            continue
        snap = load_snapshot(project, resolve_models(project, [r["name"]])[0])
        same.append((r["name"], checkpoint_paths(snap, model_name=r["name"]).model_out_dir))
    return same, other


def download(project: str, d: dict[str, Any]) -> dict[str, Any]:
    from services.graph_manager.graph_job_validator import normalize_export_config
    from services.graph_manager.plotters import save_figure
    import matplotlib.pyplot as plt

    job = {"id": "model diagram", "export": normalize_export_config({"id": "model diagram", "export": d.get("export") or {}})}
    job["export"]["save_raw_plot_data"] = False
    folder = _preview_dir(project, str(d.get("env") or "dev"))
    tag = _safe(str(d.get("tag") or "")) if d.get("tag") else ""
    fig = figure(d)
    try:
        written = save_figure(fig, folder / _safe(str(d.get("name") or d.get("title") or "model")), job)
    finally:
        plt.close(fig)
    copied: list[str] = []
    skipped: list[str] = []
    denied: list[str] = []
    if d.get("model_id"):
        same, skipped = _runs_drawn(project, str(d["model_id"]), d.get("layers") or [])
        if d.get("traces") and d.get("run"):                  # lines through one run's weights: that run's folder only
            skipped += [name for name, _ in same if name != d["run"]]
            same = [(name, out) for name, out in same if name == d["run"]]
        for name, model_out in same:
            try:
                model_out.mkdir(parents=True, exist_ok=True)
                for f in written:
                    dest = model_out / f"{name}_diagram{'__' + tag if tag else ''}{f.suffix}"
                    shutil.copyfile(f, dest)                  # replaces an earlier diagram
                    copied.append(str(dest))
            except PermissionError:                           # the container wrote the folder as root: guide 03 §3.1
                denied.append(name)
    return {"written": [str(p) for p in written], "copied": copied, "skipped": skipped, "denied": denied}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The planner's Download: a model's diagram, saved as graph_manager saves figures.")
    parser.add_argument("--project", required=True)
    args = parser.parse_args(argv)
    try:
        out = download(args.project, json.load(sys.stdin))
    except Exception as exc:  # noqa: BLE001
        out = {"error": f"{type(exc).__name__}: {exc}", "written": [], "copied": [], "skipped": [], "denied": []}
    print(json.dumps(out))
    return 0 if not out.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
