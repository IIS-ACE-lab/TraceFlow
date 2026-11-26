# services/visualization/vscode_monitor/brand/make_panel_diagram.py
"""Draw the TraceFlow Monitor's drop-down panels, numbered, with their legend at the right, and the
panel icon (the TraceFlow mark) one clicks to open it, in the Activity Bar at the panel's top left
(README §1.5). The share button sits on the title line, as traceflow-share-{light,dark}.png draws it
(make_brand.py: the mark, its halo as two wider, fainter strokes); its view is not drawn. The [Running] tile is chosen: its run's card is listed under it, drawn while the run's
test attacks (panel.js runCardHtml, phaseHtml, liveHtml): the orange attack bar in the epoch's place, the train
bar under it, the test values so far beside the last epoch's and the key-recovery line (core/monitor.py).
Buttons, section titles, tiles and lights are drawn as webview/panel.js and panel.css draw them: the
same labels, letter case and colours (VS Code's Light Modern and Dark Modern themes; the status lights
in panel.css's fixed colours). Writes, next to this file:

    TraceFlow_Monitor_light.svg   light theme, with traceflow-mark-light.svg
    TraceFlow_Monitor_dark.svg    dark theme, with traceflow-mark-dark.svg
    traceflow_panel.svg           either, by the viewer's system setting (prefers-color-scheme)

The mark's paths and colour are read from traceflow-mark-{light,dark}.svg.

    python services/visualization/vscode_monitor/brand/make_panel_diagram.py
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

BRAND = Path(__file__).resolve().parent
OUTS = {"light": BRAND / "TraceFlow_Monitor_light.svg", "dark": BRAND / "TraceFlow_Monitor_dark.svg",
        "auto": BRAND / "traceflow_panel.svg"}

AB = 48                              # the Activity Bar at the panel's left, holding the TraceFlow mark
EDGE = 1                             # no outer border: the README shows the picture at width 100%, so its edges
                                     # are the text's margins (1 px keeps the frame's stroke inside)
PX, PW = EDGE + AB, 660              # panel left and width; the canvas ends where the legend's longest line does
BADGE_X = PX + PW + 24               # numbered badges, between the panel and the legend
LX = BADGE_X + 26                    # legend left

# Light Modern / Dark Modern values of the theme variables panel.css reads.
LIGHT = """
  .bg{fill:#ffffff}.panel{fill:#f8f8f8;stroke:#e5e5e5}.bar{fill:#f0f0f0}
  .sec{fill:none;stroke:#e5e5e5}.card{fill:none;stroke:#cecece}
  .t{fill:#3b3b3b}.m{fill:#717171}.h{fill:#3b3b3b;font-weight:600}.cap{fill:#3b3b3b;letter-spacing:.5px}
  .link{fill:#005fb8}
  .btn{fill:#e5e5e5}.bt{fill:#3b3b3b}.dt{fill:#c72e0f}.chev{fill:#616161}
  .lead{stroke:#424a53;stroke-width:1.6;stroke-dasharray:3 3;fill:none}.guide{stroke:#d0d7de;stroke-width:1;fill:none}
  .badge{fill:#005fb8}.bn{fill:#ffffff;font-weight:700}
  .lt{fill:#1f2328;font-weight:600;font-size:12px}.desc{fill:#59636e}
  .grey{fill:#8b949e;opacity:.7}
  .spark{fill:none;stroke:#005fb8;stroke-width:1.4}.prog{fill:#005fb8}.track{fill:#e5e5e5}
  .tile{fill:none;stroke:#e5e5e5}.focus{fill:none;stroke:#005fb8;stroke-width:2}
  .cq{fill:#cca700}.cr{fill:#1a85ff}.cd{fill:#388a34}.cf{fill:#e51400}
  .sq{stroke:#cca700}.sr{stroke:#1a85ff}.sd{stroke:#388a34}.sf{stroke:#e51400}
  .pill{fill:none;stroke:#0e70c0}.pt{fill:#0e70c0}.pn{fill:none;stroke:#717171}.pf{fill:none;stroke:#c72e0f}
  .pd{fill:none;stroke:#388a34}.it{fill:#717171;font-style:italic}
  .grid{stroke:#d4d4d4;fill:none}
  .abar{fill:#f8f8f8;stroke:#e5e5e5}.accent{fill:#005fb8}
  .shc{stroke:#005fb8;fill:#005fb8}.sh1{stroke:#d3e2ed;fill:#d3e2ed}.sh2{stroke:#a8cce8;fill:#a8cce8}
  .atrack{fill:#f9e2c6}.ah1{fill:#fbe9d2}.ah2{fill:#fbc788}.okf{fill:#388a34}
"""
DARK = """
  .bg{fill:#0d1117}.panel{fill:#181818;stroke:#2b2b2b}.bar{fill:#1f1f1f}
  .sec{fill:none;stroke:#2b2b2b}.card{fill:none;stroke:#3c3c3c}
  .t{fill:#cccccc}.m{fill:#9d9d9d}.h{fill:#cccccc;font-weight:600}.cap{fill:#cccccc;letter-spacing:.5px}
  .link{fill:#4daafc}
  .btn{fill:#313131}.bt{fill:#cccccc}.dt{fill:#f85149}.chev{fill:#c5c5c5}
  .lead{stroke:#d0d7de;stroke-width:1.6;stroke-dasharray:3 3;fill:none}.guide{stroke:#3c3c3c;stroke-width:1;fill:none}
  .badge{fill:#2f81f7}.bn{fill:#0d1117;font-weight:700}
  .lt{fill:#e6edf3;font-weight:600;font-size:12px}.desc{fill:#9da5b4}
  .grey{fill:#8b949e;opacity:.7}
  .spark{fill:none;stroke:#4daafc;stroke-width:1.4}.prog{fill:#0078d4}.track{fill:#2b2b2b}
  .tile{fill:none;stroke:#2b2b2b}.focus{fill:none;stroke:#0078d4;stroke-width:2}
  .cq{fill:#cca700}.cr{fill:#3794ff}.cd{fill:#89d185}.cf{fill:#f14c4c}
  .sq{stroke:#cca700}.sr{stroke:#3794ff}.sd{stroke:#89d185}.sf{stroke:#f14c4c}
  .pill{fill:none;stroke:#0e70c0}.pt{fill:#3794ff}.pn{fill:none;stroke:#9d9d9d}.pf{fill:none;stroke:#f85149}
  .pd{fill:none;stroke:#89d185}.it{fill:#9d9d9d;font-style:italic}
  .grid{stroke:#3c3c3c;fill:none}
  .abar{fill:#181818;stroke:#2b2b2b}.accent{fill:#0078d4}
  .shc{stroke:#8cc8ff;fill:#8cc8ff}.sh1{stroke:#233447;fill:#233447}.sh2{stroke:#294e79;fill:#294e79}
  .atrack{fill:#462f13}.ah1{fill:#33240f}.ah2{fill:#804c0d}.okf{fill:#89d185}
"""
# the status lights and the live-metric dots: panel.css's fixed colours (one for both themes), a thin ring
LAMPS = """
  .green{fill:#22d34a;stroke:#000000;stroke-opacity:.35}.amber{fill:#ff9f0a;stroke:#000000;stroke-opacity:.35}
  .red{fill:#ff3b30;stroke:#000000;stroke-opacity:.35}
  .better{fill:#22d34a}.worse{fill:#ff3b30}.same{fill:#8b949e;opacity:.7}
  .atk{fill:#ff8c00;font-weight:600}.ov{fill:#ff8c00}.afill{fill:#ff8c00}
"""                                  # one selector per rule: some SVG renderers skip grouped selectors
TEXT = LAMPS + '  text{font-family:-apple-system,"Segoe UI",system-ui,Helvetica,Arial,sans-serif;font-size:12px}\n'


def style(theme: str) -> str:
    """The CSS of one theme, or of both for `auto` (dark under prefers-color-scheme: dark, where the
    dark mark replaces the light one)."""
    if theme == "light":
        return LIGHT + TEXT
    if theme == "dark":
        return DARK + TEXT
    dark = "".join(f"  {l}\n" for l in DARK.strip("\n").splitlines())
    return (LIGHT + "  .mk-dark{display:none}\n  @media (prefers-color-scheme: dark){\n" + dark +
            "    .mk-light{display:none}.mk-dark{display:inline}\n  }\n" + TEXT)


def mark(theme: str) -> tuple[list[str], str]:
    """The paths and stroke colour of traceflow-mark-<theme>.svg."""
    text = (BRAND / f"traceflow-mark-{theme}.svg").read_text(encoding="utf-8")
    colour = re.search(r'stroke="(#[0-9A-Fa-f]{3,8})"', text).group(1)
    return re.findall(r'<path d="([^"]+)"', text), colour

LEGEND = [
    ("Share", "a read-only status page for any browser, phone included: one link|and QR code per person; × withdraws it (VPN; services/doc guide 11)"),
    ("GPU strip", "utilisation · memory · temperature of every GPU, with sparklines"),
    ("Docker row", "Docker tear down · the image build and its bar (full once built) · Kill build|(only while building) · container health"),
    ("running here", "the experiment on this machine; (+N) counts its other processes;|under it the panel's build: version and panel.js digest"),
    ("Master Control", "master.yaml as a form: every project's entry, Add a Project;|read only while a run is ongoing"),
    ("Project section", "one per project of master.yaml: light, ENV, status, progress (here ZvW|open, DLSCA_ASCAD closed); the ENV it follows and from where"),
    ("Tiles", "Queued · Running · Done · Failed; click one to list its runs; Done: Retest|and three ranks (validation · test at its traces · test at all); Failed: Delete"),
    ("The run card", "under [Running], in the run's place; it leaves when the run moves to [Done]:|Terminate Run · Log · Attach · Container log · Kill session"),
    ("While it tests", "the orange attack bar, glowing while it runs, in the place of epoch n/N;|the train bar under it, then ETA (run · batch), queue k of n, tmux session"),
    ("Live metrics", "the latest epoch per process: train · validation (CV in a fold); a dot: better ·|worse · unchanged than the epoch before; the key-recovery line under them"),
    ("QUEUES", "Set Params · Set Queue; per experiment_queue.yaml key: state, done of the|runs it makes, commands, its graph jobs; Start · Terminate Run · Retry Start ·|Rerun · Overlay Graph (a whole study done) · Log"),
    ("RUN SINGLE EXPERIMENT", "single_runs.yaml: one make orchestrate per line (ENV, DATASET,|MODEL_ID menus), Start"),
    ("FIGURES", "Graphs: Draw · Draw all · Export to Facet · Open · Folder · Delete;|Faceting: the facets of graph_config.yaml, drawn the same way"),
    ("TESTS", "external tests of every DONE model (test_config.yaml): Test · Log · Registry"),
    ("EXPERIMENT SETTINGS", "every config file of the project as a form, with its role"),
    ("HOUSEKEEPING", "filter (ENV … PARAM), tick, DELETE runs: a dry run first"),
    ("SNAPSHOTS", "reproduce runs from their config snapshots"),
    ("<project>", "Scan with filters · Open · Reproduce (asks where the outputs go)"),
    ("CUSTOM", "Choose folder… · Scan · Open · Reproduce"),
]


class Svg:
    SIZES = {"s": 11, "xs": 10, "l": 15}

    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, cls, rx=4):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>')

    def text(self, x, y, s, cls="t", anchor="start"):
        """One colour class per element (some SVG renderers apply only one); a size token (s, xs, l)
        in `cls` becomes the font-size attribute."""
        words = cls.split()
        size = next((self.SIZES[w] for w in words if w in self.SIZES), None)
        colour = " ".join(w for w in words if w not in self.SIZES) or "t"
        fs = f' font-size="{size}"' if size else ""
        self.add(f'<text x="{x}" y="{y}" class="{colour}"{fs} text-anchor="{anchor}">{escape(s)}</text>')

    def dot(self, x, y, cls, r=4.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" class="{cls}"/>')

    def button(self, x, y, label, danger=False, h=18, dim=False):
        """panel.css button: the theme's secondary button colour, no border; danger: red text."""
        w = 12 + 6.2 * len(label)
        op = ' opacity="0.45"' if dim else ""
        self.add(f'<g{op}><rect x="{x}" y="{y}" width="{w:.0f}" height="{h}" rx="2" class="btn"/>')
        self.text(x + w / 2, y + h - 5, label, f'{"dt" if danger else "bt"} s', "middle")
        self.add('</g>')
        return x + w + 4

    def spark(self, x, y, pts, w=46, h=12):
        step = w / (len(pts) - 1)
        d = " ".join(f"{'M' if i == 0 else 'L'}{x + i * step:.1f},{y + h - p * h:.1f}" for i, p in enumerate(pts))
        self.add(f'<path d="{d}" class="spark"/>')

    def share(self, cx, cy, s=1.25):
        """make_brand.py's share mark (three nodes, two links on a 16-unit box, centred), s px a unit: the halo
        as the mark drawn twice wider in the glow colour mixed into the title bar (.sh1, .sh2: opaque, as some
        SVG renderers ignore opacity), the mark over it."""
        a, b, e = [(cx + dx * s, cy + dy * s) for dx, dy in ((3.5, -4.2), (-3.5, 0), (3.5, 4.2))]
        d = f"M{b[0]:.1f},{b[1]:.1f} L{a[0]:.1f},{a[1]:.1f} M{b[0]:.1f},{b[1]:.1f} L{e[0]:.1f},{e[1]:.1f}"
        for cls, grow in (("sh1", 2.6), ("sh2", 1.3), ("shc", 0)):
            self.add(f'<g class="{cls}" stroke-linecap="round">'
                     f'<path d="{d}" fill="none" stroke-width="{1.7 * s + 2 * grow:.1f}"/>'
                     + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.1 * s + grow:.1f}" stroke="none"/>'
                               for x, y in (a, b, e)) + '</g>')

    def chevron(self, x, y, open_):
        """A drop-down's arrow, drawn (a font may lack ▾ ▸): pointing down when open, right when closed."""
        y -= 4
        pts = f"{x},{y - 3} {x + 8},{y - 3} {x + 4},{y + 3}" if open_ else f"{x + 1},{y - 5} {x + 7},{y} {x + 1},{y + 5}"
        self.add(f'<polygon points="{pts}" class="chev"/>')

    def glyph(self, x, y, state):
        """The tile glyphs of panel.js (○ ◉ ✓ ✕), drawn: shape as well as colour."""
        stroke = {"queued": "sq", "running": "sr", "done": "sd", "failed": "sf"}[state]
        if state == "queued":
            self.add(f'<circle cx="{x}" cy="{y}" r="4" class="{stroke}" fill="none" stroke-width="1.4"/>')
        elif state == "running":
            self.add(f'<circle cx="{x}" cy="{y}" r="4.5" class="{stroke}" fill="none" stroke-width="1.2"/>')
            self.add(f'<circle cx="{x}" cy="{y}" r="2.2" class="cr"/>')
        elif state == "done":
            self.add(f'<path d="M{x - 4},{y} L{x - 1},{y + 3} L{x + 4},{y - 4}" class="{stroke}" fill="none" stroke-width="1.6"/>')
        else:
            self.add(f'<path d="M{x - 3.5},{y - 3.5} L{x + 3.5},{y + 3.5} M{x + 3.5},{y - 3.5} L{x - 3.5},{y + 3.5}" '
                     f'class="{stroke}" fill="none" stroke-width="1.6"/>')

    def summary(self, x, y, w, title, note, open_, box=True):
        """A nested <details> summary: uppercase title (panel.css), its count in small normal case."""
        if box:
            self.rect(x, y, w, 24, "sec")
        self.chevron(x + 8, y + 16, open_)
        self.text(x + 22, y + 16, title.upper(), "cap s")
        if note:
            self.text(x + 30 + 7.4 * len(title), y + 16, note, "m xs")


def activity_bar(theme: str, top: float, bottom: float) -> str:
    """VS Code's Activity Bar at the panel's left with only the TraceFlow mark, selected (the accent bar
    at its left edge): the icon one clicks to open the panel. `auto` draws both marks; the CSS shows
    the one of the viewer's theme."""
    x = PX - AB
    parts = [f'<rect x="{x}" y="{top}" width="{AB}" height="{bottom - top}" class="abar"/>',
             f'<rect x="{x}" y="{top + 6}" width="2" height="{AB - 12}" class="accent"/>']
    for variant in (["light", "dark"] if theme == "auto" else [theme]):
        paths, colour = mark(variant)
        cls = f' class="mk-{variant}"' if theme == "auto" else ""
        body = "".join(f'<path d="{d}"/>' for d in paths)
        parts.append(f'<g{cls} transform="translate({x + 9},{top + 9}) scale(1.25)" fill="none" stroke="{colour}" '
                     f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{body}</g>')
    return "\n".join(parts)


def draw(theme: str = "auto") -> str:
    g = Svg()
    rows: list[float] = []                     # the y of every numbered row, in legend order
    y = EDGE
    top = y
    g.rect(PX, y, PW, 30, "bar", 0)
    g.text(PX + 12, y + 19, "TraceFlow Monitor", "cap s")
    g.share(PX + PW - 24, y + 15)               # 1 the share button, at the title line's right end
    rows.append(y + 15)
    y += 38
    x = PX + 12
    inner = PW - 24                              # content width inside the panel

    # 2 GPU strip
    g.text(x, y + 13, "GPU0", "t s")
    g.text(x + 40, y + 13, "0%", "t s")
    g.spark(x + 70, y + 2, [0.0, 0.9, 0.0, 0.0, 0.95, 0.9, 0.0])
    g.text(x + 126, y + 13, "750/32607 MiB", "t s")
    g.spark(x + 218, y + 2, [0.05, 0.05, 0.6, 0.62, 0.62, 0.6, 0.05])
    g.text(x + 274, y + 13, "47°C", "t s")
    rows.append(y + 9)
    y += 26

    # 3 Docker row: the build bar as panel.js draws it (an SVG track and fill), full and green once built
    nx = g.button(x, y, "Docker tear down", danger=True)
    built = "image built · sidechannel:dev-2564873f1298"
    g.text(nx + 4, y + 13, built, "t s")
    bx = nx + 4 + 5.4 * len(built) + 6
    g.rect(bx, y + 5, 120, 8, "track", 0)
    g.rect(bx, y + 5, 120, 8, "okf", 0)
    g.button(bx + 126, y + 1, "Kill build", danger=True, h=16, dim=True)
    g.dot(nx + 8, y + 31, "green")
    g.text(nx + 17, y + 35, "sidechannel: running", "t s")
    g.text(nx + 132, y + 35, "(Up)", "m xs")
    rows.append(y + 18)
    y += 50

    # 4 running here, the panel's build under it
    g.dot(x + 4, y + 9, "green")
    g.text(x + 14, y + 13, "running here: make experiment ENV=dev PROJECT=ZvW QUEUE=z_fig12_ascad_n100 (+3)", "t xs")
    g.text(x, y + 31, "build 0.1.6 307719e3ea", "m xs")
    rows.append(y + 9)
    y += 42

    # 5 Master Control (master.yaml), closed; read only while a run is ongoing
    g.chevron(x, y + 15, False)
    g.text(x + 16, y + 15, "Master Control", "h")
    g.text(x + 116, y + 15, "master.yaml · read only: a run is ongoing", "m xs")
    rows.append(y + 11)
    y += 26
    g.add(f'<line x1="{x}" y1="{y}" x2="{x + inner}" y2="{y}" class="guide"/>')
    y += 4

    # a closed project: its light, name and status
    g.chevron(x, y + 15, False)
    g.dot(x + 18, y + 11, "grey")
    g.text(x + 28, y + 15, "DLSCA_ASCAD", "h")
    g.text(x + 120, y + 15, "dev · done", "m s")
    y += 26
    g.add(f'<line x1="{x}" y1="{y}" x2="{x + inner}" y2="{y}" class="guide"/>')
    y += 4

    # 6 project section, the ENV it follows under its title
    sec_top = y
    g.chevron(x, y + 15, True)
    g.dot(x + 18, y + 11, "green")
    g.text(x + 28, y + 15, "ZvW", "h")
    g.text(x + 60, y + 15, "dev · running · 98%", "m s")
    rows.append(y + 11)
    y += 26
    ix = x + 14                                  # indented content
    iw = inner - 14
    g.text(ix, y + 12, "ENV dev, followed from the running run bugger_fantastic", "m xs")
    y += 20

    # 7 tiles, [Running] chosen
    tw = (iw - 12) / 4
    tx = ix
    for state, n, word, count_cls in (("queued", 0, "Queued", "cq"), ("running", 1, "Running", "cr"),
                                      ("done", 8, "Done", "cd"), ("failed", 0, "Failed", "cf")):
        g.rect(tx, y, tw, 40, "tile")
        if state == "running":
            g.rect(tx + 1, y + 1, tw - 2, 38, "focus", 3)
        label = str(n)
        g.add('<g opacity="0.55">' if n == 0 else '<g>')        # panel.css .tile.none .count: nothing in this state
        g.glyph(tx + tw / 2 - 4 - 4.5 * len(label), y + 13, state)
        g.text(tx + tw / 2 + 5, y + 19, label, f"{count_cls} l", "middle")
        g.add('</g>')
        g.text(tx + tw / 2, y + 33, word, "t xs", "middle")
        tx += tw + 4
    rows.append(y + 20)
    y += 50

    # 8 the run card, the [Running] list's item in the run's place: name, model · dataset, its buttons
    cx0 = ix                                      # the card's content left
    g.dot(ix + 4, y + 9, "green", r=4)
    g.text(ix + 14, y + 13, "bugger_fantastic", "link s")
    g.text(ix + 112, y + 13, "Zaid_ASCAD_N100 · ascad_n100_none", "m xs")
    acts = ("Terminate Run", "Log", "Detach", "Container log", "Kill session")
    bx = ix + iw - sum(12 + 6.2 * len(a) + 4 for a in acts) + 4
    for label in acts:                            # .run-acts wraps under the name in a panel this wide
        bx = g.button(bx, y + 22, label, danger=label in ("Terminate Run", "Kill session"))
    rows.append(y + 9)

    # 9 while its test attacks: the phase line is the orange bar (glowing while it runs) and attack i of n, where
    # epoch n/N stood, labelled test; the train bar under it, both starting in one column
    py = y + 50
    lab = 34                                      # .bar-label: 2.8em
    count = "attack 33 of 100"
    aw = iw - lab - 8 - 6.4 * len(count)
    ax = cx0 + lab
    g.text(cx0, py + 4, "test", "m xs")
    g.rect(ax - 3, py - 4.5, aw + 6, 12, "ah1", 6)          # the glow: two opaque rings (renderers may drop opacity)
    g.rect(ax - 1.5, py - 3, aw + 3, 9, "ah2", 4.5)
    g.rect(ax, py - 1.5, aw, 6, "atrack", 3)
    g.rect(ax, py - 1.5, aw * 0.33, 6, "afill", 3)
    g.text(ax + aw + 8, py + 4, count, "atk xs")
    rows.append(py + 1)
    ty = py + 18
    g.text(cx0, ty + 4, "train", "m xs")
    tbw = iw - lab - 44
    g.rect(ax, ty - 1.5, tbw, 6, "track", 3)
    g.rect(ax, ty - 1.5, tbw * 0.98, 6, "prog", 3)
    g.text(ax + tbw + 8, ty + 4, "98%", "t xs")
    # the ETA table: ETA | run: … | batch: …
    ety, eth = ty + 12, 20
    c1, c2 = cx0 + 36, cx0 + 36 + (iw - 36) / 2
    g.rect(cx0, ety, iw, eth, "grid", 3)
    g.add(f'<line x1="{c1}" y1="{ety}" x2="{c1}" y2="{ety + eth}" class="grid"/>')
    g.add(f'<line x1="{c2}" y1="{ety}" x2="{c2}" y2="{ety + eth}" class="grid"/>')
    g.text(cx0 + 6, ety + 14, "ETA", "h xs")
    g.text(c1 + 6, ety + 14, "run: 23:37 Taipei (in 1 min)", "t xs")
    g.text(c2 + 6, ety + 14, "batch: 23:37 Taipei (in 1 min)", "t xs")
    g.text(cx0, ety + 38, "queue z_fig12_ascad_n100: 1 of 1 done · trial 1/1", "t s")
    g.text(cx0, ety + 54, "tmux session: tf-ZvW", "t s")
    y = ety + 62

    # 10 its live metrics (core/monitor.snapshot), as panel.js liveHtml draws them: a head row (the epoch over the
    # metric names, each process over its values) and a rule under it, then one row per metric with its values
    # under their process, each with its dot; the key-recovery line, the dots' key
    lx0 = cx0
    lines = (("trainable_parameters", {"train": ("141596", "same")}),
             ("η_t", {"train": ("0.00001000", "worse")}),
             ("mean_accuracy", {"train": ("0.01887", "worse"), "validation": ("0.006000", "worse")}),
             ("CrossEntropyLoss", {"train": ("5.206", "better"), "validation": ("5.362", "better")}),
             ("mean_attack_rank_curve", {"validation": ("1", "same")}),
             ("per_attack_curves", {"validation": ("1", "same")}),
             ("final_ranks", {"validation": ("1", "same")}),
             ("traces_to_ge_1", {"validation": ("548", "same")}),
             ("per_attack_traces_to_ge_1", {"validation": ("167.79", "better")}))
    NAME_PX = 5.3
    head_y = y + 11                                # the head row: "epoch 50" | train | validation
    g.text(lx0, head_y, "epoch 50", "h xs")
    g.add(f'<line x1="{lx0}" y1="{head_y + 4}" x2="{lx0 + iw}" y2="{head_y + 4}" class="guide"/>')   # the rule under it
    top_row = y + 28
    cx = lx0 + max(NAME_PX * len(n) for n, _ in lines) + 10
    width = (lx0 + iw - cx) / 2                   # .live-rows width: 100%, the two process columns share the rest
    for k, process in enumerate(("train", "validation")):
        # the | before each process column, from the head row down to the last metric
        g.add(f'<line x1="{cx:.1f}" y1="{head_y - 9}" x2="{cx:.1f}" y2="{top_row + 13 * len(lines) - 8}" class="guide"/>')
        g.text(cx + width - 4, head_y, process, "h xs", "end")
        for j, (_, values) in enumerate(lines):
            if process in values:
                value, state = values[process]
                my = top_row + 13 * j
                g.text(cx + width - 14, my, value, "t xs", "end")
                g.dot(cx + width - 6, my - 3.5, state, r=3.5)
        cx += width
    for j, (name, _) in enumerate(lines):
        g.text(lx0, top_row + 13 * j, name, "t xs")
    rows.append(top_row + 40)
    ky = top_row + 13 * len(lines) + 6
    g.text(lx0, ky, "key recovery · validation, epoch 50: recovered · true key rank 1 · top guess 0xE0 · GE 1 · SR@1 1.00", "t xs")
    ky += 18
    kx = lx0
    for state, words in (("better", "better than the epoch before"), ("worse", "worse than the epoch before"),
                         ("same", "unchanged, or the first epoch")):
        g.dot(kx + 3, ky - 3, state, r=3)
        g.text(kx + 12, ky, words, "m xs")
        kx += 36 + 5.0 * len(words)                 # panel.css .live-legend gap
    y = ky + 18

    # 11 QUEUES, opened: Set Params · Set Queue and the reminder; the action follows the state
    queues = (("z_fig12_ascad_n100", "running", "1/1", "1 cmd", "graph #9", ["Terminate Run"]),
              ("z_fig8_ascad_n0", "done", "1/1", "1 cmd", "graph #5, #6", ["Overlay Graph", "Rerun"]),
              ("z_fig7_aes_hd", "done", "1/1", "1 cmd", "graph #3", ["Rerun"]),
              ("w_fig4_ascad", "not started", "0/10", "1 cmd", "graph #12", ["Start"]))
    KX = 76                                        # the key column's extra width: ZvW's keys are long
    width = lambda labels: sum(12 + 6.2 * len(l) + 4 for l in labels)
    # panel.css .acts wraps under the row when the buttons would reach the graph ids
    heights = [22 + (20 if ix + iw - 10 - width(acts + ["Log"]) < ix + KX + 232 + 5.4 * len(graphs) + 8 else 0)
               for *_, graphs, acts in queues]
    qh = 50 + sum(heights) + 4
    g.rect(ix, y, iw, qh, "sec")
    g.summary(ix, y, iw, "Queues", "8/26 done", True, box=False)
    nx = g.button(ix + 10, y + 26, "Set Params")
    g.button(nx, y + 26, "Set Queue")
    g.text(ix + iw - 10, y + 39, "Configure parameters correctly before starting the Run", "it xs", "end")
    pill = {"running": ("pill", "pt"), "failed": ("pf", "dt"), "done": ("pd", "cd"), "not started": ("pn", "m")}
    ry = y + 50
    for (key, state, cnt, cmds, graphs, acts), h in zip(queues, heights):
        g.text(ix + 12, ry + 13, key, "t s")
        w = 16 + 5.8 * len(state)                  # the word with 8 px each side (panel.css .state: padding 0 4px, 10 px text)
        box, ink = pill[state]
        g.rect(ix + KX + 70, ry + 1, w, 18, box, 3)            # tall enough for f, l, d: the word sits in the middle
        g.text(ix + KX + 70 + w / 2, ry + 13.5, state, f"{ink} xs", "middle")
        g.text(ix + KX + 160, ry + 13, cnt, "t xs")
        g.text(ix + KX + 192, ry + 13, cmds, "m xs")
        g.text(ix + KX + 232, ry + 13, graphs, "m xs")
        labels = acts + ["Log"]
        bxx = ix + iw - 10 - width(labels)
        by = ry if h == 22 else ry + 20                         # panel.css .acts wraps under the row when narrow
        for label in labels:
            bxx = g.button(bxx, by, label, danger=label == "Terminate Run")
        ry += h
    rows.append(y + 12)
    y += qh + 6

    # 12-16 closed drop-downs
    for title, note in (("Run Single Experiment", "1 run"), ("Figures", "2/14 drawn · 0/1 facets"),
                        ("Tests", "8 models · 0 external"), ("Experiment Settings", "10 files"), ("Housekeeping", "")):
        g.summary(ix, y, iw, title, note, False)
        rows.append(y + 12)
        y += 30
    sec_bottom = y
    g.add(f'<line x1="{x + 4}" y1="{sec_top + 26}" x2="{x + 4}" y2="{sec_bottom - 6}" class="guide"/>')

    # 17-19 SNAPSHOTS
    y += 4
    g.chevron(x, y + 15, True)
    g.text(x + 16, y + 15, "SNAPSHOTS", "h")
    g.text(x + 100, y + 15, "reproduce runs from their config snapshots", "m xs")
    rows.append(y + 11)
    y += 26
    for title, note in (("ZvW", "8 of 8 scanned"), ("CUSTOM", "")):
        g.summary(ix, y, iw, title, note, False)
        rows.append(y + 12)
        y += 30
    bottom = y + 6

    frame = (f'<rect x="{PX}" y="{top}" width="{PW}" height="{bottom - top}" rx="6" class="panel"/>\n'
             + activity_bar(theme, top, bottom))

    # badges and leaders, then the legend: each entry as tall as its lines
    out = Svg()
    ly = top + 6
    for i, ((title, desc), ry) in enumerate(zip(LEGEND, rows), 1):
        out.add(f'<line x1="{PX + PW - 6}" y1="{ry}" x2="{BADGE_X - 10}" y2="{ry}" class="lead"/>')
        out.add(f'<circle cx="{BADGE_X}" cy="{ry}" r="9" class="badge"/>')
        out.text(BADGE_X, ry + 4, str(i), "bn xs", "middle")
        out.add(f'<circle cx="{LX + 9}" cy="{ly + 8}" r="9" class="badge"/>')
        out.text(LX + 9, ly + 12, str(i), "bn xs", "middle")
        out.text(LX + 26, ly + 12, title, "lt")
        lines = desc.split("|")
        for k, line in enumerate(lines):
            out.text(LX + 26, ly + 27 + 14 * k, line, "desc s")
        ly += 26 + 14 * len(lines) + 2

    # the key of the marks, under the legend
    ky = ly + 14
    out.text(LX, ky, "Marks", "lt")
    out.chevron(LX + 2, ky + 20, True)
    out.chevron(LX + 18, ky + 20, False)
    out.text(LX + 96, ky + 20, "a drop-down, open / closed (click its title)", "desc s")
    out.button(LX, ky + 27, "Delete", danger=True, h=16)
    out.text(LX + 96, ky + 39, "red label: stops or deletes something, asks first", "desc s")
    lights = [("green", "running"), ("amber", "late beat"), ("red", "no beat"), ("grey", "idle, ended")]
    lx = LX
    for cls, what in lights:
        out.dot(lx + 4, ky + 56, cls)
        out.text(lx + 12, ky + 60, what, "desc s")
        lx += 24 + 6.0 * len(what)
    bottom = max(bottom, ky + 70)

    height = bottom + EDGE
    # the canvas ends at the legend's longest line (about 5.5 px a character of its 11 px text in a browser)
    W = round(max([LX + 26 + 7.0 * len(t) for t, _ in LEGEND] +
                  [LX + 26 + 5.5 * len(line) for _, d in LEGEND for line in d.split("|")] +
                  [LX + 96 + 5.5 * len("red label: stops or deletes something, asks first")])) + EDGE
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
            f'role="img" aria-label="The TraceFlow Monitor panel: its drop-down sections, numbered, with their legend">\n'
            f'<title>The TraceFlow Monitor panel</title>\n<style>{style(theme)}</style>\n'
            f'<rect width="{W}" height="{height}" class="bg"/>\n{frame}\n'
            + "\n".join(g.parts) + "\n" + "\n".join(out.parts) + "\n</svg>\n")


if __name__ == "__main__":
    for theme, out in OUTS.items():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(draw(theme), encoding="utf-8")
        print(out)
