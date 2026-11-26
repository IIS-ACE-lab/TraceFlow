#!/usr/bin/env python3
"""Colour themes shared by TraceFlow's figures: the paper's system figures and the planner's model diagrams.

Location : services/visualization/utils/fig_theme.py
Used by  : paper/draft_4/figures/system/ (fig_training_protocol.py, make_system_design.py,
           make_model_pipeline.py, make_system_figures.py, make_evaluation_workflow.py, P2_report.py);
           services/visualization/vscode_monitor/model_diagram.py (the planner's Download, its theme menu)

A theme is a page, a panel, two text colours and eleven palette slots. A
figure script does not name colours: it maps each of its own roles to a slot,

    ROLES = {"profiling": "key", "attack": "secondary", ...}
    T = fig_theme.resolve("samurai", ROLES)      # T["profiling"] -> "#0B0B0B"

so every theme below is available to every figure, and a new theme added here
reaches all of them at once.

Flags every figure script takes (added by add_arguments):
    --theme <name>     one theme
    --all-themes       one output per theme, named <stem>_<theme>

Run this file to list the themes:
    python services/visualization/utils/fig_theme.py

Author: Shraddha Mishra
"""
from __future__ import annotations

import sys

from matplotlib.colors import to_rgb

DEFAULT_THEME = "samurai"

# What every theme defines.
#   page        figure background
#   panel       background of a panel, a strip or an inset
#   ink         text and neutral arrows
#   muted       secondary text: paths, notes
# and the palette slots, by the part they usually play in a figure:
#   key         the element everything starts from (a source, an entry point)
#   neutral     a quiet element next to the key one
#   dark        stored or passive things (files, a flatten step)
#   primary     the main actor of the figure
#   secondary   the second actor
#   tertiary    configuration, preparation
#   deep        the end of the chain (a result, a verdict)
#   support     a helper beside the main actor
#   accent      one element that must stand apart
#   highlight   a second accent
#   fresh       a third accent
BASE = ("page", "panel", "ink", "muted")
SLOTS = ("key", "neutral", "dark", "primary", "secondary", "tertiary", "deep",
         "support", "accent", "highlight", "fresh")

THEMES = {
    "classic": dict(        # the original palette of the model figures: muted blue, teal, gold, red
        page="#FFFFFF", panel="#F7F7F9", ink="#222222", muted="#666666",
        key="#2F4B7C", neutral="#6C757D", dark="#8C8C8C", primary="#4C78A8",
        secondary="#B279A2", tertiary="#E8B92C", deep="#E45756", support="#3BA7A0",
        accent="#54A24B", highlight="#F58518", fresh="#8BB84A"),
    "samurai": dict(        # white, black, red, gold, orange, lacquer red
        page="#FFFFFF", panel="#F5F4F1", ink="#0B0B0B", muted="#5E5A55",
        key="#0B0B0B", neutral="#4A4A4A", dark="#2B2B2B", primary="#D3121F",
        secondary="#F07F13", tertiary="#C9A227", deep="#7A0E18", support="#9C7A1C",
        accent="#B8860B", highlight="#F07F13", fresh="#A63A00"),
    "sakura": dict(         # cherry blossom: pinks, plum, bark brown, young leaf
        page="#FFFFFF", panel="#FBF3F6", ink="#3A2A30", muted="#7B6A70",
        key="#5B3A46", neutral="#8C5A6B", dark="#7A5C4F", primary="#E0527F",
        secondary="#B565A7", tertiary="#D9A441", deep="#C2185B", support="#6FA98F",
        accent="#6B8E5A", highlight="#EC7FA9", fresh="#8D6E63"),
    "mafia": dict(          # black suit, blood red, old gold, cigar brown, olive on cream
        page="#FBF8F1", panel="#F1ECE0", ink="#111111", muted="#5A534A",
        key="#111111", neutral="#3B3B3B", dark="#2F2F2F", primary="#8B0000",
        secondary="#5C4033", tertiary="#B8963E", deep="#B22222", support="#556B2F",
        accent="#4B5320", highlight="#8C7853", fresh="#A67C52"),
    "german": dict(         # black, red, gold
        page="#FFFFFF", panel="#F4F4F4", ink="#000000", muted="#555555",
        key="#000000", neutral="#4D4D4D", dark="#333333", primary="#DD0000",
        secondary="#B38F00", tertiary="#FFCE00", deep="#990000", support="#7A7A7A",
        accent="#1A1A1A", highlight="#E0A800", fresh="#8C6D00"),
    "macarons": dict(       # pastel pistachio, apricot, blueberry, rose, lemon
        page="#FFFBF6", panel="#FAF0EA", ink="#4A3F45", muted="#8A7D83",
        key="#7CC68D", neutral="#B4A7D6", dark="#B0A199", primary="#F07AA5",
        secondary="#F7A072", tertiary="#7FA6EE", deep="#E3BF3D", support="#8E7CC3",
        accent="#6CCBC0", highlight="#F2A9C4", fresh="#BFD97A"),
    "anime": dict(          # saturated cel-shading colours
        page="#FFFFFF", panel="#EEF3FF", ink="#1B1B3A", muted="#5C5C7A",
        key="#35C82E", neutral="#6E7191", dark="#5C6BC0", primary="#FF3D8E",
        secondary="#FF8413", tertiary="#1FA9FF", deep="#8E5BFF", support="#00C9A7",
        accent="#FFC400", highlight="#FF5E5E", fresh="#9BE15D"),
    "serious": dict(        # navy and slate greys, for a formal report
        page="#FFFFFF", panel="#F3F5F8", ink="#111827", muted="#4B5563",
        key="#1F2A44", neutral="#4A5568", dark="#4B5563", primary="#1F3A5F",
        secondary="#3D5A80", tertiary="#6B7280", deep="#2F4858", support="#5C7A99",
        accent="#33475B", highlight="#52616B", fresh="#7A8CA3"),
    "attack_defend": dict(  # blue = the side that builds and trains, red = the side that attacks and judges
        page="#FFFFFF", panel="#F5F6FA", ink="#111827", muted="#4B5563",
        key="#2B2D42", neutral="#4A4E69", dark="#6B7280", primary="#1D4ED8",
        secondary="#2563EB", tertiary="#8D99AE", deep="#DC2626", support="#0EA5E9",
        accent="#B91C1C", highlight="#EA580C", fresh="#7F1D1D"),
    "neon": dict(           # neon colours on a white page (paper)
        page="#FFFFFF", panel="#F3F4F8", ink="#0A0A12", muted="#5A5F73",
        key="#22E000", neutral="#8A8FA3", dark="#3A3D4D", primary="#FF2BC2",
        secondary="#FF7A00", tertiary="#00C8F0", deep="#F2C200", support="#7A5CFF",
        accent="#00D98B", highlight="#FF4D4D", fresh="#B6F000"),
    "neon_dark": dict(      # neon colours on black (slides)
        page="#05050A", panel="#10101A", ink="#F4F4FA", muted="#A0A6BA",
        key="#39FF14", neutral="#9AA0B5", dark="#C8CCDA", primary="#FF3DCB",
        secondary="#FF9A1F", tertiary="#00E5FF", deep="#FFF200", support="#9D7BFF",
        accent="#1FFFA8", highlight="#FF5C5C", fresh="#C6FF3D"),
    "matrix": dict(         # terminal green on black, amber and red accents
        page="#010A03", panel="#04160A", ink="#C9FFD6", muted="#7FCF98",
        key="#00FF41", neutral="#6FBF8A", dark="#9BE8B0", primary="#FF4545",
        secondary="#FFB000", tertiary="#19E6C3", deep="#D4FF3A", support="#2BD4FF",
        accent="#7CFF6B", highlight="#FF8A3D", fresh="#B8FF8A"),
    "traceflow": dict(      # the TraceFlow planner's light colours: VS Code Light Modern's chart colours on white
        page="#FFFFFF", panel="#F6F8FA", ink="#1F2328", muted="#59636E",
        key="#1F2328", neutral="#6E7781", dark="#BC05BC", primary="#1A85FF",
        secondary="#D18616", tertiary="#0598BC", deep="#BF8803", support="#652D90",
        accent="#388A34", highlight="#E51400", fresh="#3794FF"),
    "cyberpunk": dict(      # TraceFlow at night: the panel's dark ground, its neon status lights (green, amber, red) and
                            # the share button's glowing blue, with magenta and cyan
        page="#0D1117", panel="#161B22", ink="#E6EDF3", muted="#8B949E",
        key="#E6EDF3", neutral="#8B949E", dark="#FF2BD6", primary="#00E5FF",
        secondary="#FF9F0A", tertiary="#3794FF", deep="#FFE600", support="#B180FF",
        accent="#22D34A", highlight="#FF3B30", fresh="#8CC8FF"),
    "okabe_ito": dict(      # Okabe & Ito's colour-blind safe palette (Color Universal Design)
        page="#FFFFFF", panel="#F5F5F5", ink="#000000", muted="#555555",
        key="#000000", neutral="#BBBBBB", dark="#7F7F7F", primary="#0072B2",
        secondary="#E69F00", tertiary="#56B4E9", deep="#F0E442", support="#CC79A7",
        accent="#009E73", highlight="#D55E00", fresh="#56B4E9"),
    "grayscale": dict(      # black and greys, for a black-and-white print
        page="#FFFFFF", panel="#F2F2F2", ink="#000000", muted="#555555",
        key="#000000", neutral="#B0B0B0", dark="#404040", primary="#1A1A1A",
        secondary="#6E6E6E", tertiary="#8C8C8C", deep="#4D4D4D", support="#2E2E2E",
        accent="#A0A0A0", highlight="#5A5A5A", fresh="#C8C8C8"),
    "solarized": dict(      # Solarized light: base3 page, base01 text, its eight accents
        page="#FDF6E3", panel="#EEE8D5", ink="#073642", muted="#586E75",
        key="#073642", neutral="#93A1A1", dark="#657B83", primary="#268BD2",
        secondary="#CB4B16", tertiary="#2AA198", deep="#B58900", support="#6C71C4",
        accent="#859900", highlight="#DC322F", fresh="#D33682"),
    "solarized_dark": dict(  # Solarized dark: base03 page, base1 text, the same accents (slides)
        page="#002B36", panel="#073642", ink="#EEE8D5", muted="#93A1A1",
        key="#EEE8D5", neutral="#839496", dark="#D33682", primary="#268BD2",
        secondary="#CB4B16", tertiary="#2AA198", deep="#B58900", support="#6C71C4",
        accent="#859900", highlight="#DC322F", fresh="#93A1A1"),
    "dracula": dict(        # Dracula: purple-grey night, pink, cyan, green, orange, yellow (slides)
        page="#282A36", panel="#343746", ink="#F8F8F2", muted="#A3A7C2",
        key="#F8F8F2", neutral="#6272A4", dark="#FF79C6", primary="#8BE9FD",
        secondary="#FFB86C", tertiary="#BD93F9", deep="#F1FA8C", support="#BD93F9",
        accent="#50FA7B", highlight="#FF5555", fresh="#8BE9FD"),
    "monokai": dict(        # Monokai: the editor's charcoal, magenta, orange, yellow, green, blue, purple (slides)
        page="#272822", panel="#33342C", ink="#F8F8F2", muted="#A59F85",
        key="#F8F8F2", neutral="#75715E", dark="#F92672", primary="#66D9EF",
        secondary="#FD971F", tertiary="#AE81FF", deep="#E6DB74", support="#AE81FF",
        accent="#A6E22E", highlight="#F92672", fresh="#66D9EF"),
    "gruvbox": dict(        # Gruvbox light: warm cream, earthy red, orange, yellow, green, aqua, blue, purple
        page="#FBF1C7", panel="#F2E5BC", ink="#3C3836", muted="#7C6F64",
        key="#3C3836", neutral="#928374", dark="#8F3F71", primary="#076678",
        secondary="#AF3A03", tertiary="#427B58", deep="#B57614", support="#8F3F71",
        accent="#79740E", highlight="#9D0006", fresh="#458588"),
    "tokyo_night": dict(    # Tokyo Night: deep indigo night, neon blue, magenta, teal, amber (slides)
        page="#1A1B26", panel="#24283B", ink="#C0CAF5", muted="#7A82AA",
        key="#C0CAF5", neutral="#565F89", dark="#BB9AF7", primary="#7AA2F7",
        secondary="#FF9E64", tertiary="#2AC3DE", deep="#E0AF68", support="#BB9AF7",
        accent="#9ECE6A", highlight="#F7768E", fresh="#73DACA"),
    "viridis": dict(        # matplotlib's viridis: perceptually even, readable in grey and by colour-blind readers
        page="#FFFFFF", panel="#F4F6F5", ink="#1B1B1B", muted="#555555",
        key="#440154", neutral="#8C8C8C", dark="#482878", primary="#3E4A89",
        secondary="#31688E", tertiary="#26828E", deep="#FDE725", support="#1F9E89",
        accent="#35B779", highlight="#6DCD59", fresh="#B4DE2C"),
    "results": dict(        # the P2 results figure (P2_report.py): navy heat map, blue agrees, forest-green repository
                            # agrees (orange until 2026-10-28: it met the report's fire-shade rank headers), teal
                            # recovered (green until 2026-10-28: it met the repository's forest green),
                            # maroon / hot pink contradicts, purple test GE, goldenrod N_val and partly
        page="#FFFFFF", panel="#F4F4F4", ink="#222222", muted="#666666",
        key="#08306B", neutral="#A0A0A0", dark="#4A148C", primary="#00897B",
        secondary="#228B22", tertiary="#1565C0", deep="#800000", support="#7E57C2",
        accent="#FF2D95", highlight="#E53935", fresh="#B8860B"),
    "nord": dict(           # Nord: polar night on snow storm, frost blues and the aurora's colours
        page="#ECEFF4", panel="#E5E9F0", ink="#2E3440", muted="#4C566A",
        key="#2E3440", neutral="#81A1C1", dark="#4C566A", primary="#5E81AC",
        secondary="#D08770", tertiary="#88C0D0", deep="#EBCB8B", support="#B48EAD",
        accent="#A3BE8C", highlight="#BF616A", fresh="#8FBCBB"),
}


def names() -> list:
    """Every theme name, in the order of THEMES."""
    return list(THEMES)


def get(name: str) -> dict:
    """The theme as a dict of its base colours and palette slots."""
    if name not in THEMES:
        sys.exit(f"unknown theme '{name}'. Available: {', '.join(THEMES)}")
    return dict(THEMES[name])


def resolve(name: str, roles: dict) -> dict:
    """The colours of one figure: the base colours plus one colour per role.

    roles = {role of the figure: palette slot}, e.g. {"profiling": "key"}.
    """
    theme = get(name)
    unknown = sorted({slot for slot in roles.values() if slot not in SLOTS})
    if unknown:
        raise KeyError(f"not a palette slot: {', '.join(unknown)}. Slots: {', '.join(SLOTS)}")
    colours = {key: theme[key] for key in BASE}
    colours.update({role: theme[slot] for role, slot in roles.items()})
    return colours


def add_arguments(parser, default: str = DEFAULT_THEME) -> None:
    """The theme flags of a figure script."""
    parser.add_argument("--theme", choices=names(), default=default,
                        help=f"colour theme (default: {default})")
    parser.add_argument("--all-themes", action="store_true",
                        help="one output per theme, named <stem>_<theme>")


def selected(args) -> list:
    """The themes to draw for these parsed arguments."""
    return names() if args.all_themes else [args.theme]


def tint(colour, alpha: float, page):
    """Opaque light tint: `colour` at `alpha` over the page colour."""
    c, p = to_rgb(colour), to_rgb(page)
    return tuple(p[i] + (c[i] - p[i]) * alpha for i in range(3))


def on(colour, page, ink, limit: float = 0.72):
    """Text colour that reads on `colour`: the page colour, or the ink on a light colour."""
    r, g, b = to_rgb(colour)
    light = 0.299 * r + 0.587 * g + 0.114 * b
    dark_page = sum(to_rgb(page)) < 1.5
    if dark_page:                                    # bright bands on a dark page take the dark page colour
        return page if light > 0.35 else ink
    return page if light < limit else ink


for _name, _theme in THEMES.items():                 # every theme defines every key
    _missing = [k for k in BASE + SLOTS if k not in _theme]
    if _missing:
        raise KeyError(f"theme '{_name}' lacks: {', '.join(_missing)}")


if __name__ == "__main__":
    print("themes:", ", ".join(names()))
    print("slots :", ", ".join(SLOTS))
