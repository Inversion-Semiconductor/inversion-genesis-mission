"""Phoebus display widgets as ElementTree elements, for the generators in this directory
(`generate_lpa_display.py`, `generate_linked_synoptic.py`)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

_FONT_FAMILY = "Liberation Sans"


def button(
    parent: ET.Element,
    name: str,
    text: str,
    file: str,
    description: str,
    x: int,
    y: int,
    w: int,
    h: int = 26,
) -> ET.Element:
    """A button that opens the display `file` in a window."""
    widget = ET.SubElement(parent, "widget", type="action_button", version="3.0.0")
    _text(widget, "name", name)
    open_display(ET.SubElement(widget, "actions"), file, description)
    _text(widget, "text", text)
    _geometry(widget, x, y, w, h)
    return widget


def combo(
    parent: ET.Element, name: str, pv: str, x: int, y: int, w: int, h: int = 24
) -> ET.Element:
    """A combo box that writes the enum PV `pv` and takes its items from it."""
    widget = _widget(parent, "combo", name, x, y, w, h)
    _text(widget, "pv_name", pv)
    _text(widget, "items_from_pv", "true")
    return widget


def display(name: str, width: int, height: int) -> ET.Element:
    """The root of a display."""
    root = ET.Element("display", version="2.0.0")
    _text(root, "name", name)
    _text(root, "width", str(width))
    _text(root, "height", str(height))
    return root


def label(
    parent: ET.Element,
    name: str,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int = 22,
    *,
    align: int | None = None,
    bold: bool = False,
    size: float | None = None,
    wrap: bool = False,
) -> ET.Element:
    """A label: bold or of another font `size` (12 otherwise), and aligned (0 left, 1 centre,
    2 right) and word-wrapped as asked."""
    widget = _widget(parent, "label", name, x, y, w, h)
    _text(widget, "text", text)
    if bold or size:
        ET.SubElement(
            ET.SubElement(widget, "font"),
            "font",
            family=_FONT_FAMILY,
            style="BOLD" if bold else "REGULAR",
            size=str(size or 12.0),
        )
    if align is not None:
        _text(widget, "horizontal_alignment", str(align))
    if wrap:
        _text(widget, "wrap_words", "true")
    return widget


def led(
    parent: ET.Element,
    name: str,
    pv: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    off_color: tuple[int, int, int],
    on_color: tuple[int, int, int],
) -> ET.Element:
    """A LED showing the PV `pv`, in the colours `(red, green, blue)` given."""
    widget = _widget(parent, "led", name, x, y, w, h)
    _text(widget, "pv_name", pv)
    for tag, rgb in (("off_color", off_color), ("on_color", on_color)):
        ET.SubElement(
            ET.SubElement(widget, tag),
            "color",
            **{channel: str(v) for channel, v in zip(("red", "green", "blue"), rgb)},
        )
    return widget


def open_display(actions: ET.Element, file: str, description: str) -> ET.Element:
    """An action, in the `actions` of a button, that opens the display `file` in a window."""
    action = ET.SubElement(actions, "action", type="open_display")
    _text(action, "file", file)
    ET.SubElement(action, "macros")
    _text(action, "target", "window")
    _text(action, "description", description)
    return action


def rule(parent: ET.Element, name: str, x: int, y: int, w: int) -> ET.Element:
    """A grey horizontal line."""
    widget = _widget(parent, "rectangle", name, x, y, w, 1)
    _text(widget, "line_width", "0")
    ET.SubElement(
        ET.SubElement(widget, "background_color"),
        "color",
        red="150",
        green="150",
        blue="150",
        alpha="255",
    )
    return widget


def to_xml(root: ET.Element) -> str:
    """The text of the display `root`, indented, with its XML declaration."""
    ET.indent(root, space="  ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def value(
    parent: ET.Element,
    name: str,
    pv: str,
    x: int,
    y: int,
    w: int,
    h: int = 22,
    *,
    exponential: bool = False,
    precision: int | None = None,
) -> ET.Element:
    """A text update of the PV `pv`, without its units, with `precision` decimals (the PV's
    otherwise), in exponential notation if asked."""
    widget = _widget(parent, "textupdate", name, x, y, w, h)
    _text(widget, "pv_name", pv)
    if exponential:
        _text(widget, "format", "2")
    if precision is not None:
        _text(widget, "precision", str(precision))
        _text(widget, "precision_from_pv", "false")
    _text(widget, "show_units", "false")
    return widget


def _geometry(widget: ET.Element, x: int, y: int, w: int, h: int) -> None:
    for tag, number in (("x", x), ("y", y), ("width", w), ("height", h)):
        _text(widget, tag, str(number))


def _text(parent: ET.Element, tag: str, text: str) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = text
    return child


def _widget(
    parent: ET.Element, kind: str, name: str, x: int, y: int, w: int, h: int
) -> ET.Element:
    widget = ET.SubElement(parent, "widget", type=kind, version="2.0.0")
    _text(widget, "name", name)
    _geometry(widget, x, y, w, h)
    return widget
