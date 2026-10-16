"""Tema tampilan Studio: terang (bawaan) atau gelap.

Tema dibaca SEKALI saat Studio dimulai, dari ``STUDIO_TEMA`` (``gelap``/``terang``)
atau dari ``~/.config/studio_rgbd/tampilan.json``. Untuk tema gelap,
:func:`pasang` mengaitkan ``tkinter.Misc._options`` -- jalur yang dilalui SEMUA
pembuatan widget, ``configure`` dan ``itemconfigure`` -- sehingga setiap warna
terang yang tertulis di kode diterjemahkan ke padanan gelapnya menurut perannya
(latar atau tinta). Warna gambar kanvas (opsi ``fill``/``outline``: mask, poligon,
garis) sengaja tidak diterjemahkan. Tombol bulat (TombolRounded) menggambar
dengan :func:`latar`/:func:`tinta` sendiri, dan gaya ttk diatur di ``_gaya``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BERKAS = Path.home() / ".config" / "studio_rgbd" / "tampilan.json"


def baca_tema() -> str:
    env = os.environ.get("STUDIO_TEMA", "").strip().lower()
    if env in ("gelap", "terang"):
        return env
    try:
        return "gelap" if json.loads(BERKAS.read_text(encoding="utf-8")).get("tema") == "gelap" else "terang"
    except (OSError, ValueError, AttributeError):
        return "terang"


def simpan_tema(tema: str) -> None:
    BERKAS.parent.mkdir(parents=True, exist_ok=True)
    BERKAS.write_text(json.dumps({"tema": tema}), encoding="utf-8")


TEMA = baca_tema()
GELAP = TEMA == "gelap"

# Warna terang -> gelap untuk LATAR (background, highlightbackground, troughcolor ...).
PETA_LATAR = {
    "#F3EEE7": "#181513",   # BG jendela
    "#FFFDFC": "#24201E",   # PANEL / kartu
    "#FFFFFF": "#24201E", "WHITE": "#24201E",
    "#FFF9F4": "#1D1A18",   # daftar
    "#FAF7F5": "#201C1A", "#F7F3F0": "#221E1C",
    "#E8DDD5": "#37302C",   # tombol sekunder
    "#E9D8CC": "#4A3A31",   # ACCENT_SOFT: pilihan daftar, lencana
    "#EEE7E2": "#2C2724",   # tab nonaktif
    "#E6DED8": "#3A332F", "#F0EAE6": "#332C29", "#DED7D2": "#463E39", "#DED2C8": "#3E3632",
    "#E4DAD3": "#3A322E",
    "#D8D0CB": "#433C38",   # tombol nonaktif
    "#D8CAC1": "#4D433D",   # garis tombol sekunder
    "#F3D8D4": "#5A2D29",   # tombol bahaya
    "#754C3B": "#A9694E",   # ACCENT sebagai isi tombol
    "#7FA96B": "#5E8A4C", "#6B8B63": "#557A4D", "#407EA3": "#3A6F91",
    "#FBEEDA": "#4A3A1E", "#DCEFD8": "#1F3A24", "#FDE7E5": "#4A2422",
    "#EFE7FD": "#30254A", "#EEF1F5": "#2A2F36", "#FFD9D9": "#4A2626",
}
# Warna terang -> gelap untuk TINTA (foreground, insertbackground ...).
PETA_TINTA = {
    "#382D29": "#ECE5DF",   # INK
    "#786962": "#ABA097",   # MUTED
    "#7A6E66": "#A69B93",
    "#754C3B": "#E2A889",   # ACCENT sebagai teks
    "#8A5A12": "#E8C27A", "#1E5B2A": "#8FD19E", "#B42318": "#FF9C94",
    "#5B21B6": "#C4B5FD", "#475569": "#B8C2CF",
    "#1E8E3E": "#6FCF86", "#D93025": "#FF7B72", "#7C3AED": "#B69CFF", "#64748B": "#9AA6B6", "#B0A8A0": "#7D756F",
    "#6B8B63": "#8FBF84", "#407EA3": "#7FB6D9", "#AA5A55": "#E08A84",
    "#000000": "#ECE5DF", "BLACK": "#ECE5DF",
}
_LATAR = {"background", "bg", "activebackground", "highlightbackground", "selectbackground", "troughcolor",
          "selectcolor", "readonlybackground", "disabledbackground"}
_TINTA = {"foreground", "fg", "activeforeground", "selectforeground", "disabledforeground", "insertbackground",
          "highlightcolor"}


def latar(c):
    if not GELAP or not isinstance(c, str):
        return c
    return PETA_LATAR.get(c.strip().upper(), c)


def tinta(c):
    if not GELAP or not isinstance(c, str):
        return c
    return PETA_TINTA.get(c.strip().upper(), c)


def pasang(root=None) -> None:
    """Pasang penerjemah warna (hanya tema gelap). Panggil sebelum widget dibuat."""
    if not GELAP:
        return
    import tkinter as tk
    if getattr(tk.Misc, "_tema_asli_options", None) is None:
        asli = tk.Misc._options

        def _options(self, cnf, kw=None):
            def ubah(d):
                if not d or not isinstance(d, dict):
                    return d
                d = dict(d)
                for k, v in d.items():
                    kunci = k.rstrip("_")
                    if kunci in _LATAR:
                        d[k] = latar(v)
                    elif kunci in _TINTA:
                        d[k] = tinta(v)
                return d
            return asli(self, ubah(cnf), ubah(kw))

        tk.Misc._tema_asli_options = asli
        tk.Misc._options = _options
    if root is not None:
        # Bawaan untuk widget tanpa warna eksplisit (dialog, Entry, Scrollbar, menu).
        for pola, nilai in (("*Background", "#24201E"), ("*Foreground", "#ECE5DF"),
                            ("*activeBackground", "#37302C"), ("*activeForeground", "#ECE5DF"),
                            ("*selectBackground", "#4A3A31"), ("*selectForeground", "#ECE5DF"),
                            ("*insertBackground", "#ECE5DF"), ("*highlightBackground", "#24201E"),
                            ("*troughColor", "#1D1A18"), ("*Entry.Background", "#1D1A18"),
                            ("*Text.Background", "#1D1A18"), ("*Listbox.Background", "#1D1A18"),
                            ("*Spinbox.Background", "#1D1A18"), ("*TCombobox*Listbox.background", "#1D1A18"),
                            ("*TCombobox*Listbox.foreground", "#ECE5DF"), ("*Menu.Background", "#24201E")):
            root.option_add(pola, nilai)
