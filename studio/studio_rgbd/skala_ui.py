"""Skala antarmuka Studio mengikuti skala layar (Xft.dpi).

Di Wayland dengan ``xwayland-native-scaling``, jendela X digambar pada skala bulat
(Xft.dpi 96 x skala). Teks Tk ikut membesar lewat Xft.dpi, tetapi angka piksel di kode
(lebar panel, tombol, padding, wraplength, ukuran jendela) tidak, sehingga pada 192 dpi
tulisan terpotong di kontrol yang sempit. :func:`pasang` mengaitkan titik pembuatan
widget dan tata letak agar angka itu ikut dikali faktor; pada 96 dpi tidak ada yang berubah.

Skala dibaca SEKALI saat Studio dimulai. Tk tidak menata ulang jendela yang sudah
terbuka, jadi kalau skala layar berubah Studio perlu ditutup-buka (lihat
``Studio._pantau_skala``). ``STUDIO_SKALA`` memaksa faktor tertentu.
"""
from __future__ import annotations

import os
import re
import subprocess
import tkinter as tk

FAKTOR = 1.0
_terpasang = False


def baca_faktor() -> float:
    """Faktor skala layar saat ini; 1,0 bila tidak diketahui (mis. bukan X11)."""
    try:
        return max(1.0, float(os.environ["STUDIO_SKALA"]))
    except (KeyError, ValueError):
        pass
    try:
        keluar = subprocess.run(["xrdb", "-query"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return 1.0
    m = re.search(r"^Xft\.dpi:\s*([\d.]+)", keluar, re.M)
    return max(1.0, float(m.group(1)) / 96) if m else 1.0


def px(n: float) -> int:
    return int(round(n * FAKTOR))


def _kali(nilai):
    if isinstance(nilai, (int, float)) and not isinstance(nilai, bool):
        return px(nilai)
    if isinstance(nilai, (tuple, list)):
        return tuple(_kali(v) for v in nilai)
    return nilai


def _bungkus_init(kelas, opsi, bawaan=None):
    asli = kelas.__init__

    def init(self, master=None, cnf={}, **kw):       # noqa: B006 -- tanda tangan tkinter
        for k, v in (bawaan or {}).items():
            kw.setdefault(k, v)
        for k in opsi:
            if k in kw:
                kw[k] = _kali(kw[k])
        asli(self, master, cnf, **kw)
    kelas.__init__ = init


def _bungkus_tata(kelas, *nama):
    asli = getattr(kelas, nama[0])

    def atur(self, cnf={}, **kw):                    # noqa: B006 -- tanda tangan tkinter
        for k in ("padx", "pady", "ipadx", "ipady"):
            if k in kw:
                kw[k] = _kali(kw[k])
        return asli(self, cnf, **kw)
    for n in nama:
        setattr(kelas, n, atur)


def pasang() -> float:
    """Baca skala layar dan kaitkan pembuatan widget; panggil sebelum widget pertama dibuat."""
    global FAKTOR, _terpasang
    if _terpasang:
        return FAKTOR
    _terpasang = True
    FAKTOR = baca_faktor()
    if FAKTOR == 1.0:
        return FAKTOR
    bingkai = ("width", "height", "padx", "pady")
    teks = ("wraplength", "padx", "pady")
    for kelas, opsi in ((tk.Frame, bingkai), (tk.LabelFrame, bingkai), (tk.Toplevel, bingkai),
                        (tk.Canvas, ("width", "height")), (tk.Text, ("padx", "pady")),
                        (tk.Label, teks), (tk.Button, teks), (tk.Checkbutton, teks), (tk.Radiobutton, teks)):
        _bungkus_init(kelas, opsi)
    _bungkus_init(tk.Scale, ("length", "width", "sliderlength"), {"width": 15, "sliderlength": 30})
    _bungkus_tata(tk.Pack, "pack_configure", "pack")
    _bungkus_tata(tk.Grid, "grid_configure", "grid")

    asli_geometri, asli_minsize = tk.Wm.wm_geometry, tk.Wm.wm_minsize

    def geometri(self, newGeometry=None):            # noqa: N803 -- nama tkinter
        m = re.fullmatch(r"(\d+)x(\d+)(.*)", newGeometry) if newGeometry else None
        if m:
            newGeometry = f"{px(int(m[1]))}x{px(int(m[2]))}{m[3]}"
        return asli_geometri(self, newGeometry)

    def minsize(self, width=None, height=None):
        return asli_minsize(self, None if width is None else px(width), None if height is None else px(height))
    tk.Wm.wm_geometry = tk.Wm.geometry = geometri
    tk.Wm.wm_minsize = tk.Wm.minsize = minsize
    return FAKTOR
