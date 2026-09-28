"""Pembantu tata letak Tk yang dipakai beberapa tab Studio."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def kolom_gulir(induk, lebar: int, bg: str, **pack) -> tk.Frame:
    """Kolom selebar ``lebar`` yang bisa di-scroll; kembalikan bingkai isinya.

    Panel kontrol berukuran tetap kehabisan tempat pada layar laptop pendek, dan
    Tk menyembunyikan atau menjepit kontrol yang tidak kebagian tempat. Roda
    tetikus hanya menggulir kolom ini bila kursor berada di atasnya; daftar dan
    kotak teks di dalamnya tetap menggulir dirinya sendiri.
    """
    luar = tk.Frame(induk, bg=bg, width=lebar)
    luar.pack(**({"side": "left", "fill": "y"} | pack))
    luar.pack_propagate(False)
    kanvas = tk.Canvas(luar, bg=bg, highlightthickness=0)
    bilah = ttk.Scrollbar(luar, orient="vertical", command=kanvas.yview)
    kanvas.configure(yscrollcommand=bilah.set)
    bilah.pack(side="right", fill="y")
    kanvas.pack(side="left", fill="both", expand=True)
    isi = tk.Frame(kanvas, bg=bg)
    jendela = kanvas.create_window((0, 0), window=isi, anchor="nw")
    isi.bind("<Configure>", lambda _e: kanvas.configure(scrollregion=kanvas.bbox("all")))
    kanvas.bind("<Configure>", lambda e: kanvas.itemconfigure(jendela, width=e.width))
    akar = induk.winfo_toplevel()

    def gulir(e, langkah):
        w = akar.winfo_containing(e.x_root, e.y_root)
        if isinstance(w, (tk.Listbox, tk.Text)):
            return None
        while w is not None and w is not kanvas:
            w = getattr(w, "master", None)
        if w is None or kanvas.yview() == (0.0, 1.0):
            return None
        kanvas.yview_scroll(langkah, "units")
        return "break"
    akar.bind_all("<Button-4>", lambda e: gulir(e, -3), add="+")
    akar.bind_all("<Button-5>", lambda e: gulir(e, 3), add="+")
    akar.bind_all("<MouseWheel>", lambda e: gulir(e, -3 if e.delta > 0 else 3), add="+")
    return isi
