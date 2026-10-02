"""Alat uji: cari kontrol Studio yang terpotong, tersembunyi, atau terjepit pada ukuran layar tertentu.

Tk tidak menggambar widget di luar batas; ia menyembunyikan (unmap) atau menjepit
widget yang tidak kebagian tempat. Alat ini memeriksa setiap tombol/kontrol di
setiap tab, kecuali yang berada di panel yang bisa di-scroll.

    cd kode && .venv/bin/python -m studio_rgbd.uji_tata_letak 1366 700 [folder_dataset]
"""
from __future__ import annotations

import sys
import time
import types
import tkinter as tk
from tkinter import ttk


def audit(lebar: int, tinggi: int, keluar: str) -> list[str]:
    from . import studio_dataset_rgbd as S
    args = types.SimpleNamespace(keluar=keluar, lebar=848, tinggi=480, fps=30, preset="jangan",
                                 batas_frame=8000, exposure=0, fps_preview=20)
    app = S.Studio(args)
    app.minsize(1, 1)
    app.geometry(f"{lebar}x{tinggi}+0+0")

    def jalan(d):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < d:
            app.update(); time.sleep(.002)
    jalan(2)
    app.tabs.select(app.tab_tinjau); app.muat_daftar(); jalan(.3)
    if app.list_sesi.size():
        app.list_sesi.selection_set(0); app.pilih_sesi(); jalan(.3)
        app.tabs.select(app.tab_label); app.muat_frame()
        if app.frame_paths:
            app._pilih_indeks_frame(0); jalan(.5)
    kontrol = (tk.Button, tk.Checkbutton, tk.Radiobutton, tk.Scale, tk.Entry, tk.Spinbox, tk.Listbox, tk.Text,
               ttk.Combobox, ttk.Button, ttk.Entry, ttk.Treeview, ttk.Checkbutton)

    def dalam_gulir(w):
        while w is not None:
            if isinstance(w.master, tk.Canvas) and w.master.cget("yscrollcommand"):
                return True
            w = w.master
        return False

    def teks(w):
        for k in ("text", "label"):
            try:
                t = w.cget(k)
                if t:
                    return str(t)[:40]
            except tk.TclError:
                pass
        return type(w).__name__

    laporan = []
    for tab in (app.tab_rekam, app.tab_tinjau, app.tab_ekspor, app.tab_label, app.tab_uji, app.tab_uji_berkas, app.tab_split):
        app.tabs.select(tab); jalan(1.0)
        bx0, by0 = tab.winfo_rootx(), tab.winfo_rooty()
        bx1, by1 = bx0 + tab.winfo_width(), by0 + tab.winfo_height()
        stack = [tab]
        while stack:
            w = stack.pop(); stack += w.winfo_children()
            if not isinstance(w, kontrol) or dalam_gulir(w):
                continue
            nama = app.tabs.tab(tab, "text").strip()
            if not w.winfo_ismapped():
                if w.winfo_manager() in ("pack", "grid") and w.master.winfo_ismapped():
                    laporan.append(f"{nama}: {teks(w)} tidak kebagian tempat")
                continue
            y1 = w.winfo_rooty() + w.winfo_height(); x1 = w.winfo_rootx() + w.winfo_width()
            jepit = not isinstance(w, (tk.Listbox, tk.Text, ttk.Treeview)) and \
                (w.winfo_reqheight() > w.winfo_height() + 3 or w.winfo_reqwidth() > w.winfo_width() + 12)
            if jepit or y1 > by1 + 1 or x1 > bx1 + 1:
                laporan.append(f"{nama}: {teks(w)} terjepit/terpotong ({w.winfo_width()}x{w.winfo_height()})")
    app.destroy()
    return laporan


if __name__ == "__main__":
    lebar, tinggi = int(sys.argv[1]), int(sys.argv[2])
    hasil = audit(lebar, tinggi, sys.argv[3] if len(sys.argv) > 3 else "/tmp/studio_uji")
    print(f"{lebar}x{tinggi}: " + ("tidak ada kontrol terpotong" if not hasil else f"{len(hasil)} masalah"))
    for h in hasil:
        print("  ", h)
