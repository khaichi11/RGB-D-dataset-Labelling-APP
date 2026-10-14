"""Kotak atribut tangga fisik: lokasi, luar/dalam ruang, dan bahan (dipakai tab Tinjau dan Split).

Lokasi menentukan pembagian latih/uji yang adil (satu bangunan utuh di latih atau
di uji). Luar/dalam dan bahan dicatat untuk melaporkan hasil per kondisi.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

try:
    from . import catatan_rekaman as CR
except ImportError:
    import catatan_rekaman as CR

KOSONG = "—"


class AtributTangga(tk.Frame):
    def __init__(self, induk, akar_data, sesudah=None, bg="#FFFDFC", fg_muted="#786962") -> None:
        super().__init__(induk, bg=bg)
        self.akar = Path(akar_data)
        self.sesudah = sesudah                    # dipanggil setelah atribut berubah (segarkan tampilan)
        self.tid: str | None = None
        self.lokasi = ttk.Combobox(self, width=16)
        self.lingkungan = ttk.Combobox(self, width=12, state="readonly", values=(KOSONG,) + CR.LINGKUNGAN)
        self.bahan = ttk.Combobox(self, width=12, values=CR.BAHAN)
        for teks, w in (("Lokasi", self.lokasi), ("Luar/dalam", self.lingkungan), ("Bahan", self.bahan)):
            kol = tk.Frame(self, bg=bg)
            kol.pack(side="left", fill="x", expand=True, padx=(0, 3))
            tk.Label(kol, text=teks, bg=bg, fg=fg_muted, font=("Segoe UI", 8)).pack(anchor="w")
            w.pack(in_=kol, fill="x")
        for w, kunci in ((self.lokasi, "lokasi"), (self.bahan, "bahan")):
            for ev in ("<Return>", "<KP_Enter>", "<<ComboboxSelected>>", "<FocusOut>"):
                w.bind(ev, lambda _e, w=w, kunci=kunci: self._simpan(kunci, w.get()))
        self.lingkungan.bind("<<ComboboxSelected>>", lambda _e: self._simpan("lingkungan", self.lingkungan.get()))
        self.tampilkan(None)

    def tampilkan(self, tid: str | None) -> None:
        daftar = CR.baca_tangga(self.akar)
        self.tid = tid if tid in daftar else None          # ID tanpa entri tangga.json tidak bisa diberi atribut
        info = daftar.get(self.tid or "", {})
        self.lokasi.configure(values=sorted({v.get("lokasi", "") for v in daftar.values() if v.get("lokasi")}))
        keadaan = "normal" if self.tid else "disabled"
        for w in (self.lokasi, self.bahan):
            w.configure(state=keadaan)
        self.lingkungan.configure(state="readonly" if self.tid else "disabled")
        self.lokasi.set(info.get("lokasi", "")) if self.tid else self.lokasi.set("")
        self.lingkungan.set(info.get("lingkungan") or KOSONG)
        self.bahan.set(info.get("bahan", ""))

    def _simpan(self, kunci: str, nilai: str) -> None:
        if not self.tid:
            return
        nilai = "" if nilai == KOSONG else " ".join(nilai.split())
        if CR.baca_tangga(self.akar).get(self.tid, {}).get(kunci, "") == nilai:
            return
        CR.atur_atribut_tangga(self.akar, self.tid, **{kunci: nilai})
        if self.sesudah is not None:
            self.sesudah()
