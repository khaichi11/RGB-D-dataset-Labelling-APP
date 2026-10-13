"""Tempat sampah Studio: frame/rekaman yang "dihapus permanen" dipindah ke sini, bisa dipulihkan.

Sebelum 13 Oktober 2026, "Hapus permanen" langsung menghapus folder (shutil.rmtree)
dan tidak bisa dibatalkan. Sekarang folder dipindah (``os.replace``, seketika pada
disk yang sama) ke ``<dataset>/tempat_sampah/<waktu>__<nama>`` beserta ``asal.json``
yang mencatat lokasi aslinya. Ruang disk baru kembali setelah "Hapus selamanya" atau
"Kosongkan" di jendela Tempat sampah (tab Tinjau, kartu Rekaman).

Berbeda dengan "Sampahkan" (frame/rekaman hanya DITANDAI di ``frame_state.json`` /
``state.json`` dan tetap di tempatnya), isi tempat sampah ini tidak ikut daftar,
pelatihan, maupun ekspor.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

BG, PANEL, INK, MUTED, LINE = "#F3EEE7", "#FFFDFC", "#382D29", "#786962", "#DED2C8"
ASAL = "asal.json"


def folder(akar_data: Path) -> Path:
    p = Path(akar_data) / "tempat_sampah"
    p.mkdir(exist_ok=True)
    return p


def buang(akar_data: Path, target: Path, jenis: str) -> Path:
    """Pindahkan ``target`` (folder frame atau sesi) ke tempat sampah Studio. -> lokasi barunya."""
    target = Path(target).resolve()
    akar = Path(akar_data).resolve()
    relatif = target.relative_to(akar)                   # gagal (ValueError) bila di luar dataset
    nama = "__".join(relatif.parts[1:]) if relatif.parts[0] == "rekaman" else "__".join(relatif.parts)
    tujuan = folder(akar) / f"{datetime.now():%Y%m%d_%H%M%S}__{nama.replace('__exports__frames', '')}"
    os.replace(target, tujuan)
    (tujuan / ASAL).write_text(json.dumps({"asal": str(relatif), "jenis": jenis,
                                           "dihapus_iso": datetime.now().isoformat(timespec="seconds")},
                                          indent=1, ensure_ascii=False), encoding="utf-8")
    return tujuan


def isi(akar_data: Path) -> list[dict]:
    hasil = []
    for p in sorted(folder(akar_data).iterdir(), reverse=True):
        if not p.is_dir():
            continue
        try:
            a = json.loads((p / ASAL).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            a = {}
        hasil.append({"path": p, "asal": a.get("asal", ""), "jenis": a.get("jenis", "?"),
                      "dihapus": a.get("dihapus_iso", "")})
    return hasil


def ukuran(p: Path) -> int:
    n = 0
    for akar, _, berkas in os.walk(p):
        for b in berkas:
            try:
                n += os.lstat(os.path.join(akar, b)).st_size
            except OSError:
                pass
    return n


def pulihkan(akar_data: Path, item: dict) -> Path:
    asal = Path(akar_data) / item["asal"]
    if not item["asal"]:
        raise ValueError("lokasi asal tidak tercatat")
    if asal.exists():
        raise FileExistsError(f"sudah ada folder di lokasi asal:\n{asal}")
    asal.parent.mkdir(parents=True, exist_ok=True)
    (item["path"] / ASAL).unlink(missing_ok=True)
    os.replace(item["path"], asal)
    return asal


class JendelaTempatSampah(tk.Toplevel):
    def __init__(self, studio) -> None:
        super().__init__(studio)
        self.studio, self.akar = studio, Path(studio.root_data)
        self.title("Tempat sampah Studio")
        self.configure(bg=BG)
        self.geometry("760x480")
        self.item: list[dict] = []
        self.info = tk.StringVar()
        tk.Label(self, text="Frame dan rekaman yang dihapus dengan \"Hapus permanen\". Pulihkan mengembalikannya ke "
                            "tempat semula; ruang disk baru kembali setelah Hapus selamanya / Kosongkan.",
                 bg=BG, fg=MUTED, wraplength=720, justify="left").pack(anchor="w", padx=12, pady=(10, 4))
        self.daftar = tk.Listbox(self, bg=PANEL, fg=INK, relief="flat", selectmode="extended", activestyle="none",
                                 font=("Consolas", 9), highlightthickness=1, highlightbackground=LINE)
        self.daftar.pack(fill="both", expand=True, padx=12)
        tk.Label(self, textvariable=self.info, bg=BG, fg=INK).pack(anchor="w", padx=12, pady=(4, 0))
        baris = tk.Frame(self, bg=BG)
        baris.pack(fill="x", padx=12, pady=10)
        tombol = getattr(studio, "tombol_ringkas", None)
        for teks, perintah, warna in (("♻ Pulihkan", self.pulihkan, "#CDEBC5"), ("Hapus selamanya", self.hapus, "#F3D8D4"),
                                      ("Kosongkan semua", self.kosongkan, "#F3D8D4")):
            b = tombol(baris, teks, perintah, warna, INK, width=150) if tombol else tk.Button(baris, text=teks, command=perintah)
            b.pack(side="left", padx=(0, 6))
        self.muat()

    def muat(self) -> None:
        self.item = isi(self.akar)
        self.daftar.delete(0, "end")
        for it in self.item:
            self.daftar.insert("end", f"{it['dihapus'][:16].replace('T', ' '):16s}  {it['jenis']:8s}  {it['asal']}")
        self.info.set(f"{len(self.item)} item" if self.item else "Tempat sampah kosong.")
        threading.Thread(target=self._hitung_ukuran, args=(list(self.item),), daemon=True).start()

    def _hitung_ukuran(self, item) -> None:
        total = sum(ukuran(it["path"]) for it in item)
        try:
            self.after(0, lambda: self.winfo_exists() and self.info.set(f"{len(item)} item · {total / 1e9:.2f} GB"))
        except (RuntimeError, tk.TclError):
            pass

    def _terpilih(self) -> list[dict]:
        pilih = [self.item[i] for i in self.daftar.curselection() if i < len(self.item)]
        if not pilih:
            messagebox.showinfo("Pilih item", "Pilih item di daftar dahulu.", parent=self)
        return pilih

    def _segarkan_studio(self) -> None:
        for nama in ("muat_daftar", "muat_frame"):
            f = getattr(self.studio, nama, None)
            if f is None or (nama == "muat_frame" and not getattr(self.studio, "sesi", None)):
                continue
            try:
                f()
            except Exception:                                    # noqa: BLE001 -- penyegaran tidak boleh menggagalkan
                pass

    def pulihkan(self) -> None:
        gagal = []
        for it in self._terpilih():
            try:
                pulihkan(self.akar, it)
            except (OSError, ValueError) as e:
                gagal.append(f"{it['asal']}: {e}")
        if gagal:
            messagebox.showerror("Pulihkan", "\n\n".join(gagal), parent=self)
        self.muat()
        self._segarkan_studio()

    def hapus(self, semua: bool = False) -> None:
        pilih = list(self.item) if semua else self._terpilih()
        if not pilih:
            return
        if not messagebox.askyesno("Hapus selamanya", f"Hapus SELAMANYA {len(pilih)} item dari disk?\n"
                                   "Ini tidak bisa dibatalkan.", icon="warning", parent=self):
            return
        for it in pilih:
            shutil.rmtree(it["path"], ignore_errors=True)
        self.muat()

    def kosongkan(self) -> None:
        self.hapus(semua=True)
