"""Jumlah frame berlabel per rekaman, dan berapa yang akan menjadi data latih.

Dipakai jendela "📊 Jumlah data" (tab Tinjau) dan dari baris perintah::

    cd kode && .venv/bin/python -m studio_rgbd.hitung_data [--selesai] [--kategori tangga_naik] [--csv]

Definisinya sama dengan yang dipakai pelatihan dan tab Split Dataset:

- status frame = penanda daftar frame Studio (:func:`status_label`);
- "siap latih" = ``frame_bersih`` skrip latih (``rgbd_convnext.data.dataset_d435``):
  berkas wajib lengkap dan tidak di tempat sampah;
- "siap latih, diperiksa" = ``frame_bersih(hanya_diperiksa=True)``, bawaan tab
  Split Dataset ("Hanya label yang sudah divalidasi manual");
- train/val/test: dari ``split_dataset.json`` bila sudah ada, selain itu
  perkiraan pembagian per blok frame 70/15/15 tab Split Dataset (awal rekaman
  train, tengah val, akhir test).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

try:
    from . import catatan_rekaman as CR
except ImportError:
    import catatan_rekaman as CR

# Sama dengan rgbd_convnext.data.dataset_d435.BERKAS_WAJIB (modul itu mengimpor torch).
BERKAS_WAJIB = ("color_raw.png", "depth_aligned_to_color.npy", "mask_objek.png", "mask_acuan.png")
STATUS = ("diperiksa", "claude", "claude_cek", "claude_kosong", "usulan", "")
SET = ("train", "val", "test")
RASIO_BAWAAN = {"train": 0.70, "val": 0.15, "test": 0.15}       # tab Split Dataset, 70/15/15


def _json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def status_label(frame: Path) -> str:
    """'diperiksa' | 'claude_cek' | 'claude' | 'claude_kosong' | 'usulan' | '' (tanpa draf) untuk satu folder frame."""
    j = _json(Path(frame) / "label_draft.json")
    if not j:
        return ""
    if j.get("diperiksa_manual", not j.get("otomatis", False)):
        return "diperiksa"
    if j.get("oleh_claude"):
        if j.get("perlu_dicek"):
            return "claude_cek"
        return "claude" if any(j.get("poligon", {}).values()) else "claude_kosong"
    return "usulan"


def _bagi_blok(n: int, rasio: dict) -> dict:
    """Pembagian per blok frame tab Split Dataset (``_rencana_blok``) untuk n frame satu rekaman."""
    a = round(n * rasio["train"])
    b = a + round(n * rasio["val"])
    return {"train": a, "val": b - a, "test": n - b}


def hitung_rekaman(sesi: Path, split: dict | None = None, rasio: dict = RASIO_BAWAAN) -> dict:
    """Ringkasan satu folder sesi (``rekaman/<kategori>/<sesi>``)."""
    sesi = Path(sesi)
    hasil = {"rekaman": sesi.name, "kategori": sesi.parent.name, "selesai": bool(CR.baca(sesi).get("selesai")),
             "frame": 0, "sampah": 0, "siap_latih": 0, "siap_diperiksa": 0, **{f"st_{s or 'kosong'}": 0 for s in STATUS}}
    per_set = {s: 0 for s in SET + ("abaikan", "belum")}
    for d in sorted((sesi / "exports" / "frames").glob("frame_*")):
        if not d.is_dir():
            continue
        if _json(d / "frame_state.json").get("di_sampah"):
            hasil["sampah"] += 1
            continue
        hasil["frame"] += 1
        hasil[f"st_{status_label(d) or 'kosong'}"] += 1
        if not all((d / n).exists() for n in BERKAS_WAJIB):
            continue
        hasil["siap_latih"] += 1
        j = _json(d / "label_draft.json")
        if not j.get("diperiksa_manual", not j.get("otomatis", False)):      # aturan frame_bersih(hanya_diperiksa)
            continue
        hasil["siap_diperiksa"] += 1
        if split is not None:
            s = split.get("frame", {}).get(f"{sesi.name}/{d.name}") or split.get("rekaman", {}).get(sesi.name)
            per_set[s if s in per_set else "belum"] += 1
    if split is not None:
        hasil.update({f"set_{k}": v for k, v in per_set.items()})
    else:
        hasil.update({f"set_{k}": v for k, v in _bagi_blok(hasil["siap_diperiksa"], rasio).items()})
    hasil["set_dari_split"] = split is not None
    return hasil


def daftar_sesi(akar_data: Path, kategori: str | None = None) -> list[Path]:
    """Sesi yang tampil di daftar tab Tinjau: punya ekspor dan tidak di tempat sampah (``state.json``)."""
    akar = Path(akar_data) / "rekaman"
    pola = f"{kategori}/*" if kategori else "*/*"
    return sorted(p for p in akar.glob(pola) if p.is_dir() and (p / "exports").exists()
                  and not _json(p / "state.json").get("di_sampah"))


def hitung_semua(akar_data: Path, kategori: str | None = None, hanya_selesai: bool = False) -> list[dict]:
    p_split = Path(akar_data) / "split_dataset.json"
    split = _json(p_split) if p_split.exists() else None
    baris = [hitung_rekaman(s, split) for s in daftar_sesi(akar_data, kategori)]
    return [b for b in baris if b["selesai"] or not hanya_selesai]


KOLOM = (("rekaman", "Rekaman"), ("frame", "Frame"), ("st_diperiksa", "✔ diperiksa"), ("st_claude", "🤖 Claude"),
         ("st_claude_cek", "🤖⚠ cek"), ("st_claude_kosong", "🤖∅ kosong"), ("st_kosong", "○ belum"),
         ("siap_latih", "Siap latih"), ("siap_diperiksa", "Siap latih ✔"),
         ("set_train", "Train"), ("set_val", "Val"), ("set_test", "Test"))


def jumlahkan(baris: list[dict]) -> dict:
    total = {k: sum(int(b.get(k, 0)) for b in baris) for k, _ in KOLOM if k != "rekaman"}
    total["rekaman"] = f"{len(baris)} rekaman"
    return total


BG, PANEL, INK, MUTED, LINE = "#F3EEE7", "#FFFDFC", "#382D29", "#786962", "#DED2C8"


def ringkas_teks(baris: list[dict]) -> str:
    t = jumlahkan(baris)
    asal = "split tersimpan" if baris and baris[0]["set_dari_split"] else "perkiraan pembagian per blok 70/15/15"
    return (f"{t['rekaman']} · {CR.ribuan(t['frame'])} frame · {CR.ribuan(t['st_diperiksa'])} diperiksa · "
            f"{CR.ribuan(t['st_claude'] + t['st_claude_cek'])} label Claude belum diperiksa · "
            f"{CR.ribuan(t['st_kosong'])} belum berlabel\n"
            f"Siap latih (label diperiksa): {CR.ribuan(t['siap_diperiksa'])} frame -> train {CR.ribuan(t['set_train'])}, "
            f"val {CR.ribuan(t['set_val'])}, test {CR.ribuan(t['set_test'])} ({asal})")


class JendelaJumlahData(tk.Toplevel):
    """Tabel jumlah frame berlabel per rekaman (tab Tinjau, tombol "📊 Jumlah data")."""

    def __init__(self, studio, kategori: str, daftar_kategori=("tangga_naik",)) -> None:
        super().__init__(studio)
        self.studio = studio
        self.title("Jumlah data berlabel")
        self.configure(bg=BG)
        self.geometry(f"{min(1180, self.winfo_screenwidth() - 80)}x{min(720, self.winfo_screenheight() - 120)}")
        self.kategori = tk.StringVar(value=kategori)
        self.hanya_selesai = tk.BooleanVar(value=True)
        self.ringkas = tk.StringVar(value="Menghitung…")
        self.baris: list[dict] = []
        self._urut = ("rekaman", False)

        atas = tk.Frame(self, bg=BG); atas.pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(atas, text="Kategori", bg=BG, fg=MUTED).pack(side="left")
        kat = ttk.Combobox(atas, textvariable=self.kategori, values=tuple(daftar_kategori), state="readonly", width=12)
        kat.pack(side="left", padx=(4, 12))
        kat.bind("<<ComboboxSelected>>", lambda e: self.hitung())
        tk.Checkbutton(atas, text="Hanya rekaman ✔ selesai", variable=self.hanya_selesai, command=self.hitung,
                       bg=BG, fg=INK, selectcolor=PANEL, activebackground=BG).pack(side="left")
        tk.Button(atas, text="Salin tabel (CSV)", command=self.salin, bg="#E8DDD5", fg=INK, relief="flat",
                  padx=10).pack(side="right")
        tk.Button(atas, text="↻ Hitung ulang", command=self.hitung, bg="#E8DDD5", fg=INK, relief="flat",
                  padx=10).pack(side="right", padx=(0, 6))
        tk.Label(self, textvariable=self.ringkas, bg=BG, fg=INK, justify="left", anchor="w",
                 font=("Segoe UI", 10, "bold")).pack(fill="x", padx=12, pady=(4, 2))
        tk.Label(self, text="Siap latih = berkas lengkap, bukan sampah (sama dengan skrip latih). Siap latih ✔ = hanya "
                            "label yang sudah diperiksa manual, bawaan tab Split Dataset. Train/Val/Test dari "
                            "split_dataset.json bila sudah dibuat, selain itu perkiraan per blok 70/15/15. "
                            "Klik judul kolom untuk mengurutkan; klik dua kali baris untuk membuka rekamannya.",
                 bg=BG, fg=MUTED, justify="left", anchor="w", wraplength=1100,
                 font=("Segoe UI", 8)).pack(fill="x", padx=12, pady=(0, 6))
        bingkai = tk.Frame(self, bg=BG); bingkai.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.tabel = ttk.Treeview(bingkai, columns=[k for k, _ in KOLOM], show="headings")
        for k, judul in KOLOM:
            self.tabel.heading(k, text=judul, command=lambda k=k: self._urutkan(k))
            self.tabel.column(k, width=190 if k == "rekaman" else 96, anchor="w" if k == "rekaman" else "e",
                              stretch=k == "rekaman")
        self.tabel.tag_configure("total", font=("Segoe UI", 9, "bold"))
        gulir = ttk.Scrollbar(bingkai, orient="vertical", command=self.tabel.yview)
        self.tabel.configure(yscrollcommand=gulir.set)
        self.tabel.pack(side="left", fill="both", expand=True)
        gulir.pack(side="right", fill="y")
        self.tabel.bind("<Double-1>", self._buka)
        self.bind("<Escape>", lambda e: self.destroy())
        self.hitung()

    def hitung(self) -> None:
        self.ringkas.set("Menghitung…")
        akar, kat, selesai = Path(self.studio.root_data), self.kategori.get() or None, self.hanya_selesai.get()

        kotak: list = []                     # diisi thread; dibaca thread UI lewat _tunggu (Tk tidak aman lintas thread)

        def kerja():
            try:
                kotak.append(hitung_semua(akar, kat, selesai))
            except Exception as e:                                   # noqa: BLE001
                kotak.append(e)

        self._kotak = kotak
        threading.Thread(target=kerja, daemon=True).start()
        self.after(100, lambda: self._tunggu(kotak))

    def _tunggu(self, kotak: list) -> None:
        if kotak is not self._kotak or not self.winfo_exists():
            return                           # hitungan lama yang sudah digantikan
        if not kotak:
            self.after(100, lambda: self._tunggu(kotak))
            return
        self._tampilkan(kotak[0])

    def _tampilkan(self, hasil) -> None:
        if not self.winfo_exists():
            return
        if isinstance(hasil, Exception):
            self.ringkas.set(f"Gagal menghitung: {hasil}")
            return
        self.baris = hasil
        self.ringkas.set(ringkas_teks(hasil) if hasil else "Tidak ada rekaman yang cocok.")
        self._isi()

    def _isi(self) -> None:
        self.tabel.delete(*self.tabel.get_children())
        k, turun = self._urut
        for b in sorted(self.baris, key=lambda b: b.get(k, 0), reverse=turun):
            nama = ("✔ " if b["selesai"] else "") + b["rekaman"].replace("TANGGA_NAIK_", "")
            self.tabel.insert("", "end", iid=b["rekaman"], values=[nama] + [b.get(k2, "") for k2, _ in KOLOM[1:]])
        if self.baris:
            t = jumlahkan(self.baris)
            self.tabel.insert("", "end", iid="__total__", tags=("total",),
                              values=[f"Jumlah ({t['rekaman']})"] + [t[k2] for k2, _ in KOLOM[1:]])

    def _urutkan(self, k: str) -> None:
        self._urut = (k, not self._urut[1] if self._urut[0] == k else k != "rekaman")
        self._isi()

    def salin(self) -> None:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([k for k, _ in KOLOM])
        w.writerows([[b.get(k, "") for k, _ in KOLOM] for b in self.baris + ([jumlahkan(self.baris)] if self.baris else [])])
        self.clipboard_clear()
        self.clipboard_append(buf.getvalue())
        self.ringkas.set(ringkas_teks(self.baris) + "\nTabel disalin (CSV); tempel di spreadsheet.")

    def _buka(self, _e) -> None:
        iid = self.tabel.focus()
        peta = getattr(self.studio, "_map_sesi", None)
        if not iid or iid == "__total__" or peta is None:
            return
        for i, sesi in enumerate(peta):
            if sesi is not None and Path(sesi).name == iid:
                lst = self.studio.list_sesi
                lst.selection_clear(0, "end"); lst.selection_set(i); lst.see(i)
                self.studio.pilih_sesi()
                return
        self.ringkas.set(ringkas_teks(self.baris) + f"\n{iid} tidak ada di daftar tab Tinjau (kategori lain atau di sampah).")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--akar", default=str(Path(__file__).resolve().parents[2] / "dataset" / "studio_rgbd"))
    ap.add_argument("--kategori", default="tangga_naik", help="kosongkan ('') untuk semua kategori")
    ap.add_argument("--selesai", action="store_true", help="hanya rekaman bertanda ✔ selesai")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args(argv)
    baris = hitung_semua(Path(a.akar), a.kategori or None, a.selesai)
    if a.csv:
        w = csv.writer(sys.stdout)
        w.writerow([k for k, _ in KOLOM])
        w.writerows([[b.get(k, "") for k, _ in KOLOM] for b in baris + [jumlahkan(baris)]])
        return
    lebar = [max(len(j), 8) for _, j in KOLOM]
    lebar[0] = 28
    print("  ".join(j.rjust(w) if i else j.ljust(w) for i, ((_, j), w) in enumerate(zip(KOLOM, lebar))))
    for b in baris + [jumlahkan(baris)]:
        nama = ("✔ " if b.get("selesai") else "  ") + str(b["rekaman"]).replace("TANGGA_NAIK_", "")
        print("  ".join((nama.ljust(w) if i == 0 else str(b.get(k, "")).rjust(w))
                        for i, ((k, _), w) in enumerate(zip(KOLOM, lebar))))
    if baris and not baris[0]["set_dari_split"]:
        print("\nTrain/Val/Test: perkiraan pembagian per blok 70/15/15 dari frame siap latih ✔ "
              "(split_dataset.json belum ada).")


if __name__ == "__main__":
    main()
