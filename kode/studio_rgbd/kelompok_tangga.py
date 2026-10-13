"""Jendela "Kelompokkan tangga": tandai rekaman yang merekam tangga fisik yang sama.

Dibuka dari kartu catatan tab Tinjau (tombol "Kelompokkan…"). Semua rekaman
satu kategori tampil sebagai kartu bergambar, urut waktu rekam. Gambarnya frame
ekspor berlabel dengan poligon terbanyak (frame paling jelas memperlihatkan
tangga); frame gelap memakai citra IR bila ada, atau dicerahkan.

- Klik kartu = pilih/lepas; Shift+klik = pilih rentang; Ctrl+A = pilih semua.
- "Jadikan tangga baru" membuat ID tangga (T01, T02, ...) dan memasangnya ke
  semua rekaman terpilih; "Pasang ke tangga" memakai tangga yang dipilih di
  daftar kanan; "Lepas" menghapus tangga rekaman terpilih.
- Klik tangga di daftar kanan = pilih semua rekamannya.

Yang ditulis sama dengan kotak "Tangga fisik" di kartu catatan: ``tangga`` di
``catatan.json`` rekaman dan daftar di ``tangga.json``. Tangga per scene (satu
rekaman melewati beberapa tangga) tetap diatur dari daftar scene tab Tinjau.
"""
from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from . import catatan_rekaman as CR
except ImportError:
    import catatan_rekaman as CR

BG, PANEL, INK, MUTED, LINE, ACCENT, GREEN = ("#F3EEE7", "#FFFDFC", "#382D29", "#786962", "#DED2C8",
                                             "#754C3B", "#6B8B63")
W_GAMBAR, H_GAMBAR, KOLOM = 224, 126, 4


def _json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def frame_wakil(sesi: Path) -> Path | None:
    """Frame ekspor berpoligon terbanyak (yang paling jelas memperlihatkan tangga), atau frame tengah."""
    frames = sorted(d for d in (sesi / "exports" / "frames").glob("frame_*") if d.is_dir())
    if not frames:
        return None
    terbaik, n_terbaik = None, 0
    for d in frames:
        n = sum(len(v) for v in (_json(d / "label_draft.json").get("poligon") or {}).values())
        if n > n_terbaik:
            terbaik, n_terbaik = d, n
    return terbaik or frames[len(frames) // 2]


def gambar_wakil(sesi: Path) -> np.ndarray | None:
    """Citra RGB kecil untuk kartu; IR atau dicerahkan bila frame gelap."""
    d = frame_wakil(sesi)
    if d is None:
        thumb = sesi / "derived" / "thumbnail.jpg"
        img = cv2.imread(str(thumb)) if thumb.exists() else None
        if img is not None:
            img = img[:, : img.shape[1] // 2] if img.shape[1] > 2 * img.shape[0] else img
    else:
        img = cv2.imread(str(d / "color_raw.png"))
        if img is not None and float(img.mean()) < 45:
            ir = cv2.imread(str(d / "ir_left_raw.png"), cv2.IMREAD_GRAYSCALE)
            if ir is not None and ir.shape[:2] == img.shape[:2] and float(ir.mean()) > 20:
                img = cv2.cvtColor(cv2.createCLAHE(2.0, (8, 8)).apply(ir), cv2.COLOR_GRAY2BGR)
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[..., 0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if img is None:
        return None
    return cv2.cvtColor(cv2.resize(img, (W_GAMBAR, H_GAMBAR), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


class JendelaKelompokTangga(tk.Toplevel):
    def __init__(self, studio, kategori: str = "tangga_naik") -> None:
        super().__init__(studio)
        self.studio = studio
        self.akar = Path(studio.root_data)
        self.kategori = kategori
        self.title("Kelompokkan rekaman per tangga fisik")
        self.configure(bg=BG)
        self.geometry(f"{min(1320, self.winfo_screenwidth() - 60)}x{min(860, self.winfo_screenheight() - 100)}")
        self.sesi: list[Path] = []
        self.pilih: set[Path] = set()
        self._jangkar: int | None = None
        self._kartu: dict[Path, dict] = {}
        self._foto: dict[Path, ImageTk.PhotoImage] = {}
        self.info = tk.StringVar()
        self._bangun_ui()
        self.bind("<Control-a>", lambda e: self.pilih_semua())
        self.bind("<Escape>", lambda e: self.lepas_pilihan())
        self.muat()

    # ------------------------------------------------------------ tata letak
    def _tombol(self, induk, teks, perintah, warna="#E8DDD5", fg=INK, lebar=150):
        if hasattr(self.studio, "tombol_ringkas"):
            return self.studio.tombol_ringkas(induk, teks, perintah, warna, fg, width=lebar)
        return tk.Button(induk, text=teks, command=perintah, bg=warna, fg=fg, relief="flat")

    def _bangun_ui(self) -> None:
        kanan = tk.Frame(self, bg=PANEL, width=320, highlightthickness=1, highlightbackground=LINE)
        kanan.pack(side="right", fill="y", padx=(0, 10), pady=10)
        kanan.pack_propagate(False)
        kiri = tk.Frame(self, bg=BG)
        kiri.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        tk.Label(kiri, text="Pilih rekaman yang merekam tangga yang SAMA (klik; Shift+klik rentang; Ctrl+A semua), "
                            "lalu tekan Jadikan tangga baru atau Pasang ke tangga. Urut waktu rekam.",
                 bg=BG, fg=MUTED, wraplength=900, justify="left").pack(anchor="w")
        tk.Label(kiri, textvariable=self.info, bg=BG, fg=INK, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 6))
        bingkai = tk.Frame(kiri, bg=BG)
        bingkai.pack(fill="both", expand=True)
        self.kanvas = tk.Canvas(bingkai, bg=BG, highlightthickness=0)
        gulir = tk.Scrollbar(bingkai, orient="vertical", command=self.kanvas.yview)
        self.kanvas.configure(yscrollcommand=gulir.set)
        gulir.pack(side="right", fill="y")
        self.kanvas.pack(side="left", fill="both", expand=True)
        self.isi = tk.Frame(self.kanvas, bg=BG)
        self.kanvas.create_window((0, 0), window=self.isi, anchor="nw")
        self.isi.bind("<Configure>", lambda e: self.kanvas.configure(scrollregion=self.kanvas.bbox("all")))
        for urutan, d in (("<Button-4>", -3), ("<Button-5>", 3)):
            self.kanvas.bind_all(urutan, lambda e, d=d: self._gulir(e, d), add="+")
        self.kanvas.bind_all("<MouseWheel>", lambda e: self._gulir(e, -3 if e.delta > 0 else 3), add="+")

        isi = tk.Frame(kanan, bg=PANEL)
        isi.pack(fill="both", expand=True, padx=12, pady=10)
        tk.Label(isi, text="Rekaman terpilih", bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_pilih = tk.Label(isi, text="", bg=PANEL, fg=MUTED, wraplength=290, justify="left")
        self.lbl_pilih.pack(anchor="w", pady=(0, 6))
        self._tombol(isi, "+ Jadikan tangga baru", self.tangga_baru, GREEN, "white", 290).pack(fill="x")
        self._tombol(isi, "Pasang ke tangga yang dipilih ↓", self.pasang, ACCENT, "white", 290).pack(fill="x", pady=(4, 0))
        self._tombol(isi, "Lepas tangga", self.lepas, "#F3D8D4", INK, 290).pack(fill="x", pady=(4, 0))
        tk.Frame(isi, bg=LINE, height=1).pack(fill="x", pady=10)
        tk.Label(isi, text="Tangga fisik (klik = pilih rekamannya)", bg=PANEL, fg=INK,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.daftar = tk.Listbox(isi, bg="#FFF9F4", fg=INK, relief="flat", height=16, exportselection=False,
                                 activestyle="none", selectbackground="#E9D8CC")
        self.daftar.pack(fill="both", expand=True, pady=(4, 0))
        self.daftar.bind("<<ListboxSelect>>", lambda e: self._klik_tangga())
        baris = tk.Frame(isi, bg=PANEL)
        baris.pack(fill="x", pady=(4, 0))
        self._tombol(baris, "Ganti nama", self.ganti_nama, lebar=140).pack(side="left", fill="x", expand=True)
        self._tombol(baris, "↻ Muat ulang", self.muat, lebar=140).pack(side="left", fill="x", expand=True, padx=(4, 0))
        tk.Label(isi, text="Tangga yang sama di train dan test membuat angka uji terlalu optimis. Rekaman yang "
                           "melewati beberapa tangga: atur per scene di tab Tinjau.",
                 bg=PANEL, fg=MUTED, wraplength=290, justify="left", font=("Segoe UI", 8)).pack(anchor="w", pady=(8, 0))

    def _gulir(self, e, d) -> None:
        try:
            if str(e.widget).startswith(str(self)):
                self.kanvas.yview_scroll(d, "units")
        except tk.TclError:
            pass

    # ------------------------------------------------------------ data
    def muat(self) -> None:
        dasar = self.akar / "rekaman" / self.kategori
        sesi = [p for p in dasar.glob("*") if p.is_dir() and not _json(p / "state.json").get("di_sampah")
                and ((p / "exports").exists() or any((p / "source").glob("raw.*")))]
        waktu = {p: (CR.waktu_rekaman(p) or (None,))[0] for p in sesi}
        self.sesi = sorted(sesi, key=lambda p: (waktu[p].timestamp() if waktu[p] else 0, p.name))
        self._waktu = waktu
        self.pilih &= set(self.sesi)
        for w in self.isi.winfo_children():
            w.destroy()
        self._kartu.clear()
        for i, p in enumerate(self.sesi):
            self._buat_kartu(i, p)
        self._segarkan()
        threading.Thread(target=self._muat_gambar, args=(list(self.sesi),), daemon=True).start()

    def _muat_gambar(self, daftar: list[Path]) -> None:
        for p in daftar:
            if p in self._foto:
                continue
            try:
                img = gambar_wakil(p)
            except Exception:                                    # noqa: BLE001 -- kartu tetap tanpa gambar
                img = None
            if img is not None:
                try:
                    self.after(0, lambda p=p, img=img: self._pasang_gambar(p, img))
                except (RuntimeError, tk.TclError):
                    return

    def _pasang_gambar(self, p: Path, img: np.ndarray) -> None:
        if not self.winfo_exists() or p not in self._kartu:
            return
        self._foto[p] = ImageTk.PhotoImage(Image.fromarray(img))
        self._kartu[p]["gambar"].configure(image=self._foto[p], width=W_GAMBAR, height=H_GAMBAR)

    def _buat_kartu(self, i: int, p: Path) -> None:
        kartu = tk.Frame(self.isi, bg=PANEL, highlightthickness=3, highlightbackground=PANEL, cursor="hand2")
        kartu.grid(row=i // KOLOM, column=i % KOLOM, padx=5, pady=5, sticky="n")
        gambar = tk.Label(kartu, bg="#222222", width=30, height=7, text="memuat…", fg="#999999")
        if p in self._foto:
            gambar.configure(image=self._foto[p], width=W_GAMBAR, height=H_GAMBAR)
        gambar.pack()
        judul = tk.Label(kartu, text="", bg=PANEL, fg=INK, font=("Segoe UI", 9, "bold"), anchor="w", width=30)
        judul.pack(fill="x", padx=4)
        sub = tk.Label(kartu, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 8), anchor="w", justify="left",
                       wraplength=W_GAMBAR - 4, width=30)
        sub.pack(fill="x", padx=4, pady=(0, 4))
        self._kartu[p] = {"kartu": kartu, "gambar": gambar, "judul": judul, "sub": sub}
        for w in (kartu, gambar, judul, sub):
            w.bind("<Button-1>", lambda e, i=i: self._klik(i, False))
            w.bind("<Shift-Button-1>", lambda e, i=i: self._klik(i, True))

    def _segarkan(self) -> None:
        daftar = CR.baca_tangga(self.akar)
        jumlah: dict[str, int] = {}
        for p in self.sesi:
            c = CR.baca(p)
            tid = c.get("tangga") or ""
            if tid:
                jumlah[tid] = jumlah.get(tid, 0) + 1
            scene = sorted({s.get("tangga") for s in c.get("scene", []) if s.get("tangga")})
            w = self._waktu.get(p)
            k = self._kartu[p]
            warna = CR.WARNA.get((daftar.get(tid) or {}).get("warna")) if tid else None
            k["judul"].configure(text=("✔ " if c.get("selesai") else "") + p.name.replace("TANGGA_NAIK_", ""),
                                 bg=warna or PANEL)
            k["sub"].configure(text=(f"{w:%d %b %H.%M} · " if w else "")
                               + (CR.label_tangga(tid, daftar) if tid else "belum ada tangga")
                               + (f" · scene: {', '.join(scene)}" if scene else "")
                               + (f"\n{c['catatan'][:60]}" if c.get("catatan") else ""))
            k["kartu"].configure(highlightbackground=ACCENT if p in self.pilih else (warna or PANEL))
        pilih_lama = self._tid_terpilih()
        self.daftar.delete(0, "end")
        self._daftar_id = list(daftar)
        for tid in self._daftar_id:
            info = daftar[tid]
            self.daftar.insert("end", f"{CR.label_tangga(tid, daftar)}  ({jumlah.get(tid, 0)} rekaman)"
                               + (f" — {info['lokasi']}" if info.get("lokasi") else ""))
            if info.get("warna") in CR.WARNA:
                self.daftar.itemconfigure("end", background=CR.WARNA[info["warna"]])
        if pilih_lama in self._daftar_id:
            self.daftar.selection_set(self._daftar_id.index(pilih_lama))
        tanpa = sum(1 for p in self.sesi if not CR.baca(p).get("tangga"))
        self.info.set(f"{len(self.sesi)} rekaman · {len(daftar)} tangga fisik · {tanpa} rekaman belum diberi tangga")
        self.lbl_pilih.configure(text=f"{len(self.pilih)} rekaman dipilih" if self.pilih else "belum ada (klik kartu)")

    # ------------------------------------------------------------ pilihan
    def _klik(self, i: int, rentang: bool) -> None:
        p = self.sesi[i]
        if rentang and self._jangkar is not None:
            a, b = sorted((self._jangkar, i))
            self.pilih |= set(self.sesi[a:b + 1])
        elif p in self.pilih:
            self.pilih.discard(p)
        else:
            self.pilih.add(p)
        self._jangkar = i
        self._segarkan()

    def pilih_semua(self) -> None:
        self.pilih = set(self.sesi)
        self._segarkan()

    def lepas_pilihan(self) -> None:
        self.pilih.clear()
        self._segarkan()

    def _tid_terpilih(self) -> str | None:
        sel = self.daftar.curselection()
        return self._daftar_id[sel[0]] if sel and sel[0] < len(getattr(self, "_daftar_id", [])) else None

    def _klik_tangga(self) -> None:
        tid = self._tid_terpilih()
        if tid:
            self.pilih = {p for p in self.sesi if CR.baca(p).get("tangga") == tid}
            self._segarkan()
            if self.pilih:
                pertama = min(self.sesi.index(p) for p in self.pilih)
                self.update_idletasks()
                self.kanvas.yview_moveto(self._kartu[self.sesi[pertama]]["kartu"].winfo_y()
                                         / max(self.isi.winfo_height(), 1))

    # ------------------------------------------------------------ aksi
    def _tulis(self, tid: str) -> None:
        if hasattr(self.studio, "_simpan_catatan_sekarang"):
            self.studio._simpan_catatan_sekarang()       # catatan yang sedang diketik tidak tertimpa
        for p in self.pilih:
            c = CR.baca(p)
            c["tangga"] = tid
            CR.tulis(p, c)
        for p in self.pilih:
            if hasattr(self.studio, "_segarkan_item_sesi"):
                self.studio._segarkan_item_sesi(p)
        if getattr(self.studio, "sesi", None) in self.pilih and hasattr(self.studio, "_muat_catatan_ui"):
            self.studio._muat_catatan_ui()
        self._segarkan()

    def _perlu_pilihan(self) -> bool:
        if not self.pilih:
            messagebox.showinfo("Pilih rekaman", "Klik kartu rekaman yang merekam tangga yang sama dahulu.", parent=self)
            return False
        return True

    def tangga_baru(self) -> None:
        if not self._perlu_pilihan():
            return
        daftar = CR.baca_tangga(self.akar)
        nama = simpledialog.askstring("Tangga baru", f"Nama tangga {CR.id_tangga_baru(daftar)} untuk {len(self.pilih)} "
                                                     "rekaman\n(mis. Gedung F tangga depan; nama yang sudah ada = "
                                                     "digabung ke tangga itu):", parent=self)
        if not nama or not nama.strip():
            return
        lokasi = (simpledialog.askstring("Tangga baru", "Lokasi/keterangan (boleh kosong):", parent=self) or ""
                  if CR.cari_tangga(daftar, nama) is None else "")
        tid, _ = CR.tangga_dari_nama(self.akar, nama, lokasi)
        self._tulis(tid)
        self.pilih.clear()
        self._segarkan()

    def pasang(self) -> None:
        if not self._perlu_pilihan():
            return
        tid = self._tid_terpilih()
        if not tid:
            messagebox.showinfo("Pilih tangga", "Klik tangga tujuan di daftar kanan dahulu "
                                                "(atau pakai Jadikan tangga baru).", parent=self)
            return
        self._tulis(tid)

    def lepas(self) -> None:
        if not self._perlu_pilihan():
            return
        if messagebox.askyesno("Lepas tangga", f"Hapus tangga dari {len(self.pilih)} rekaman terpilih?", parent=self):
            self._tulis("")

    def ganti_nama(self) -> None:
        tid = self._tid_terpilih()
        if not tid:
            messagebox.showinfo("Pilih tangga", "Klik tangga di daftar kanan dahulu.", parent=self)
            return
        daftar = CR.baca_tangga(self.akar)
        nama = simpledialog.askstring("Ganti nama", f"Nama baru {tid}:", initialvalue=daftar[tid]["nama"], parent=self)
        if nama and nama.strip():
            daftar[tid]["nama"] = nama.strip()
            CR.tulis_tangga(self.akar, daftar)
            self._segarkan()
