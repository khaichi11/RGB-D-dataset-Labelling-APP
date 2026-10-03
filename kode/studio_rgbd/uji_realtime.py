"""Tab uji sistem tangga — model akhir, pelacak, dan ukuran dengan mekanisme inferensi deployment.

Tab 1-4 di Studio adalah alur PEMBUATAN dataset (rekam, tinjau, ekspor, label).
Tab ini beda tujuan: menjalankan sistem yang sudah jadi pada kamera langsung
atau rekaman, untuk memeriksa mask, pelacakan, ukuran anak tangga, dan
kestabilan lajunya.

Sejak 3 Oktober 2026 tab ini memakai mekanisme yang SAMA dengan perintah
``tangga`` di ``Train-RGB-D-Model/aplikasi_tangga`` (paket ``rgbd_convnext``):
empat proses paralel (``paralel.jalankan_paralel``), model akhir ConvNeXt V2
Atto RGB-D 384 (FP16 + graf CUDA bila ada GPU), pelacak geometri, dan render
yang sama. Model StairFusion lama (kepala garis, 512) tidak dipakai lagi.

Torch dan ``rgbd_convnext`` baru diimpor saat tombol Mulai ditekan, jadi
Studio tetap bisa dipakai melabeli di mesin tanpa GPU.
"""
from __future__ import annotations
try:
    from .ui_bantu import kolom_gulir
except ImportError:
    from ui_bantu import kolom_gulir

import collections
import json
import math
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np

BG, PANEL, INK, MUTED, ACCENT, LINE = "#F7F3F0", "#FFFFFF", "#2B2622", "#7A6E66", "#C1613C", "#E4DAD3"
GREEN, BLUE, RED = "#3C7A5A", "#3C5F7A", "#A6452F"

AKAR = Path(__file__).resolve().parents[2]
AKAR_APLIKASI = AKAR / "Train-RGB-D-Model" / "aplikasi_tangga"
BOBOT_BAWAAN = AKAR_APLIKASI / "bobot" / "final_d435" / "cnx_atto_in1k_384.pt"
JENDELA_STAT = 90                      # frame terakhir untuk median waktu per tahap


def _impor_tangga():
    """Paket ``rgbd_convnext`` dari Train-RGB-D-Model (atau yang sudah terpasang lewat pip)."""
    try:
        import rgbd_convnext  # noqa: F401
    except ImportError:
        if str(AKAR_APLIKASI) not in sys.path:
            sys.path.insert(0, str(AKAR_APLIKASI))
    from rgbd_convnext.jalankan import ringkas
    from rgbd_convnext.paralel import jalankan_paralel
    return jalankan_paralel, ringkas


def _cm(nilai) -> str:
    return f"{nilai * 100:5.1f} cm" if nilai is not None and math.isfinite(nilai) else "    -   "


class PenghitungTerlewat:
    """Frame kamera yang terlewat, dari celah cap waktu kamera (> 1,5x interval wajar)."""

    def __init__(self) -> None:
        self.selisih = collections.deque(maxlen=60)
        self.lalu = None
        self.jumlah = 0

    def tambah(self, cap: float) -> None:
        if self.lalu is not None and cap > self.lalu:
            d = cap - self.lalu
            if len(self.selisih) >= 5:
                interval = float(np.median(self.selisih))
                if d > 1.5 * interval:
                    self.jumlah += int(round(d / interval)) - 1
                    d = None
            if d is not None:
                self.selisih.append(d)
        self.lalu = cap


class UjiRealtime:
    """Panel uji sistem tangga. Dipasang ke sebuah frame Tk milik Studio."""

    def __init__(self, induk: tk.Frame, studio) -> None:
        self.induk, self.studio = induk, studio
        self.jalan = False
        self.thread: threading.Thread | None = None
        self.foto = None
        self._kunci = threading.Lock()
        # Thread pekerja TIDAK menyentuh Tk (StringVar dsb. hanya aman di thread utama);
        # hasilnya diletakkan di atribut biasa ini dan diterapkan oleh _segarkan().
        self._terbaru = None               # (tampilan BGR, teks laju, teks tahap, teks ukuran)
        self._galat = None
        self._status_baru = None
        self._simpanan = None

        self.bobot = tk.StringVar(value=str(BOBOT_BAWAAN) if BOBOT_BAWAAN.exists() else "")
        self.presisi = tk.StringVar(value="fp16")
        self.sumber = tk.StringVar(value="rekaman")
        self.rekaman = tk.StringVar(value="")
        self.waktu_nyata = tk.BooleanVar(value=True)
        self.simpan = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Pilih sumber lalu tekan Mulai.")
        self.fps = tk.StringVar(value="-")
        self.statistik = tk.StringVar(value="-")
        self.ukuran = tk.StringVar(value="-")
        self._bangun()

    # ------------------------------------------------------------------ UI

    def _kartu(self, induk, judul):
        luar = tk.Frame(induk, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        tk.Label(luar, text=judul, bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(10, 4))
        dalam = tk.Frame(luar, bg=PANEL)
        dalam.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        return luar, dalam

    def _tombol(self, induk, teks, perintah, warna=ACCENT, fg="#FFFFFF"):
        return tk.Button(induk, text=teks, command=perintah, bg=warna, fg=fg,
                         relief="flat", font=("Segoe UI", 10, "bold"),
                         activebackground=warna, cursor="hand2", pady=7)

    def _radio(self, induk, teks, var, nilai):
        tk.Radiobutton(induk, text=teks, variable=var, value=nilai, bg=PANEL, fg=INK, selectcolor=PANEL,
                       activebackground=PANEL, anchor="w").pack(fill="x")

    def _centang(self, induk, teks, var):
        tk.Checkbutton(induk, text=teks, variable=var, bg=PANEL, fg=INK, selectcolor=PANEL,
                       activebackground=PANEL, anchor="w", wraplength=310, justify="left").pack(fill="x", pady=(4, 0))

    def _bangun(self) -> None:
        f = tk.Frame(self.induk, bg=BG)
        f.pack(fill="both", expand=True, padx=24, pady=24)
        kiri = tk.Frame(f, bg=BG)
        kiri.pack(side="left", fill="both", expand=True, padx=(0, 12))
        kanan = kolom_gulir(f, 370, BG)          # bisa di-scroll: kontrol tidak hilang di layar pendek

        b, i = self._kartu(kiri, "Tampilan (mekanisme perintah tangga)")
        b.pack(fill="both", expand=True)
        self.kanvas = tk.Label(i, bg="#111111")
        self.kanvas.pack(fill="both", expand=True)

        b, i = self._kartu(kanan, "Model")
        b.pack(fill="x")
        tk.Entry(i, textvariable=self.bobot, bg="#FAF7F5", fg=INK,
                 relief="flat", highlightbackground=LINE, highlightthickness=1).pack(fill="x", pady=(0, 6))
        self._tombol(i, "Pilih berkas bobot (.pt)", self.pilih_bobot, "#E8DDD5", INK).pack(fill="x", pady=2)
        self._radio(i, "FP16 + graf CUDA (bawaan deployment)", self.presisi, "fp16")
        self._radio(i, "FP32 tanpa graf (pembanding)", self.presisi, "fp32")

        b, i = self._kartu(kanan, "Sumber")
        b.pack(fill="x", pady=(10, 0))
        self._radio(i, "Rekaman (raw.db3 / raw.bag)", self.sumber, "rekaman")
        self._radio(i, "Kamera D435 langsung", self.sumber, "kamera")
        tk.Entry(i, textvariable=self.rekaman, bg="#FAF7F5", fg=INK,
                 relief="flat", highlightbackground=LINE, highlightthickness=1).pack(fill="x", pady=(6, 4))
        baris = tk.Frame(i, bg=PANEL)
        baris.pack(fill="x")
        self._tombol(baris, "Sesi aktif", self.pakai_sesi_aktif, "#E8DDD5", INK).pack(
            side="left", fill="x", expand=True, padx=(0, 2))
        self._tombol(baris, "Pilih rekaman…", self.pilih_rekaman, "#E8DDD5", INK).pack(
            side="left", fill="x", expand=True, padx=(2, 0))
        self._centang(i, "Putar seperti kamera (laju asli rekaman, frame yang tertinggal dibuang)",
                      self.waktu_nyata)

        b, i = self._kartu(kanan, "Kendali")
        b.pack(fill="x", pady=(10, 0))
        self.btn_mulai = self._tombol(i, "▶  Mulai", self.toggle, GREEN)
        self.btn_mulai.pack(fill="x", pady=2)
        self._centang(i, "Simpan video + JSON ukuran ke aplikasi_tangga/keluaran/", self.simpan)

        b, i = self._kartu(kanan, "Laju, latensi, waktu per tahap")
        b.pack(fill="x", pady=(10, 0))
        tk.Label(i, textvariable=self.fps, bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold"),
                 justify="left", anchor="w").pack(fill="x")
        tk.Label(i, textvariable=self.statistik, bg=PANEL, fg=INK, font=("Consolas", 9),
                 justify="left", anchor="w").pack(fill="x", pady=(4, 0))

        b, i = self._kartu(kanan, "Ukuran anak tangga")
        b.pack(fill="x", pady=(10, 0))
        tk.Label(i, textvariable=self.ukuran, bg=PANEL, fg=GREEN, font=("Consolas", 9),
                 justify="left", anchor="w").pack(fill="x")
        tk.Label(i, textvariable=self.status, bg=PANEL, fg=MUTED, wraplength=310,
                 justify="left", anchor="w").pack(fill="x", pady=(8, 0))

        b, i = self._kartu(kanan, "Catatan")
        b.pack(fill="x", pady=(10, 0))
        tk.Label(i, text="Tab ini TIDAK merekam apa pun. Pipeline-nya sama dengan perintah "
                         "`tangga`; hanya gambarnya dirender di proses Studio agar tampil di sini, "
                         "jadi `tangga --tampil` sedikit lebih cepat. Angka cm hanya sah bila "
                         "depth cukup rapat.",
                 bg=PANEL, fg=MUTED, wraplength=310, justify="left").pack(fill="x")

    # -------------------------------------------------------------- aksi

    def pilih_bobot(self) -> None:
        p = filedialog.askopenfilename(title="Pilih bobot model",
                                       filetypes=[("PyTorch", "*.pt"), ("Semua", "*.*")],
                                       initialdir=str(BOBOT_BAWAAN.parent if BOBOT_BAWAAN.exists() else AKAR))
        if p:
            self.bobot.set(p)

    def pakai_sesi_aktif(self) -> None:
        sesi = getattr(self.studio, "sesi", None)
        if not sesi:
            self.status.set("Belum ada sesi aktif; pilih sesi di tab Tinjau atau pilih berkas rekaman.")
            return
        bag = Path(self.studio.bag(Path(sesi)))
        if not bag.exists():
            self.status.set(f"Sesi {Path(sesi).name} tidak punya rekaman mentah.")
            return
        self.rekaman.set(str(bag))
        self.sumber.set("rekaman")

    def pilih_rekaman(self) -> None:
        awal = getattr(self.studio, "sesi", None)
        p = filedialog.askopenfilename(title="Pilih rekaman RealSense",
                                       filetypes=[("RealSense", "*.db3 *.bag"), ("Semua", "*.*")],
                                       initialdir=str(Path(awal) / "source") if awal else str(AKAR))
        if p:
            self.rekaman.set(p)
            self.sumber.set("rekaman")

    def toggle(self) -> None:
        if self.jalan:
            self.berhenti()
        else:
            self.mulai()

    def mulai(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.status.set("Masih menutup jalannya yang sebelumnya; coba lagi sebentar.")
            return
        bobot = Path(self.bobot.get())
        if not bobot.exists():
            self.status.set("Berkas bobot tidak ditemukan.")
            return
        if self.sumber.get() == "rekaman":
            if not self.rekaman.get():
                self.pakai_sesi_aktif()
            if not self.rekaman.get() or not Path(self.rekaman.get()).exists():
                self.status.set("Pilih rekaman dulu (Sesi aktif / Pilih rekaman…).")
                return
        else:
            if getattr(self.studio, "sedang_rekam", False):
                self.status.set("Sedang merekam dataset. Hentikan dulu agar kamera tidak berebut.")
                return
            # Studio memakai kamera untuk preview/rekam; keduanya tidak boleh
            # membuka device yang sama bersamaan.
            try:
                if getattr(self.studio, "cam", None) is not None and self.studio.cam.hidup:
                    self.studio.preview_diminta = False
                    self.studio._hentikan_render()
                    self.studio.cam.hentikan()
            except Exception:
                pass
        atur = {"bobot": bobot, "presisi": self.presisi.get(), "sumber": self.sumber.get(),
                "rekaman": self.rekaman.get(), "waktu_nyata": bool(self.waktu_nyata.get()),
                "simpan": bool(self.simpan.get())}
        self.jalan = True
        self._terbaru, self._galat, self._status_baru = None, None, None
        self.fps.set("memuat model dan memanaskan graf CUDA…")
        self.statistik.set("-")
        self.ukuran.set("-")
        self.status.set("Memulai…")
        self.btn_mulai.configure(text="■  Hentikan", bg=RED)
        self.thread = threading.Thread(target=self._loop, args=(atur,), daemon=True)
        self.thread.start()
        self.induk.after(50, self._segarkan)

    def berhenti(self) -> None:
        self.jalan = False
        try:
            self.btn_mulai.configure(text="▶  Mulai", bg=GREEN)
        except Exception:
            pass

    # -------------------------------------------------------------- pekerja

    @staticmethod
    def _spek(atur: dict):
        import torch
        gpu = torch.cuda.is_available()
        if atur["presisi"] == "fp16" and gpu:
            mesin = {"bobot": str(atur["bobot"]), "fp16": True, "graf": True}
        else:
            mesin = {"bobot": str(atur["bobot"])}
        if atur["sumber"] == "kamera":
            sumber = {"jenis": "kamera", "lewati": 30, "maks": 0}
        else:
            sumber = {"jenis": "rekaman", "jalur": atur["rekaman"], "lewati": 20, "maks": 0,
                      "waktu_nyata": atur["waktu_nyata"]}
        return sumber, mesin, gpu

    def _loop(self, atur: dict) -> None:
        penulis = berkas_json = None
        try:
            jalankan_paralel, ringkas = _impor_tangga()
            from rgbd_convnext.paralel import sumber_langsung
            from rgbd_convnext.render import PenulisVideo
            sumber, mesin, gpu = self._spek(atur)
            nama = Path(atur["rekaman"]).parent.parent.name if sumber["jenis"] == "rekaman" else "kamera"
            if atur["simpan"]:
                keluar = AKAR_APLIKASI / "keluaran"
                keluar.mkdir(parents=True, exist_ok=True)
                dasar = keluar / f"studio_uji_{nama}_{time.strftime('%Y%m%d_%H%M%S')}"
                # Video ditulis di thread sendiri: disk yang tersendat tidak menahan pengukuran.
                penulis = PenulisVideo(dasar.with_suffix(".mp4"), 30.0, buang=sumber_langsung(sumber))
                berkas_json = open(dasar.with_suffix(".jsonl"), "w", encoding="utf-8")
                self._simpanan = dasar.with_suffix(".mp4")
            mesin_teks = "GPU FP16 + graf CUDA" if mesin.get("graf") else ("GPU FP32" if gpu else "CPU FP32")
            self._status_baru = (f"Berjalan: {mesin_teks}, sumber {nama}"
                                 f"{' (seperti kamera)' if sumber.get('waktu_nyata') else ' (secepatnya)'}.")
            tiba = collections.deque(maxlen=600)
            ms = collections.defaultdict(lambda: collections.deque(maxlen=JENDELA_STAT))
            terlewat, n = PenghitungTerlewat(), 0
            for h in jalankan_paralel(sumber, mesin, render=True, judul=f"Studio: {nama}"):
                if not self.jalan:
                    break
                sekarang = time.perf_counter()
                n += 1
                tiba.append(sekarang)
                terlewat.tambah(h.waktu)
                for k, v in h.waktu_ms.items():
                    ms[k].append(v)
                if berkas_json is not None:
                    berkas_json.write(json.dumps(ringkas(h), ensure_ascii=False) + "\n")
                    penulis.tulis(h.tampilan)
                satu_detik = sum(1 for t in tiba if t > sekarang - 1.0)
                lat = np.asarray(ms["latensi"])
                teks_fps = (f"{satu_detik} frame/detik · frame {n} · terlewat {terlewat.jumlah}\n"
                            f"latensi median {np.median(lat):.1f} ms · maks {lat.max():.1f} ms")
                tahap = "\n".join(f"{k:<13s} {np.median(v):6.2f} ms" for k, v in ms.items() if k != "latensi")
                with self._kunci:
                    self._terbaru = (h.tampilan, teks_fps, tahap, self._teks_ukuran(h))
        except Exception as e:                                   # galat pipeline ditampilkan, Studio tetap hidup
            baris = [b for b in str(e).strip().splitlines() if b.strip()]
            self._galat = f"{type(e).__name__}: {baris[-1][:300] if baris else ''}"
        finally:
            try:
                if penulis is not None:
                    penulis.tutup()
            except Exception as e:                               # noqa: BLE001
                self._galat = self._galat or f"video: {e}"
            if berkas_json is not None:
                berkas_json.close()
            self.jalan = False

    @staticmethod
    def _teks_ukuran(h) -> str:
        baris = []
        for s in h.langkah:
            jarak = f"{s.jarak_m:4.2f} m" if s.jarak_pasti and math.isfinite(s.jarak_m) else "   -  "
            baris.append(f"#{str(s.nomor or '-'):<2} riser {_cm(s.tinggi_riser_m)} tread {_cm(s.panjang_tread_m)}"
                         f" {jarak} {s.status}")
        for b in h.batas:
            baris.append(f"{'lantai' if b.arah == 'bawah' else 'bordes'} {b.jarak_m:4.2f} m")
        if h.bordes_lewat is not None:
            baris.append("tangga selesai")
        return "\n".join(baris) if baris else "belum ada anak tangga terlacak"

    # -------------------------------------------------------------- tampilan (thread Tk)

    def _segarkan(self) -> None:
        with self._kunci:
            terbaru, self._terbaru = self._terbaru, None
        if terbaru is not None:
            tampilan, teks_fps, tahap, ukuran = terbaru
            self.fps.set(teks_fps)
            self.statistik.set(tahap)
            self.ukuran.set(ukuran)
            self._tampilkan(tampilan)
        if self._status_baru:
            self.status.set(self._status_baru)
            self._status_baru = None
        if self._galat:
            self.status.set(f"Berhenti karena galat: {self._galat}")
            self._galat = None
        if self.jalan or (self.thread is not None and self.thread.is_alive()):
            self.induk.after(33, self._segarkan)                 # tampilan paling banyak ~30 kali/detik
            return
        self.berhenti()
        if not self.status.get().startswith("Berhenti karena galat"):
            simpan = self._simpanan
            self.status.set(f"Selesai.{f' Video dan JSON: {simpan}' if simpan else ''}")
        self._simpanan = None

    def _tampilkan(self, bgr: np.ndarray) -> None:
        try:
            from PIL import Image, ImageTk
        except Exception:
            return
        lebar = max(320, self.kanvas.winfo_width())
        tinggi = max(240, self.kanvas.winfo_height())
        skala = min(1.0, lebar / bgr.shape[1], tinggi / bgr.shape[0])
        kecil = cv2.resize(bgr, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA) if skala < 1 else bgr
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(kecil, cv2.COLOR_BGR2RGB)))
        self.foto = img                       # tahan referensi; tanpa ini gambar dikumpulkan GC
        self.kanvas.configure(image=img)
