"""Pemutar "Putar & isi meteran": mengisi ukuran anak tangga sambil menonton rekamannya.

Pada frame ekspor anak tangga ke-1, 2, 3 sulit dibedakan, sedangkan pada video
urutannya jelas. Pemutar ini memutar ``derived/preview.mp4`` sesi di dalam
Studio. Nomor anak tangga dari pelacak (``derived/ukuran_sistem.json``, hasil
"Hitung nomor dari video") ditempel di atas gambar, dicocokkan per frame lewat
cap waktu kamera di ``derived/frame_index.csv``.

- Mask "Detektor" (bawaan): peta kelas segmentasi model per frame
  (``derived/ukuran_semantik/``) diwarnai dengan fungsi yang sama dengan video
  detektor (``rgbd_convnext.render.tempel``: riser merah, tread hijau), jadi
  tampilannya persis video itu. Mask "Per anak tangga": poligon tiap bidang
  bernomor (``mask`` di ``ukuran_sistem.json``) yang dihaluskan antar-frame.
- Nomor (dan poligon mode per anak tangga) dihaluskan antar-frame: titik
  poligon dipasangkan dari frame ke frame, celah pengamatan <= 3 frame diisi,
  lalu dirata-rata Gauss (sigma 2 frame, terpusat sehingga tidak tertinggal)
  agar tidak bergetar atau berkedip. Penghalusan hanya untuk tampilan; nomor
  "baru terlihat" tetap memakai pengamatan mentah.
- Klik nomor ``#n R`` / ``#n T`` atau mask-nya di video: kotak isian muncul di atas video
  untuk tinggi riser dan panjang tread meteran anak tangga itu.
- Saat sebuah anak tangga pertama kali terlihat dan ukurannya belum lengkap,
  video berhenti sendiri dan menanyakannya. Fitur ini bisa dimatikan.
- Isian disimpan lewat ``ukuran_meteran.tulis`` ke tempat yang sama dengan
  kartu "Ukuran tangga (meteran)": per tangga fisik bila sesi sudah dikaitkan,
  selain itu per sesi.

Pemutar hanya membaca turunan rekaman. RAW dan label tidak disentuh.
"""
from __future__ import annotations

import csv
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from . import catatan_rekaman as CR
    from . import ukuran_meteran as UM
    from .panel_ukuran import WARNA_R, WARNA_T, _cm, _modul_ukuran
except ImportError:
    import catatan_rekaman as CR
    import ukuran_meteran as UM
    from panel_ukuran import WARNA_R, WARNA_T, _cm, _modul_ukuran

BG, PANEL, INK, MUTED, LINE, ACCENT, GREEN = ("#F3EEE7", "#FFFDFC", "#382D29", "#786962", "#DED2C8",
                                             "#754C3B", "#6B8B63")
LATAR_VIDEO = "#111111"
WARNA_PILIH = "#FFE066"         # bingkai nomor anak tangga yang sedang diisi
WARNA_BELUM = "#D9822B"         # penanda pita: ukuran meteran belum lengkap
WARNA_DET = {"R": "#FF9650", "T": "#5AFF8C"}   # warna label riser/tread video detektor (render.WARNA_LABEL)
TAMPILAN = ("RGB", "Depth", "RGB + Depth")
MODE_MASK = ("Detektor", "Per anak tangga", "Mati")
KECEPATAN = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}
MIN_FRAME_TANYA = 5             # anak tangga yang teramati < 5 frame tidak menghentikan video (kemungkinan salah lacak)
WAJAR = {"tinggi_riser_cm": (8.0, 30.0), "panjang_tread_cm": (18.0, 60.0)}
MAKS_CACHE = 40
N_TITIK = 48                    # titik per poligon setelah dicuplik ulang (agar bisa dipasangkan antar-frame)
SIGMA_FRAME = 2.0               # lebar penghalusan Gauss antar-frame
MAKS_CELAH = 3                  # celah pengamatan sependek ini diisi interpolasi (tidak berkedip)
BGR_R, BGR_T = (71, 179, 255), (255, 224, 124)      # WARNA_R / WARNA_T dalam BGR


def _cuplik_ulang(poli, n: int = N_TITIK):
    """Poligon [x0, y0, x1, ...] -> n titik berjarak sama sepanjang keliling, arah putar seragam."""
    p = np.asarray(poli, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return None
    if np.cross(p - p.mean(0), np.roll(p, -1, 0) - p.mean(0)).sum() < 0:
        p = p[::-1]
    tutup = np.vstack([p, p[:1]])
    panjang = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(tutup, axis=0).T))])
    if panjang[-1] <= 0:
        return None
    t = np.linspace(0.0, panjang[-1], n, endpoint=False)
    return np.stack([np.interp(t, panjang, tutup[:, 0]), np.interp(t, panjang, tutup[:, 1])], axis=1)


def _selaraskan(p: np.ndarray, acuan: np.ndarray) -> np.ndarray:
    """Geser urutan titik p (siklik) agar setiap titik berpasangan dengan titik terdekat di acuan."""
    n = len(p)
    idx = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
    return p[idx[int(np.argmin(((p[idx] - acuan[None]) ** 2).sum(axis=(1, 2))))]]


def _gauss(a: np.ndarray, sigma: float = SIGMA_FRAME) -> np.ndarray:
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    pad = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    return sum(k[j] * pad[j:j + len(a)] for j in range(2 * r + 1)) / k.sum()


def _haluskan_deret(deret: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """{frame: nilai} -> nilai halus pada setiap frame, celah <= MAKS_CELAH diisi, celah panjang memutus."""
    frame = sorted(deret)
    hasil: dict[int, np.ndarray] = {}
    awal = 0
    for j in range(1, len(frame) + 1):
        if j < len(frame) and frame[j] - frame[j - 1] <= MAKS_CELAH + 1:
            continue
        potong = frame[awal:j]
        awal = j
        nilai = np.stack([deret[f] for f in potong]).reshape(len(potong), -1)
        semua = np.arange(potong[0], potong[-1] + 1)
        isi = np.stack([np.interp(semua, potong, nilai[:, c]) for c in range(nilai.shape[1])], axis=1)
        halus = _gauss(isi).reshape((len(semua),) + deret[potong[0]].shape)
        hasil.update(zip(semua.tolist(), halus))
    return hasil


def jejak_halus(tanda_i: list[list], mask_i: list, n: int) -> tuple[list[list], list[list]]:
    """Tanda dan mask per frame yang sudah dihaluskan.

    -> (per frame ``[(ke, nomor, 'R'/'T', x, y, poligon Nx2 | None)]``, per frame ``[('R'/'T', poligon)]`` tak bernomor).
    """
    pusat: dict[tuple, dict[int, np.ndarray]] = {}
    for i, tanda in enumerate(tanda_i):
        for ke, nomor, k, x, y in tanda:
            pusat.setdefault((ke, nomor, k), {})[i] = np.array([x, y], dtype=np.float64)
    poli: dict[tuple, dict[int, np.ndarray]] = {}
    lain: list[list] = [[] for _ in range(n)]
    for i, m in enumerate(mask_i):
        if not m:
            continue
        for ke, nomor, k, titik in m.get("n", []):
            c = _cuplik_ulang(titik)
            if c is not None:
                poli.setdefault((ke, nomor, k), {})[i] = c
        lain[i] = [(k, np.asarray(t, dtype=np.float64).reshape(-1, 2)) for k, t in m.get("lain", []) if len(t) >= 6]
    for deret in poli.values():                  # pasangkan titik dengan frame sebelumnya pada jejak yang sama
        acuan = None
        for f in sorted(deret):
            if acuan is not None:
                deret[f] = _selaraskan(deret[f], acuan)
            acuan = deret[f]
    hasil: list[list] = [[] for _ in range(n)]
    for kunci, deret in pusat.items():
        p_halus = _haluskan_deret(poli.get(kunci, {})) if kunci in poli else {}
        for f, xy in _haluskan_deret(deret).items():
            if 0 <= f < n:
                hasil[f].append((*kunci, float(xy[0]), float(xy[1]), p_halus.get(f)))
    return hasil, lain


def _fungsi_tempel():
    """``rgbd_convnext.render.tempel``: pewarnaan mask yang sama dengan video detektor."""
    _modul_ukuran()                              # memasang jalur aplikasi
    from rgbd_convnext.render import tempel
    return tempel


def buka(studio, sesi: Path, mulai: int | None = None) -> "PemutarUkur":
    """Buka pemutar untuk ``sesi``; satu jendela saja, jendela lama dipakai ulang bila sesinya sama."""
    lama = getattr(studio, "_pemutar_ukur", None)
    if lama is not None and lama.winfo_exists():
        if lama.sesi == Path(sesi):
            lama.deiconify(); lama.lift(); lama.focus_force()
            if mulai is not None:
                lama.lompat(mulai)
            return lama
        lama.tutup()
    studio._pemutar_ukur = PemutarUkur(studio, Path(sesi), mulai)
    return studio._pemutar_ukur


def _angka(teks: str):
    """'17,5' -> 17.5; '' -> None; selain itu ValueError."""
    teks = teks.strip().replace(",", ".")
    return float(teks) if teks else None


class PemutarUkur(tk.Toplevel):
    def __init__(self, studio, sesi: Path, mulai: int | None = None) -> None:
        super().__init__(studio)
        self.studio, self.sesi = studio, Path(sesi)
        self.title(f"Putar & isi meteran: {self.sesi.name}")
        self.configure(bg=BG)
        lebar, tinggi = min(1440, self.winfo_screenwidth() - 80), min(840, self.winfo_screenheight() - 120)
        self.geometry(f"{lebar}x{tinggi}")
        self.minsize(900, 560)

        self.cap = None
        self.n, self.fps = 0, 30.0
        self.waktu: list[float] = []
        self._pos_cap = -1                       # indeks yang dikembalikan cap.read() berikutnya
        self._cache: dict[int, np.ndarray] = {}
        self._bgr = None
        self.i = 0
        self._main = False
        self._job = self._job_render = None
        self._t0, self._i0 = 0.0, 0
        self._foto = self._item_gambar = None
        self._skala, self._ox, self._oy, self._panel_x = 1.0, 0, 0, [0]

        self.sistem = None
        self.tanda_i: list[list] = []            # per indeks frame: [[ke, nomor, 'R'/'T', x, y], ...] (mentah)
        self.jejak_i: list[list] = []            # per indeks frame, halus: [(ke, nomor, 'R'/'T', x, y, poligon)]
        self.lain_i: list[list] = []             # per indeks frame: mask riser/tread tak bernomor
        self._ada_mask = False
        self.sem_i: list = []                    # per indeks frame: nomor berkas peta kelas, atau None
        self._cache_sem: dict[int, np.ndarray] = {}
        self._tempel = None
        self.muncul: dict[tuple, int] = {}       # (ke, nomor) -> frame pertama terlihat
        self.n_teramati: dict[tuple, int] = {}
        self.info_sistem: dict[tuple, dict] = {}
        self.tid: dict[int, str | None] = {}
        self.met: dict[int, dict] = {}           # ke -> anak tangga meteran {nomor fisik: {...}}
        self.lebar_cm: dict[int, object] = {}
        self.geser: dict[int, int] = {}          # ke -> nomor fisik - nomor sistem
        self.pilih: tuple | None = None
        self._antrean: list[tuple] = []
        self._dilihat: set = set()
        self._lanjut_setelah = False
        self._thread = None
        self._progres, self._galat, self._tutup = 0, None, False

        self.tampilan = tk.StringVar(value="RGB")
        self.mode_mask = tk.StringVar(value=MODE_MASK[0])
        self.kecepatan = tk.StringVar(value="1x")
        self.berhenti_baru = tk.BooleanVar(value=True)
        self.posisi = tk.IntVar(value=0)
        self.waktu_teks = tk.StringVar(value="")
        self.info = tk.StringVar(value="")
        self.status = tk.StringVar(value="")
        self.geser_var = tk.IntVar(value=1)
        self.riser = tk.StringVar()
        self.tread = tk.StringVar()

        self._bangun_ui()
        self.protocol("WM_DELETE_WINDOW", self.tutup)
        for urutan, fungsi in (("<space>", lambda e: self.putar_jeda()),
                               ("<Left>", lambda e: self.langkah(-1)), ("<Right>", lambda e: self.langkah(1)),
                               ("<Shift-Left>", lambda e: self.langkah(-10)),
                               ("<Shift-Right>", lambda e: self.langkah(10)),
                               ("<Escape>", lambda e: self.lewati()), ("<Control-w>", lambda e: self.tutup())):
            self.bind(urutan, lambda e, f=fungsi: None if self._sedang_mengetik(e) else f(e))
        self._muat(mulai)

    # ------------------------------------------------------------ tata letak
    def _tombol(self, induk, teks, perintah, warna="#E8DDD5", fg=INK, lebar=88):
        if hasattr(self.studio, "tombol_ringkas"):
            return self.studio.tombol_ringkas(induk, teks, perintah, warna, fg, width=lebar)
        return tk.Button(induk, text=teks, command=perintah, bg=warna, fg=fg, relief="flat")

    def _bangun_ui(self) -> None:
        kanan = tk.Frame(self, bg=PANEL, width=340, highlightthickness=1, highlightbackground=LINE)
        kanan.pack(side="right", fill="y", padx=(0, 10), pady=10)
        kanan.pack_propagate(False)
        kiri = tk.Frame(self, bg=BG)
        kiri.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.wadah = tk.Frame(kiri, bg=LATAR_VIDEO)
        self.wadah.pack(fill="both", expand=True)
        self.kanvas = tk.Canvas(self.wadah, bg=LATAR_VIDEO, highlightthickness=0, cursor="hand2")
        self.kanvas.pack(fill="both", expand=True)
        self.kanvas.bind("<Configure>", lambda e: self._render_nanti())
        self.kanvas.bind("<Button-1>", self._klik_kanvas)
        self._teks_pesan = self.kanvas.create_text(16, 16, anchor="nw", fill="white", font=("Segoe UI", 11),
                                                   text="Memuat…", width=700)

        # Kotak tanya di atas video (seperti pilihan pada film interaktif).
        self.kotak = tk.Frame(self.wadah, bg=PANEL, highlightthickness=2, highlightbackground=ACCENT, padx=14, pady=10)
        self.kotak_judul = tk.Label(self.kotak, text="", bg=PANEL, fg=INK, font=("Segoe UI", 13, "bold"))
        self.kotak_judul.grid(row=0, column=0, columnspan=6, sticky="w")
        self.kotak_sub = tk.Label(self.kotak, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 9))
        self.kotak_sub.grid(row=1, column=0, columnspan=6, sticky="w", pady=(0, 6))
        tk.Label(self.kotak, text="Tinggi riser", bg=PANEL, fg=INK, font=("Segoe UI", 10)).grid(row=2, column=0, sticky="w")
        self.isi_riser = tk.Entry(self.kotak, textvariable=self.riser, width=7, font=("Segoe UI", 14), justify="right")
        self.isi_riser.grid(row=2, column=1, padx=(6, 2))
        tk.Label(self.kotak, text="cm", bg=PANEL, fg=MUTED).grid(row=2, column=2, sticky="w", padx=(0, 16))
        tk.Label(self.kotak, text="Panjang tread", bg=PANEL, fg=INK, font=("Segoe UI", 10)).grid(row=2, column=3, sticky="w")
        self.isi_tread = tk.Entry(self.kotak, textvariable=self.tread, width=7, font=("Segoe UI", 14), justify="right")
        self.isi_tread.grid(row=2, column=4, padx=(6, 2))
        tk.Label(self.kotak, text="cm", bg=PANEL, fg=MUTED).grid(row=2, column=5, sticky="w")
        aksi = tk.Frame(self.kotak, bg=PANEL); aksi.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(10, 0))
        self.btn_simpan = self._tombol(aksi, "Simpan", self.simpan, ACCENT, "white", lebar=190)
        self.btn_simpan.pack(side="left")
        self._tombol(aksi, "Lewati (Esc)", self.lewati, lebar=110).pack(side="left", padx=(6, 0))
        self.kotak_status = tk.Label(self.kotak, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 8),
                                     wraplength=440, justify="left")
        self.kotak_status.grid(row=4, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self.isi_riser.bind("<Return>", lambda e: (self.isi_tread.focus_set(), self.isi_tread.select_range(0, "end")))
        self.isi_tread.bind("<Return>", lambda e: self.simpan())
        for isian in (self.isi_riser, self.isi_tread):
            isian.bind("<Escape>", lambda e: (self.lewati(), "break")[1])

        bawah = tk.Frame(kiri, bg=BG); bawah.pack(fill="x", pady=(6, 0))
        self.skala = tk.Scale(bawah, from_=0, to=0, orient="horizontal", variable=self.posisi, showvalue=0,
                              sliderlength=18, bg=BG, highlightthickness=0, command=self._geser_slider)
        self.skala.pack(fill="x")
        self.pita = tk.Canvas(bawah, height=26, bg=BG, highlightthickness=0, cursor="hand2")
        self.pita.pack(fill="x")
        self.pita.bind("<Configure>", lambda e: self._gambar_pita())
        self.pita.bind("<Button-1>", self._klik_pita)
        baris = tk.Frame(bawah, bg=BG); baris.pack(fill="x", pady=(4, 0))
        self._tombol(baris, "⏮", lambda: self.langkah(-1), lebar=36).pack(side="left")
        self.btn_putar = self._tombol(baris, "▶ Putar", self.putar_jeda, ACCENT, "white", lebar=92)
        self.btn_putar.pack(side="left", padx=4)
        self._tombol(baris, "⏭", lambda: self.langkah(1), lebar=36).pack(side="left")
        tk.Label(baris, textvariable=self.waktu_teks, bg=BG, fg=MUTED, font=("Consolas", 9)).pack(side="left", padx=10)
        tk.Checkbutton(baris, text="Berhenti di anak tangga baru", variable=self.berhenti_baru, bg=BG, fg=INK,
                       selectcolor=PANEL, activebackground=BG).pack(side="left", padx=(6, 0))
        kec = ttk.Combobox(baris, textvariable=self.kecepatan, width=5, state="readonly", values=tuple(KECEPATAN))
        kec.pack(side="right")
        kec.bind("<<ComboboxSelected>>", lambda e: self._mulai_jam())
        tk.Label(baris, text="Kecepatan", bg=BG, fg=MUTED).pack(side="right", padx=(10, 4))
        tam = ttk.Combobox(baris, textvariable=self.tampilan, width=11, state="readonly", values=TAMPILAN)
        tam.pack(side="right")
        tam.bind("<<ComboboxSelected>>", lambda e: self._render())
        tk.Label(baris, text="Tampilan", bg=BG, fg=MUTED).pack(side="right", padx=(10, 4))
        msk = ttk.Combobox(baris, textvariable=self.mode_mask, width=14, state="readonly", values=MODE_MASK)
        msk.pack(side="right")
        msk.bind("<<ComboboxSelected>>", lambda e: self._render())
        tk.Label(baris, text="Mask", bg=BG, fg=MUTED).pack(side="right", padx=(10, 4))

        isi = tk.Frame(kanan, bg=PANEL); isi.pack(fill="both", expand=True, padx=12, pady=10)
        tk.Label(isi, text=self.sesi.name, bg=PANEL, fg=INK, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(isi, textvariable=self.info, bg=PANEL, fg=MUTED, wraplength=312, justify="left",
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 6))
        g = tk.Frame(isi, bg=PANEL); g.pack(fill="x")
        self.lbl_geser = tk.Label(g, text="Nomor sistem #1 = anak tangga fisik ke-", bg=PANEL, fg=INK,
                                  font=("Segoe UI", 8))
        self.lbl_geser.pack(side="left")
        self.spin_geser = tk.Spinbox(g, from_=1, to=30, width=3, textvariable=self.geser_var,
                                     command=self._ubah_geser, font=("Segoe UI", 9))
        self.spin_geser.pack(side="left")
        self.spin_geser.bind("<Return>", lambda e: self._ubah_geser())
        self.spin_geser.bind("<FocusOut>", lambda e: self._ubah_geser())
        tk.Label(isi, text="Isi bila rekaman dimulai di atas anak tangga pertama. Nomor di video ikut berubah.",
                 bg=PANEL, fg=MUTED, wraplength=312, justify="left", font=("Segoe UI", 8)).pack(anchor="w")
        tk.Frame(isi, bg=LINE, height=1).pack(fill="x", pady=8)
        tk.Label(isi, text="Anak tangga (klik untuk melompat)", bg=PANEL, fg=INK,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.tabel = tk.Frame(isi, bg=PANEL); self.tabel.pack(fill="x", pady=(4, 0))
        self.btn_hitung = self._tombol(isi, "Hitung nomor dari video", self.hitung, "#3C5F7A", "white", lebar=312)
        self.btn_hitung.pack(fill="x", pady=(10, 0))
        tk.Label(isi, textvariable=self.status, bg=PANEL, fg=MUTED, wraplength=312, justify="left",
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 0))
        tk.Label(isi, text="Spasi putar/jeda • klik video putar/jeda • klik nomor = isi ukurannya • "
                           "←/→ 1 frame • Shift+←/→ 10 frame • Enter simpan • Esc lewati",
                 bg=PANEL, fg=MUTED, wraplength=312, justify="left", font=("Segoe UI", 8)).pack(side="bottom", anchor="w")

    # ------------------------------------------------------------ data
    def _muat(self, mulai: int | None) -> None:
        derived = self.sesi / "derived"
        preview, indeks = derived / "preview.mp4", derived / "frame_index.csv"
        if not preview.exists() or not indeks.exists():
            self._pesan("Preview video sesi ini belum ada.\nBuat dulu di tab Tinjau: pilih rekaman, lalu tekan "
                        "Buat preview.")
            return
        cap = cv2.VideoCapture(str(preview))
        if not cap.isOpened():
            cap.release()
            self._pesan("preview.mp4 tidak dapat dibuka.")
            return
        try:
            with indeks.open(newline="", encoding="utf-8") as f:
                self.waktu = [float(r["timestamp_ms"]) for r in csv.DictReader(f)]
        except (OSError, ValueError, KeyError) as e:
            cap.release()
            self._pesan(f"frame_index.csv tidak terbaca: {e}")
            return
        self.cap = cap
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.n = min(n_video, len(self.waktu)) if n_video > 0 else len(self.waktu)
        if len(self.waktu) > 1:
            jeda = float(np.median(np.diff(self.waktu)))
            self.fps = 1000.0 / jeda if jeda > 0 else 30.0
        self.skala.configure(to=max(0, self.n - 1))
        self._muat_sistem()
        self.i = int(np.clip(mulai if mulai is not None else 0, 0, max(0, self.n - 1)))
        bgr = self._baca(self.i)
        if bgr is not None:                      # malam: RGB gelap, depth lebih jelas
            self.tampilan.set("Depth" if float(bgr[:, : bgr.shape[1] // 2].mean()) < 30 else "RGB")
        self._tampilkan(self.i)

    def _muat_sistem(self) -> None:
        uk = _modul_ukuran()
        self.sistem = uk.baca(self.sesi)
        if self.sistem is not None:
            self.tanda_i = [uk.tanda_frame(self.sistem, w) for w in self.waktu[: self.n]]
        else:
            self.tanda_i = [[] for _ in range(self.n)]
        mask_i = self._per_frame((self.sistem or {}).get("mask") or {})
        self.jejak_i, self.lain_i = jejak_halus(self.tanda_i, mask_i, self.n)
        self._ada_mask = any(mask_i)
        self.sem_i = self._per_frame((self.sistem or {}).get("semantik") or {})
        self._cache_sem = {}
        self.muncul, self.n_teramati = {}, {}
        for i, tanda in enumerate(self.tanda_i):
            for s in {(ke, nomor) for ke, nomor, *_ in tanda}:
                self.muncul.setdefault(s, i)
                self.n_teramati[s] = self.n_teramati.get(s, 0) + 1
        self.info_sistem = {(t["ke"], a["nomor"]): a for t in (self.sistem or {}).get("tangga", [])
                            for a in t["anak_tangga"]}
        catatan = CR.baca(self.sesi)
        self.tid = {}
        for (ke, _), i in sorted(self.muncul.items(), key=lambda kv: kv[1]):
            self.tid.setdefault(ke, CR.tangga_untuk(catatan, i))
        self._muat_meteran()
        if self.sistem is None:
            self.status.set("Nomor anak tangga belum dihitung untuk sesi ini. Tekan Hitung nomor dari video "
                            "(pelacak berjalan di setiap frame; beberapa menit).")
        self.btn_hitung.configure(text="Hitung ulang nomor dari video" if self.sistem else "Hitung nomor dari video")

    def _per_frame(self, data: dict) -> list:
        """{cap waktu ms: isi} -> isi per indeks frame preview (cap waktu terdekat, toleransi 5 ms)."""
        hasil: list = [None] * self.n
        if not data or not self.n:
            return hasil
        waktu = np.asarray(self.waktu[: self.n])
        for kunci, isi in data.items():
            t = float(kunci)
            j = int(np.clip(np.searchsorted(waktu, t), 1, len(waktu) - 1)) if len(waktu) > 1 else 0
            j = min((j - 1, j), key=lambda q: abs(waktu[q] - t)) if len(waktu) > 1 else 0
            if abs(waktu[j] - t) <= 5.0:
                hasil[j] = isi
        return hasil

    def _daftar_ke(self) -> list[int]:
        return sorted({ke for ke, _ in self.info_sistem} | {ke for ke, _ in self.muncul}) or [1]

    def _muat_meteran(self) -> None:
        akar = self.studio.root_data
        self.met, self.lebar_cm, self.geser = {}, {}, {}
        for ke in self._daftar_ke():
            m = UM.baca(self.sesi, akar, self.tid.get(ke), ke)
            self.met[ke], self.lebar_cm[ke] = m["anak_tangga"], m["lebar_cm"]
            self.geser.update(m["geser"])
        daftar = CR.baca_tangga(akar)
        tempat = sorted({CR.label_tangga(t, daftar) for t in self.tid.values() if t})
        self.info.set((f"Disimpan untuk tangga fisik {', '.join(tempat)} (dipakai semua rekaman tangga ini)."
                       if tempat else "Disimpan untuk sesi ini saja (kaitkan sesi ke tangga fisik di tab Tinjau "
                                      "agar berlaku untuk semua rekamannya).")
                      + (f"\n{len(self._daftar_ke())} penomoran: nomor mulai dari 1 lagi sesudah bordes."
                         if len(self._daftar_ke()) > 1 else ""))
        self._segarkan_geser()
        self._bangun_tabel()
        self._gambar_pita()

    def fisik(self, ke: int, nomor: int) -> int:
        return nomor + self.geser.get(ke, 0)

    def meteran(self, ke: int, nomor: int) -> dict:
        return self.met.get(ke, {}).get(self.fisik(ke, nomor), {})

    def lengkap(self, ke: int, nomor: int) -> bool:
        m = self.meteran(ke, nomor)
        return m.get("tinggi_riser_cm") is not None and m.get("panjang_tread_cm") is not None

    def nama(self, ke: int, nomor: int) -> str:
        return (f"T{ke}·" if len(self._daftar_ke()) > 1 else "") + f"#{self.fisik(ke, nomor)}"

    def _ke_aktif(self) -> int:
        return self.pilih[0] if self.pilih else self._daftar_ke()[0]

    # ------------------------------------------------------------ video
    def _pesan(self, teks: str) -> None:
        self.kanvas.itemconfigure(self._teks_pesan, text=teks)
        self.kanvas.tag_raise(self._teks_pesan)

    def _baca(self, i: int):
        if i in self._cache:
            self._cache[i] = self._cache.pop(i)
            return self._cache[i]
        if self.cap is None:
            return None
        if self._pos_cap <= i <= self._pos_cap + 6:     # dekat di depan: maju berurutan, lebih murah dari seek
            while self._pos_cap < i:
                self.cap.grab(); self._pos_cap += 1
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = self.cap.read()
        self._pos_cap = i + 1 if ok else -1
        if not ok:
            return None
        self._cache[i] = bgr
        while len(self._cache) > MAKS_CACHE:
            self._cache.pop(next(iter(self._cache)))
        return bgr

    def _tampilkan(self, i: int) -> None:
        if self.cap is None:
            return
        self.i = int(np.clip(i, 0, max(0, self.n - 1)))
        bgr = self._baca(self.i)
        if bgr is not None:
            self._bgr = bgr
        if self.posisi.get() != self.i:
            self.posisi.set(self.i)
        detik = (self.waktu[self.i] - self.waktu[0]) / 1000.0 if self.waktu else 0.0
        self.waktu_teks.set(f"frame {self.i}/{max(0, self.n - 1)} · {int(detik // 60):d}:{detik % 60:04.1f}")
        self._render()

    def _render_nanti(self) -> None:
        if self._job_render is not None:
            self.after_cancel(self._job_render)
        self._job_render = self.after(40, self._render)

    def _render(self) -> None:
        self._job_render = None
        if self._bgr is None:
            return
        c = self.kanvas
        cw, ch = max(c.winfo_width(), 2), max(c.winfo_height(), 2)
        wp = self._bgr.shape[1] // 2
        mode, mask = self.tampilan.get(), self.mode_mask.get()
        bgr = self._bgr
        sem = self._baca_sem(self.i) if mask == "Detektor" else None
        if sem is not None and sem.shape == (bgr.shape[0], wp):
            if self._tempel is None:
                self._tempel = _fungsi_tempel()
            kiri = self._tempel(np.ascontiguousarray(bgr[:, :wp]), sem) if mode != "Depth" else bgr[:, :wp]
            kanan = self._tempel(np.ascontiguousarray(bgr[:, wp:]), sem) if mode != "RGB" else bgr[:, wp:]
            bgr = np.hstack([kiri, kanan])
        if mode == "RGB":
            img, self._panel_x = bgr[:, :wp], [0]
        elif mode == "Depth":
            img, self._panel_x = bgr[:, wp:], [0]
        else:
            img, self._panel_x = bgr, [0, wp]
        h, w = img.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        kecil = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if mask == "Per anak tangga" and self.i < len(self.jejak_i):
            kecil = self._tempel_mask(kecil, s)
        elif mask == "Detektor":
            kecil = self._garis_pilihan(kecil, s)
        self._foto = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(kecil, cv2.COLOR_BGR2RGB)))
        self._skala, self._ox, self._oy = s, (cw - nw) // 2, (ch - nh) // 2
        if self._item_gambar is None:
            self._item_gambar = c.create_image(self._ox, self._oy, anchor="nw", image=self._foto, tags=("gambar",))
            c.tag_lower(self._item_gambar)
        else:
            c.coords(self._item_gambar, self._ox, self._oy)
            c.itemconfigure(self._item_gambar, image=self._foto)
        if self.sistem is None:
            self._pesan("Nomor anak tangga belum dihitung. Tekan Hitung nomor dari video di kanan.")
        elif (mask == "Detektor" and not any(v is not None for v in self.sem_i)) or \
                (mask == "Per anak tangga" and not self._ada_mask):
            self._pesan("Mask belum tersimpan untuk sesi ini: tekan Hitung ulang nomor dari video (sekitar 40 detik).")
        else:
            self._pesan("")
        self._gambar_tanda()

    def _baca_sem(self, i: int):
        """Peta kelas segmentasi frame i (0 latar / 1 riser / 2 tread), atau None."""
        if i >= len(self.sem_i) or self.sem_i[i] is None:
            return None
        if i not in self._cache_sem:
            peta = cv2.imread(str(self.sesi / "derived" / "ukuran_semantik" / f"{int(self.sem_i[i]):06d}.png"),
                              cv2.IMREAD_UNCHANGED)
            if peta is None:
                return None
            self._cache_sem[i] = peta
            while len(self._cache_sem) > MAKS_CACHE:
                self._cache_sem.pop(next(iter(self._cache_sem)))
        return self._cache_sem[i]

    def _garis_pilihan(self, kecil: np.ndarray, s: float) -> np.ndarray:
        """Tepi kuning anak tangga yang sedang diisi di atas mask detektor."""
        if self.pilih is None or self.i >= len(self.jejak_i):
            return kecil
        for ke, nomor, _, _, _, poli in self.jejak_i[self.i]:
            if poli is not None and (ke, nomor) == self.pilih:
                for ox in self._panel_x:
                    cv2.polylines(kecil, [np.round((poli + (ox, 0)) * s).astype(np.int32)], True, (102, 224, 255), 3,
                                  cv2.LINE_AA)
        return kecil

    def _tempel_mask(self, kecil: np.ndarray, s: float) -> np.ndarray:
        """Mask anak tangga bernomor (isi transparan + garis tepi) dan mask tak bernomor (garis abu tipis)."""
        isi, garis = kecil.copy(), []
        for ke, nomor, k, _, _, poli in self.jejak_i[self.i]:
            if poli is None:
                continue
            warna = BGR_R if k == "R" else BGR_T
            for ox in self._panel_x:
                pts = np.round((poli + (ox, 0)) * s).astype(np.int32)
                cv2.fillPoly(isi, [pts], warna, cv2.LINE_AA)
                garis.append((pts, (102, 224, 255) if (ke, nomor) == self.pilih else warna,
                              3 if (ke, nomor) == self.pilih else 2))
        kecil = cv2.addWeighted(isi, 0.35, kecil, 0.65, 0)
        for k, poli in self.lain_i[self.i] if self.i < len(self.lain_i) else []:
            for ox in self._panel_x:
                cv2.polylines(kecil, [np.round((poli + (ox, 0)) * s).astype(np.int32)], True, (150, 150, 150), 1,
                              cv2.LINE_AA)
        for pts, warna, tebal in garis:
            cv2.polylines(kecil, [pts], True, warna, tebal, cv2.LINE_AA)
        return kecil

    def _gambar_tanda(self) -> None:
        c = self.kanvas
        c.delete("tanda")
        if self.i >= len(self.jejak_i):
            return
        warna_k = WARNA_DET if self.mode_mask.get() == "Detektor" else {"R": WARNA_R, "T": WARNA_T}
        for ke, nomor, k, x, y, _ in self.jejak_i[self.i]:
            dipilih = (ke, nomor) == self.pilih
            teks = f"{self.nama(ke, nomor)} {k}" + (" ✓" if self.lengkap(ke, nomor) else "")
            for ox in self._panel_x:
                cx, cy = self._ox + (x + ox) * self._skala, self._oy + y * self._skala
                tag = ("tanda", f"s:{ke}:{nomor}:{k}")
                t = c.create_text(cx, cy, text=teks, fill="#111111", tags=tag,
                                  font=("Segoe UI", 12 if dipilih else 10, "bold"))
                x0, y0, x1, y1 = c.bbox(t)
                kotak = c.create_rectangle(x0 - 5, y0 - 2, x1 + 5, y1 + 2, fill=warna_k[k],
                                           outline=WARNA_PILIH if dipilih else "#111111", width=3 if dipilih else 1,
                                           tags=tag)
                c.tag_raise(t, kotak)

    # ------------------------------------------------------------ putar
    def putar_jeda(self) -> None:
        if self._main:
            self.jeda()
        else:
            self.putar()

    def putar(self) -> None:
        if self.cap is None or self.n == 0:
            return
        self._sembunyikan_kotak()
        if self.i >= self.n - 1:
            self._tampilkan(0)
        self._main = True
        self.btn_putar.configure(text="⏸ Jeda")
        self._mulai_jam()
        self._detak()

    def jeda(self) -> None:
        self._main = False
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.btn_putar.configure(text="▶ Putar")

    def _mulai_jam(self) -> None:
        self._t0, self._i0 = time.perf_counter(), self.i

    def _detak(self) -> None:
        self._job = None
        if not self._main:
            return
        laju = KECEPATAN.get(self.kecepatan.get(), 1.0)
        target = min(self._i0 + int((time.perf_counter() - self._t0) * self.fps * laju), self.n - 1)
        tanya = []
        for j in range(self.i + 1, target + 1):     # frame yang dilompati tetap diperiksa
            tanya = self._anak_tangga_baru(j)
            if tanya:
                target = j
                break
        if target != self.i:
            self._tampilkan(target)
        if tanya:
            self.jeda()
            self._lanjut_setelah = True
            self._antrean = tanya
            self._tanya_berikut()
            return
        if self.i >= self.n - 1:
            self.jeda()
            return
        self._job = self.after(max(5, int(500 / (self.fps * laju))), self._detak)

    def _anak_tangga_baru(self, j: int) -> list[tuple]:
        """Anak tangga yang pertama kali terlihat di frame j selama pemutaran ini dan ukurannya belum lengkap."""
        if j >= len(self.tanda_i):
            return []
        baru = sorted({(ke, nomor) for ke, nomor, *_ in self.tanda_i[j]} - self._dilihat)
        self._dilihat.update(baru)
        if not self.berhenti_baru.get():
            return []
        return [s for s in baru if not self.lengkap(*s) and self.n_teramati.get(s, 0) >= MIN_FRAME_TANYA]

    def langkah(self, d: int) -> None:
        self.jeda()
        self._tampilkan(self.i + d)

    def lompat(self, i: int) -> None:
        self.jeda()
        self._tampilkan(i)

    def _geser_slider(self, nilai) -> None:
        i = int(float(nilai))
        if i != self.i:
            self.jeda()
            self._tampilkan(i)

    # ------------------------------------------------------------ klik
    def _klik_kanvas(self, e) -> None:
        # Dicari lewat koordinat, bukan item "current": label digambar ulang setiap frame,
        # sehingga "current" bisa hilang di antara gerak tetikus dan klik.
        for item in reversed(self.kanvas.find_overlapping(e.x - 1, e.y - 1, e.x + 1, e.y + 1)):
            for tag in self.kanvas.gettags(item):
                if tag.startswith("s:"):
                    _, ke, nomor, k = tag.split(":")
                    self._lanjut_setelah = self._main
                    self.jeda()
                    self._antrean = []
                    self._tanya((int(ke), int(nomor)), "T" if k == "T" else "R",
                                "Dipilih dari video. Isi ukuran meteran anak tangga ini.")
                    return
        kena = self._mask_di(e.x, e.y)
        if kena is not None:
            self._lanjut_setelah = self._main
            self.jeda()
            self._antrean = []
            self._tanya(kena[:2], kena[2], "Dipilih dari video. Isi ukuran meteran anak tangga ini.")
            return
        if self.kotak.winfo_ismapped():
            return
        self.putar_jeda()

    def _mask_di(self, cx: float, cy: float):
        """(ke, nomor, 'R'/'T') mask bernomor di titik kanvas (cx, cy), atau None."""
        if self.mode_mask.get() == "Mati" or self.i >= len(self.jejak_i) or self._bgr is None:
            return None
        u, v = (cx - self._ox) / self._skala, (cy - self._oy) / self._skala
        wp = self._bgr.shape[1] // 2
        if len(self._panel_x) > 1 and u >= wp:
            u -= wp
        for ke, nomor, k, _, _, poli in self.jejak_i[self.i]:
            if poli is not None and cv2.pointPolygonTest(poli.astype(np.float32), (float(u), float(v)), False) >= 0:
                return ke, nomor, k
        return None

    def _x_pita(self, i: int) -> float:
        lebar = max(self.pita.winfo_width(), 1)
        tepi = int(self.skala.cget("sliderlength")) / 2 + 2
        return tepi + (lebar - 2 * tepi) * min(max(i / max(self.n - 1, 1), 0.0), 1.0)

    def _gambar_pita(self) -> None:
        p = self.pita
        p.delete("all")
        if self.n <= 1:
            return
        kelompok: list[list] = []               # anak tangga yang muncul berdekatan berbagi satu penanda
        for s, i in sorted(self.muncul.items(), key=lambda kv: (kv[1], kv[0])):
            if self.n_teramati.get(s, 0) < MIN_FRAME_TANYA:
                continue
            x = self._x_pita(i)
            if kelompok and x - kelompok[-1][0] < 26:
                kelompok[-1][1].append(s)
            else:
                kelompok.append([x, [s]])
        for x, anggota in kelompok:
            warna = GREEN if all(self.lengkap(*s) for s in anggota) else WARNA_BELUM
            p.create_line(x, 0, x, 9, fill=warna, width=3)
            p.create_text(x, 18, text=" ".join(self.nama(*s) for s in anggota), fill=warna,
                          font=("Segoe UI", 8, "bold"), anchor="w" if x < 30 else "center")

    def _klik_pita(self, e) -> None:
        calon = [(s, i) for s, i in self.muncul.items() if self.n_teramati.get(s, 0) >= MIN_FRAME_TANYA]
        if not calon:
            return
        s, i = min(calon, key=lambda si: abs(self._x_pita(si[1]) - e.x))
        self._ke_anak_tangga(s)

    def _ke_anak_tangga(self, s: tuple) -> None:
        """Lompat ke frame saat anak tangga s paling jelas terlihat lalu tanyakan ukurannya."""
        frame = [j for j, t in enumerate(self.tanda_i) if any((ke, n) == s for ke, n, *_ in t)]
        if frame:
            # Pertengahan rentang teramati: anak tangga sudah dekat, belum keluar dari gambar.
            self.lompat(frame[len(frame) // 2])
        else:
            self.jeda()
        self._lanjut_setelah = False
        self._antrean = []
        self._tanya(s, "R", "Dipilih dari daftar.")

    # ------------------------------------------------------------ kotak tanya
    def _tanya_berikut(self) -> None:
        if not self._antrean:
            self._sembunyikan_kotak()
            if self._lanjut_setelah:
                self._lanjut_setelah = False
                self.putar()
            return
        s = self._antrean.pop(0)
        self._tanya(s, "R", "Baru terlihat. Berapa ukuran meterannya?")

    def _tanya(self, s: tuple, fokus: str, keterangan: str) -> None:
        self.pilih = s
        ke, nomor = s
        m = self.meteran(ke, nomor)
        self.riser.set("" if m.get("tinggi_riser_cm") is None else f"{m['tinggi_riser_cm']:g}")
        self.tread.set("" if m.get("panjang_tread_cm") is None else f"{m['panjang_tread_cm']:g}")
        self.kotak_judul.configure(text=f"Anak tangga {self.nama(ke, nomor)}"
                                        + (f"  (nomor pelacak #{nomor})" if self.geser.get(ke, 0) else ""))
        sisa = f" • {len(self._antrean)} lagi menunggu" if self._antrean else ""
        self.kotak_sub.configure(text=keterangan + sisa)
        self.kotak_status.configure(text="")
        self.btn_simpan.configure(text="Simpan & lanjutkan ▶ (Enter)" if self._lanjut_setelah or self._antrean
                                  else "Simpan (Enter)")
        self._render()
        # Kotak tidak boleh menutupi nomor yang sedang diisi: anak tangga di bawah -> kotak di atas.
        ys = [self._oy + y * self._skala for k_, n_, _, _, y, _ in (self.jejak_i[self.i] if self.i < len(self.jejak_i)
                                                                      else []) if (k_, n_) == s]
        di_bawah = bool(ys) and sum(ys) / len(ys) > self.kanvas.winfo_height() * 0.55
        self.kotak.place(relx=0.5, rely=0.0 if di_bawah else 1.0, anchor="n" if di_bawah else "s",
                         y=14 if di_bawah else -14)
        self.kotak.lift()
        isian = self.isi_tread if fokus == "T" else self.isi_riser
        isian.focus_set()
        isian.select_range(0, "end")
        self._segarkan_geser()
        self._bangun_tabel()

    def _sembunyikan_kotak(self) -> None:
        if self.kotak.winfo_ismapped():
            self.kotak.place_forget()
        self.pilih = None
        self.focus_set()
        self._render()
        self._bangun_tabel()

    def lewati(self) -> None:
        if not self.kotak.winfo_ismapped():
            return
        self._tanya_berikut()

    def simpan(self) -> None:
        if self.pilih is None:
            return
        ke, nomor = self.pilih
        try:
            nilai = {"tinggi_riser_cm": _angka(self.riser.get()), "panjang_tread_cm": _angka(self.tread.get())}
        except ValueError:
            self.kotak_status.configure(text="Angka tidak terbaca. Pakai titik atau koma desimal, mis. 17,5.")
            return
        if any(v is not None and not 0 < v <= 200 for v in nilai.values()):
            self.kotak_status.configure(text="Ukuran harus di antara 0 dan 200 cm.")
            return
        akar, tid, f = self.studio.root_data, self.tid.get(ke), self.fisik(ke, nomor)
        met = UM.baca(self.sesi, akar, tid, ke)            # baca ulang: isian kartu/impor lain tidak tertimpa
        anak = dict(met["anak_tangga"])
        isi = dict(anak.get(f, {}))
        for kunci, v in nilai.items():
            if v is None:
                isi.pop(kunci, None)
            else:
                isi[kunci] = v
        if isi:
            anak[f] = isi
        else:
            anak.pop(f, None)
        geser = dict(met["geser"])
        geser[ke] = self.geser.get(ke, 0)
        tempat = UM.tulis(self.sesi, akar, tid, anak, met["lebar_cm"], geser, ke)
        self.met[ke] = UM.baca(self.sesi, akar, tid, ke)["anak_tangga"]
        a = self.info_sistem.get((ke, nomor)) or {}
        beda = []
        for kunci, besaran, huruf in (("tinggi_riser_cm", "tinggi_riser", "R"), ("panjang_tread_cm", "panjang_tread", "T")):
            if nilai[kunci] is not None and (a.get(besaran) or {}).get("cm") is not None:
                beda.append(f"{huruf} {a[besaran]['cm'] - nilai[kunci]:+.1f}")
        aneh = [huruf for kunci, huruf in (("tinggi_riser_cm", "riser"), ("panjang_tread_cm", "tread"))
                if nilai[kunci] is not None and not WAJAR[kunci][0] <= nilai[kunci] <= WAJAR[kunci][1]]
        self.status.set(f"{self.nama(ke, nomor)} tersimpan ke {tempat}."
                        + (f" Selisih sistem - meteran (cm): {', '.join(beda)}." if beda else "")
                        + (f" Periksa lagi: {' dan '.join(aneh)} di luar ukuran tangga yang lazim." if aneh else ""))
        self._kabari_panel()
        self._gambar_pita()
        if aneh:                                  # tetap tersimpan, tapi beri kesempatan membetulkan salah ketik
            self.kotak_status.configure(text=f"Tersimpan. Periksa lagi: {' dan '.join(aneh)} di luar ukuran lazim. "
                                             "Tekan Enter sekali lagi bila sudah benar.")
            if getattr(self, "_peringatan", None) != (ke, nomor, tuple(nilai.values())):
                self._peringatan = (ke, nomor, tuple(nilai.values()))
                self._bangun_tabel()
                return
        self._peringatan = None
        self._tanya_berikut()

    # ------------------------------------------------------------ penomoran
    def _segarkan_geser(self) -> None:
        ke = self._ke_aktif()
        self.lbl_geser.configure(text=(f"Penomoran {ke}: " if len(self._daftar_ke()) > 1 else "")
                                 + "nomor sistem #1 = anak tangga fisik ke-")
        if str(self.geser_var.get() if self._spin_sah() else "") != str(self.geser.get(ke, 0) + 1):
            self.geser_var.set(self.geser.get(ke, 0) + 1)

    def _spin_sah(self) -> bool:
        try:
            self.geser_var.get()
            return True
        except (tk.TclError, ValueError):
            return False

    def _ubah_geser(self) -> None:
        if not self._spin_sah():
            return
        ke = self._ke_aktif()
        baru = max(1, int(self.geser_var.get())) - 1
        if baru == self.geser.get(ke, 0):
            return
        self.geser[ke] = baru
        akar, tid = self.studio.root_data, self.tid.get(ke)
        met = UM.baca(self.sesi, akar, tid, ke)
        geser = dict(met["geser"])
        geser[ke] = baru
        UM.tulis(self.sesi, akar, tid, met["anak_tangga"], met["lebar_cm"], geser, ke)
        self.status.set(f"Nomor sistem #1 kini anak tangga fisik ke-{baru + 1}.")
        if self.pilih is not None:
            self._tanya(self.pilih, "R", self.kotak_sub.cget("text"))
        self._kabari_panel()
        self._render()
        self._gambar_pita()
        self._bangun_tabel()

    # ------------------------------------------------------------ tabel
    def _bangun_tabel(self) -> None:
        for w in self.tabel.winfo_children():
            w.destroy()
        semua = sorted(set(self.info_sistem) | {s for s, n in self.n_teramati.items() if n >= MIN_FRAME_TANYA})
        for kol, teks in enumerate(("", "sistem R / T", "meteran R / T")):
            tk.Label(self.tabel, text=teks, bg=PANEL, fg=MUTED, font=("Segoe UI", 7, "bold")).grid(
                row=0, column=kol, sticky="w", padx=2)
        if not semua:
            tk.Label(self.tabel, text="(belum ada nomor dari pelacak)", bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8)).grid(row=1, column=0, columnspan=3, sticky="w")
            return
        for baris, s in enumerate(semua, start=1):
            a, m = self.info_sistem.get(s) or {}, self.meteran(*s)
            latar = "#E9D8CC" if s == self.pilih else PANEL
            isi = (
                (self.nama(*s), INK, ("Segoe UI", 9, "bold")),
                (f"{_cm(a.get('tinggi_riser'))} / {_cm(a.get('panjang_tread'))}", INK, ("Consolas", 9)),
                (f"{m.get('tinggi_riser_cm', '-')} / {m.get('panjang_tread_cm', '-')}"
                 + ("  ✓" if self.lengkap(*s) else ""), GREEN if self.lengkap(*s) else WARNA_BELUM,
                 ("Consolas", 9, "bold")),
            )
            for kol, (teks, warna, font) in enumerate(isi):
                lbl = tk.Label(self.tabel, text=teks, bg=latar, fg=warna, font=font, cursor="hand2", anchor="w")
                lbl.grid(row=baris, column=kol, sticky="ew", padx=0, ipadx=3, ipady=1)
                lbl.bind("<Button-1>", lambda e, s=s: self._ke_anak_tangga(s))
        self.tabel.columnconfigure(2, weight=1)

    # ------------------------------------------------------------ hitung
    def hitung(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.jeda()
        self.btn_hitung.configure(text="Menghitung…")
        self._progres, self._galat = 0, None
        sesi = self.sesi

        def kerja():
            try:
                _modul_ukuran().hitung(sesi, video=False, progres=lambda n: setattr(self, "_progres", n),
                                       henti=lambda: self._tutup)
            except Exception as e:                                  # noqa: BLE001
                baris = [b for b in str(e).strip().splitlines() if b.strip()]
                self._galat = f"{type(e).__name__}: {baris[-1][:200] if baris else ''}"

        self._thread = threading.Thread(target=kerja, daemon=True)
        self._thread.start()
        self.after(500, self._pantau)

    def _pantau(self) -> None:
        if self._tutup:
            return
        if self._thread is not None and self._thread.is_alive():
            self.status.set(f"Menjalankan pelacak pada seluruh rekaman… {self._progres or 0}/{self.n} frame")
            self.after(500, self._pantau)
            return
        if self._galat:
            self.status.set(f"Gagal: {self._galat}")
            self.btn_hitung.configure(text="Hitung nomor dari video")
            return
        self._dilihat.clear()
        self._muat_sistem()
        self._render()
        self._kabari_panel(sistem_baru=True)
        n = len(self.info_sistem)
        self.status.set(f"Selesai: {n} anak tangga bernomor. Tekan Putar; video berhenti di setiap anak tangga baru.")

    # ------------------------------------------------------------ lain-lain
    def _kabari_panel(self, sistem_baru: bool = False) -> None:
        panel = getattr(self.studio, "panel_ukuran", None)
        if panel is not None and hasattr(panel, "segarkan_dari_luar"):
            panel.segarkan_dari_luar(self.sesi, sistem_baru)

    def _sedang_mengetik(self, e) -> bool:
        return isinstance(e.widget, (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Combobox))

    def tutup(self) -> None:
        self._tutup = True
        self.jeda()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if getattr(self.studio, "_pemutar_ukur", None) is self:
            self.studio._pemutar_ukur = None
        self.destroy()
