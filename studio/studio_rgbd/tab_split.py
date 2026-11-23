"""Tab Studio "7. Pilah & Split": pilih frame yang dibuang, lalu bagi rekaman ke train, val, dan test.

Satu arti "buang" di seluruh Studio: ``frame_state.json`` ``di_sampah`` (sama dengan Buang di tab
Label); frame yang dibuang tidak dilatih, tidak diunggah, dan bisa dipulihkan kapan saja. Frame yang
disisihkan kurasi otomatis (``kurasi.json``, mis. hampir kembar dengan frame sebelumnya) tampil
sebagai "dibuang otomatis"; memulihkannya mencatat keputusan manual kurasi.

Daftar "perlu dicek" adalah gabungan pemeriksaan label (``qa_label.json``) dan "perlu dilihat"
kurasi (model CV, depth kosong, potongan mask kecil); saran buang di urutan teratas. Satu daftar
frame mencakup semua rekaman siap latih (✔ di tab 2); klik rekaman di pohon melompat ke framenya.

Pembagian disimpan di ``<dataset>/studio_rgbd/split_dataset.json`` dan dibaca skrip latih lewat
``--split``; perintah latih juga membawa ``--kurasi`` sehingga yang dilatih sama dengan yang tampil
"dipakai" di sini.

Pintasan saat tab ini aktif: ←/→ (atau D/Spasi) frame, Del buang/pulihkan, Enter buka di tab
Label, PageUp/PageDown satu strip, Home/End; 1/2/3 Train/Val/Test dan Backspace kosongkan untuk
baris terpilih di pohon rekaman.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from .segmentasi_convnext_depth import akar_aplikasi
    from . import catatan_rekaman as CR, kurasi as KU, qa_label, sinkron, unggah_hf
    from .ui_bantu import jadikan_modal, kolom_gulir, tombol_menu
except ImportError:
    from segmentasi_convnext_depth import akar_aplikasi
    import catatan_rekaman as CR
    import kurasi as KU
    import qa_label
    import sinkron
    import unggah_hf
    from ui_bantu import jadikan_modal, kolom_gulir, tombol_menu

BG, PANEL, INK, MUTED = "#F3EEE7", "#FFFDFC", "#382D29", "#786962"      # sama dengan studio_dataset_rgbd
ACCENT, GREEN, BLUE = "#754C3B", "#6B8B63", "#407EA3"
WARNA_SET = {"train": "#6B8B63", "val": "#C98A2B", "test": "#407EA3", "abaikan": "#9A8F8A", None: "#E8DDD5"}
LABEL_SET = {"train": "TRAIN", "val": "VAL", "test": "TEST", "abaikan": "ABAIKAN", None: "belum"}
WARNA_STATUS = {"pakai": "#1E8E3E", "cek": "#E8710A", "buang": "#C5221F", "otomatis": "#8A817C"}
TEKS_STATUS = {"pakai": "DIPAKAI", "cek": "PERLU DICEK", "buang": "DIBUANG", "otomatis": "DIBUANG OTOMATIS"}
RISER_RGB, TREAD_RGB = (255, 90, 90), (90, 169, 255)       # sama dengan kanvas label
TW, TH = 160, 90                                           # ukuran gambar kecil di strip
FILTER = ("perlu dicek", "saran buang", "semua", "dibuang", "latar (tanpa mask)")
BERKAS_CV = "runs/cv_data6023_20261120/hasil_cv.json"     # Dice out-of-fold terakhir untuk "perlu dilihat"


def _pustaka():
    akar = str(akar_aplikasi())
    if akar not in sys.path:
        sys.path.insert(0, akar)
    from rgbd_convnext.data import split_dataset, dataset_d435
    return split_dataset, dataset_d435


def _cerahkan(rgb: np.ndarray) -> np.ndarray:
    """Hanya untuk mata: model tetap menerima citra asli."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(l, (1, 99.5))
    l = np.clip((l - lo) / max(hi - lo, 8.0), 0, 1) ** 0.5
    lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply((l * 255).astype(np.uint8))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _tumpang_mask(rgb: np.ndarray, sem: np.ndarray, alfa: float = 0.4, tebal: int = 2) -> np.ndarray:
    out = rgb.astype(np.float32)
    for c, w in ((2, TREAD_RGB), (1, RISER_RGB)):
        m = sem == c
        out[m] = out[m] * (1 - alfa) + np.array(w, np.float32) * alfa
    out = out.astype(np.uint8)
    for c, w in ((2, TREAD_RGB), (1, RISER_RGB)):
        kontur, _ = cv2.findContours((sem == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, kontur, -1, w, tebal)
    return out


def _baca_frame(d: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """(RGB, peta kelas 0/1 riser/2 tread) persis seperti FrameD435 (tread lalu riser menimpa)."""
    bgr = cv2.imread(str(d / "color_raw.png"))
    riser = cv2.imread(str(d / "mask_objek.png"), 0)
    tread = cv2.imread(str(d / "mask_acuan.png"), 0)
    if bgr is None or riser is None or tread is None:
        return None
    sem = np.zeros(riser.shape, np.uint8)
    sem[tread > 0] = 2
    sem[riser > 0] = 1
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), sem


def _gambar_kecil(d: Path, cerah: bool) -> np.ndarray | None:
    v = _baca_frame(d)
    if v is None:
        return None
    rgb, sem = v
    rgb = cv2.resize(rgb, (TW, TH), interpolation=cv2.INTER_AREA)
    if cerah:
        rgb = _cerahkan(rgb)
    elif rgb.mean() < 40:                                  # frame malam dicerahkan agar isinya terlihat
        rgb = np.clip(rgb.astype(np.float32) * (90.0 / max(rgb.mean(), 5.0)), 0, 255).astype(np.uint8)
    return _tumpang_mask(rgb, cv2.resize(sem, (TW, TH), interpolation=cv2.INTER_NEAREST), 0.45, 1)


class Fr:
    """Satu frame siap latih (rekaman ✔, berkas lengkap, label terperiksa bila diminta), termasuk yang dibuang."""
    __slots__ = ("d", "r", "k", "i", "tg", "sampah", "mask", "mt")

    def __init__(self, d: Path, r: str, i: int | None, tg: str | None, sampah: bool, mask: bool, mt: float) -> None:
        self.d, self.r, self.k, self.i, self.tg = d, r, f"{r}/{d.name}", i, tg
        self.sampah, self.mask, self.mt = sampah, mask, mt


class TabSplit:
    def __init__(self, induk: tk.Frame, studio) -> None:
        self.studio = studio
        self.sd, self.dd = _pustaka()
        self.akar_data = Path(studio.root_data)
        self.akar = self.akar_data / "rekaman" / "tangga_naik"
        self.berkas = self.akar_data / "split_dataset.json"
        self.berkas_qa = self.akar_data / "qa_label.json"
        self.data = self.sd.baca(self.berkas)
        self.rekaman: list[str] = []
        self.semua: dict[str, list[Fr]] = {}             # rekaman -> frame siap latih termasuk yang dibuang
        self.frames: dict[str, list[Fr]] = {}            # rekaman -> frame yang dipakai melatih
        self.n_ekspor: dict[str, int] = {}
        self.catatan: dict[str, dict] = {}
        self.waktu: dict = {}
        self.daftar_tangga: dict = {}
        self.kur: dict = {"frame": {}, "manual": {}}
        self._t_kurasi = float("inf")
        self._jadwal_qa = None
        self.qa: dict = {}
        self.daftar: list[Fr] = []                       # frame pada strip (sesudah filter)
        self.i = 0                                       # posisi frame yang dilihat di daftar
        self.pilih: set[int] = set()                     # frame terpilih di strip (Shift/Ctrl+klik)
        self._jangkar = 0
        self._awal = 0                                   # indeks frame pertama yang tampil di strip
        self._q: queue.Queue = queue.Queue()
        self._sibuk = False
        self._abaikan_pilih_pohon = False
        self._sesi_awal: str | None = None
        self._geser_terakhir = 0.0
        self._jadwal_ringkas = None
        self._jadwal_kurasi = None
        self._foto = None
        self._kecil: OrderedDict = OrderedDict()         # (folder, cerah) -> PhotoImage
        self._cache_frame: OrderedDict = OrderedDict()   # folder -> (RGB, peta kelas)
        self._kunci = threading.Lock()
        self._tugas: list[tuple] = []
        self._ada_tugas = threading.Event()
        threading.Thread(target=self._pekerja, daemon=True, name="pilah-gambar").start()

        self.hanya_diperiksa = tk.BooleanVar(value=True)
        self.per_tangga = tk.BooleanVar(value=False)
        self.filter = tk.StringVar(value="perlu dicek")
        self.cerah = tk.BooleanVar(value=False)
        self.tampil_mask = tk.BooleanVar(value=True)
        self.rasio_train, self.rasio_val, self.rasio_test = (tk.IntVar(value=70), tk.IntVar(value=15),
                                                             tk.IntVar(value=15))
        self._ui(induk)
        studio.bind_all("<KeyPress>", self._tombol, add="+")
        self.after = induk.after
        self.induk = induk
        self.after(50, self._poll)
        self.muat_ulang()

    # ------------------------------------------------------------------ UI
    def _ui(self, induk: tk.Frame) -> None:
        """Sedikit tombol: yang sering dipakai tampil, sisanya di menu "⋯ Lainnya"."""
        st = self.studio
        f = tk.Frame(induk, bg=BG); f.pack(fill="both", expand=True, padx=14, pady=14)
        kiri = kolom_gulir(f, 480, BG, padx=(0, 12))
        kanan = tk.Frame(f, bg=BG); kanan.pack(side="left", fill="both", expand=True)

        # --- rekaman
        b, i = st.card(kiri, "Rekaman siap latih"); b.pack(fill="x", pady=(0, 8))
        f2 = tk.Frame(i, bg=PANEL); f2.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(f2, columns=("set", "pakai", "buang", "cek", "waktu"), show="tree headings",
                                 selectmode="extended", height=13)
        for kol, teks, lebar, rata in (("#0", "Rekaman", 146, "w"), ("set", "Set", 52, "center"),
                                       ("pakai", "Dipakai", 54, "e"), ("buang", "Dibuang", 54, "e"),
                                       ("cek", "Cek", 38, "e"), ("waktu", "Direkam", 100, "w")):
            self.tree.heading(kol, text=teks); self.tree.column(kol, width=lebar, anchor=rata)
        for s_, w in WARNA_SET.items():
            self.tree.tag_configure(str(s_), foreground=w)
        for nama, w in CR.WARNA.items():
            self.tree.tag_configure(f"stabilo_{nama}", background=w)
        self.tree.tag_configure("scene", font=("Segoe UI", 9, "italic"))
        self.tree.tag_configure("belum_selesai", foreground=MUTED)
        gulir = ttk.Scrollbar(f2, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=gulir.set)
        self.tree.pack(side="left", fill="both", expand=True); gulir.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._pilihan_berubah())
        # Panah/Spasi/Del/Enter di pohon menggerakkan frame, bukan membuka/menutup baris (bawaan Treeview).
        for tombol, aksi in (("<space>", lambda: self.geser(1)), ("<Right>", lambda: self.geser(1)),
                             ("<Left>", lambda: self.geser(-1)), ("<Delete>", self.buang_pulihkan),
                             ("<Return>", self.buka_di_label)):
            self.tree.bind(tombol, lambda e, a=aksi: (a(), "break")[1])
        self.info_rek = tk.Label(i, text="", bg=PANEL, fg=INK, justify="left", anchor="w", wraplength=420,
                                 font=("Segoe UI", 9))
        self.info_rek.pack(fill="x", pady=(4, 0))
        tk.Label(i, text="Set baris terpilih  (tombol 1 / 2 / 3, Backspace kosongkan)", bg=PANEL,
                 fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 2))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x")
        for teks, s_ in (("Train", "train"), ("Val", "val"), ("Test", "test"), ("Kosongkan", None)):
            st.tombol_ringkas(baris, teks, lambda s_=s_: self.atur_rekaman(s_), WARNA_SET[s_],
                              "white" if s_ else INK, width=90).pack(side="left", expand=True, fill="x", padx=1)

        # --- pembagian
        b, i = st.card(kiri, "Bagi train / val / test"); b.pack(fill="x", pady=(0, 8))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x")
        tk.Label(baris, text="Rasio", bg=PANEL, fg=MUTED).pack(side="left")
        self.rasio_teks = tk.StringVar(value="70/15/15")
        cb = ttk.Combobox(baris, textvariable=self.rasio_teks, state="readonly", width=9,
                          values=("70/15/15", "80/10/10", "60/20/20"))
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self._atur_rasio(*map(int, self.rasio_teks.get().split("/"))))
        st.tombol_ringkas(baris, "⚖ Bagi otomatis", self.bagi_otomatis, GREEN, width=150).pack(side="right")
        self.tabel = tk.Frame(i, bg=PANEL); self.tabel.pack(fill="x", pady=(8, 0))
        self.rekomendasi = tk.Label(i, text="", bg=PANEL, fg=ACCENT, justify="left", anchor="w",
                                    wraplength=420, font=("Segoe UI", 9))
        self.rekomendasi.pack(fill="x", pady=(6, 0))

        # --- kanan: pilah frame
        b, i = st.card(kanan, "Pilah frame")
        b.pack(fill="both", expand=True)
        self.lbl_atas = tk.Label(i, text="", bg=PANEL, fg=INK, justify="left", anchor="w", wraplength=780,
                                 font=("Segoe UI", 10, "bold"))
        self.lbl_atas.pack(fill="x")
        self.lbl_peringatan = tk.Label(i, text="", bg=PANEL, fg="#B3261E", justify="left", anchor="w",
                                       wraplength=780, font=("Segoe UI", 9))
        self.lbl_peringatan.pack(fill="x")
        kepala = tk.Frame(i, bg=PANEL); kepala.pack(fill="x", pady=(6, 0))
        tk.Label(kepala, text="Tampilkan", bg=PANEL, fg=MUTED).pack(side="left")
        cb = ttk.Combobox(kepala, textvariable=self.filter, state="readonly", width=15, values=FILTER)
        cb.pack(side="left", padx=4); cb.bind("<<ComboboxSelected>>", lambda e: self._isi_daftar())
        self.lbl_jumlah = tk.Label(kepala, text="", bg=PANEL, fg=INK, font=("Segoe UI", 9))
        self.lbl_jumlah.pack(side="left", padx=6)
        tombol_menu(st, kepala, "⋯ Lainnya", [
            ("↻ Periksa ulang (frame mirip, depth kosong, model CV)", self.periksa_ulang),
            ("📦 Versi data…", self.buka_versi),
            None,
            ("📋 Salin perintah latih", self.salin_latih),
            ("📋 Salin perintah uji (test)", self.salin_uji),
            ("📄 Ringkasan rekaman (CSV)", self.ringkasan_csv),
            None,
            ("⬇ Tarik dataset dari Hugging Face", self.tarik_hf),
            ("☁ Push dataset ke Hugging Face", self.push_hf),
            None,
            ("Tampilkan mask  (M)", self.tampil_mask, self._tampilan_berubah),
            ("Cerahkan gambar  (C)", self.cerah, self._tampilan_berubah),
            ("Kelompokkan rekaman per tangga fisik", self.per_tangga, self._isi_tree),
            ("Hanya label yang sudah divalidasi manual", self.hanya_diperiksa, self.muat_ulang),
        ], width=110).pack(side="right")
        self.btn_massal = st.tombol_ringkas(kepala, "🗑 Buang semua yang tampil", self.buang_semua_tampil,
                                            "#F3D8D4", INK, width=190)
        self.btn_massal.pack(side="right", padx=(0, 4))
        kepala2 = tk.Frame(i, bg=PANEL); kepala2.pack(fill="x", pady=(6, 0))
        self.judul = tk.Label(kepala2, text="", bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold"))
        self.judul.pack(side="left")
        self.lencana = tk.Label(kepala2, text="", bg=PANEL, fg="white", font=("Segoe UI", 9, "bold"), padx=8)
        self.lencana.pack(side="left", padx=8)
        self.info_cek = tk.Label(i, text="", bg=PANEL, fg="#8A5A12", justify="left", anchor="w", wraplength=780,
                                 font=("Segoe UI", 9))
        self.info_cek.pack(fill="x")
        self.ringkas = tk.Label(i, text="", bg=PANEL, fg=MUTED, justify="left", anchor="w", wraplength=780,
                                font=("Segoe UI", 9))
        self.ringkas.pack(side="bottom", fill="x", pady=(4, 0))
        tk.Label(i, text="Del buang / pulihkan  ·  K label benar (hapus tanda cek)  ·  Enter perbaiki di tab Label  ·  "
                         "←/→ pindah  ·  Shift/Ctrl+klik pilih banyak  ·  M mask  ·  C cerahkan",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="bottom", anchor="w")
        self.strip = tk.Canvas(i, bg="#27201E", highlightthickness=0, height=TH + 30)
        self.strip.pack(side="bottom", fill="x", pady=(4, 0))
        self.strip.bind("<Configure>", lambda e: self._gambar_strip())
        self.strip.bind("<Button-1>", lambda e: self._klik_strip(e, "satu"))
        self.strip.bind("<Shift-Button-1>", lambda e: self._klik_strip(e, "rentang"))
        self.strip.bind("<Control-Button-1>", lambda e: self._klik_strip(e, "balik"))
        self.strip.bind("<Double-Button-1>", lambda e: self.buka_di_label())
        for urutan, arah in (("<Button-4>", -1), ("<Button-5>", 1)):
            self.strip.bind(urutan, lambda e, a=arah: self.geser(a))
        self.strip.bind("<MouseWheel>", lambda e: self.geser(-1 if e.delta > 0 else 1))
        self.kanvas = tk.Canvas(i, bg="#27201E", highlightthickness=0, height=200)
        self.kanvas.pack(fill="both", expand=True, pady=(6, 0))
        self.kanvas.bind("<Configure>", lambda e: self._gambar())
        self.kanvas.bind("<Double-Button-1>", lambda e: self.buka_di_label())

    # ------------------------------------------------------------ data
    def muat_ulang(self) -> None:
        """Pindai frame siap latih di thread latar; pratinjau dan posisi frame dipertahankan."""
        if self._sibuk:
            self.induk.after(1000, self.muat_ulang)      # pekerjaan lain berjalan: muat sesudahnya
            return
        self._sibuk = True
        self.data = self.sd.baca(self.berkas)            # tanda DATA TEST bisa diubah dari tab 2
        if not self.daftar:
            self.ringkas.config(text="Membaca daftar frame…")
        hanya = self.hanya_diperiksa.get()
        akar_data, wajib = self.akar_data, self.dd.BERKAS_WAJIB

        def kerja():
            try:
                rek = sorted(p.name for p in self.akar.iterdir() if p.is_dir()) if self.akar.exists() else []
                semua, n_ekspor, catatan, waktu = {}, {}, {}, {}
                for r in rek:
                    sesi = self.akar / r
                    c = CR.baca(sesi)
                    catatan[r] = c
                    w = CR.waktu_rekaman(sesi)
                    waktu[r] = w[0] if w else None
                    ek = sesi / "exports" / "frames"
                    folder = sorted(d for d in ek.iterdir() if d.is_dir()) if ek.exists() else []
                    n_ekspor[r] = len(folder)
                    isi = []
                    if c.get("selesai") is True:           # rekaman belum ✔ tidak dipakai melatih
                        for d in folder:
                            if not all((d / n).exists() for n in wajib):
                                continue
                            try:
                                sampah = bool(json.loads((d / "frame_state.json").read_text()).get("di_sampah"))
                            except (OSError, ValueError):
                                sampah = False
                            lab = d / "label_draft.json"
                            try:
                                j, mt = json.loads(lab.read_text()), lab.stat().st_mtime
                            except (OSError, ValueError):
                                j, mt = {}, 0.0
                            if hanya and not j.get("diperiksa_manual", not j.get("otomatis", False)):
                                continue                     # sama dengan frame_bersih(hanya_diperiksa=True)
                            pol = j.get("poligon") or {}
                            mask = any(len(q) >= 3 for k in ("objek", "acuan") for q in pol.get(k, []))
                            i = CR.indeks_frame(d.name)
                            isi.append(Fr(d, r, i, CR.tangga_untuk(c, i), sampah, mask, mt))
                    semua[r] = isi
                try:
                    qa = json.loads(self.berkas_qa.read_text()) if self.berkas_qa.exists() else {}
                except (OSError, ValueError):
                    qa = {}
                self._q.put(("muat", (rek, semua, n_ekspor, catatan, waktu, CR.baca_tangga(akar_data),
                                      KU.baca(akar_data), qa)))
            except Exception as e:                           # noqa: BLE001
                self._q.put(("galat", f"Gagal membaca dataset: {e}"))
        threading.Thread(target=kerja, daemon=True).start()

    def _poll(self) -> None:
        segarkan_strip = False
        try:
            while True:
                jenis, isi = self._q.get_nowait()
                if jenis == "kecil":
                    kunci, arr = isi
                    self._kecil[kunci] = ImageTk.PhotoImage(Image.fromarray(arr))
                    while len(self._kecil) > 600:
                        self._kecil.popitem(last=False)
                    segarkan_strip = True
                elif jenis == "muat":
                    self._sibuk = False
                    self._terima_muat(*isi)
                elif jenis == "status":
                    self.ringkas.config(text=isi)
                elif jenis == "galat":
                    self._sibuk = False
                    self.ringkas.config(text=isi)
                    messagebox.showerror("Pilah & Split", isi)
                elif jenis == "kurasi_selesai":
                    self._sibuk = False
                    self.kur = isi
                    self._t_kurasi = self._waktu_kurasi()
                    self._segarkan_semua()
                    self.ringkas.config(text=f"Periksa ulang selesai ({isi.get('dibuat', '')}): "
                                             f"{sum(1 for v in isi.get('frame', {}).values() if not v['pilih'])} "
                                             "frame dibuang otomatis, sisanya di 'perlu dicek'.")
                elif jenis == "csv_selesai":
                    self._sibuk = False
                    self.ringkas.config(text=f"Ringkasan rekaman ditulis: {isi[0]}")
                    messagebox.showinfo("Ringkasan rekaman", f"Ditulis ke:\n{isi[0]}\n\n{isi[1]}")
                elif jenis == "tarik_selesai":
                    self._sibuk = False
                    teks = (f"Tarik selesai: {isi['frame_baru']} frame baru, {isi['label_diperbarui']} label "
                            f"diperbarui dari HF, {isi['label_lokal_lebih_baru']} label lokal lebih baru dipertahankan.")
                    self.ringkas.config(text=teks)
                    messagebox.showinfo("Hugging Face", teks + "\n\nPilih rekamannya di tab Tinjau untuk melabel.")
                    self.muat_ulang()
                    self.studio.muat_daftar()
                elif jenis == "hf_selesai":
                    self._sibuk = False
                    self.ringkas.config(text=f"Dataset terunggah: {isi}")
                    messagebox.showinfo("Hugging Face", f"Dataset terunggah:\n{isi}")
        except queue.Empty:
            pass
        if segarkan_strip:
            self._gambar_strip()
        self.after(50, self._poll)

    def _terima_muat(self, rek, semua, n_ekspor, catatan, waktu, tangga, kur, qa) -> None:
        lama = self._kini()
        self.rekaman, self.semua, self.n_ekspor = rek, semua, n_ekspor
        self.catatan, self.waktu, self.daftar_tangga, self.kur, self.qa = catatan, waktu, tangga, kur, qa
        self._t_kurasi = self._waktu_kurasi()
        self._hitung_frames()
        self._isi_tree()
        kini = getattr(getattr(self.studio, "sesi", None), "name", None)
        self._isi_daftar(pertahankan=lama.k if lama else None)
        if lama is None and kini in self.rekaman and self._sesi_awal != kini:
            self._sesi_awal = kini                       # rekaman yang dibuka di tab lain ikut terpilih
            self._lompat_ke_rekaman([kini])
        self._perbarui_ringkas()

    def _hitung_frames(self, rekaman=None) -> None:
        for r in (rekaman or self.rekaman):
            self.frames[r] = [f for f in self.semua.get(r, []) if self._status(f) == "pakai"]

    def _segarkan_semua(self) -> None:
        kini = self._kini()
        self._hitung_frames()
        self._perbarui_tree()
        self._isi_daftar(pertahankan=kini.k if kini else None)
        self._perbarui_ringkas()

    def _simpan(self) -> None:
        self.sd.tulis(self.berkas, self.data)

    # ------------------------------------------------------------ status frame
    def _set(self, f: Fr) -> str | None:
        return self.data["frame"].get(f.k) or self.data["rekaman"].get(f.r)

    def _status(self, f: Fr) -> str:
        """buang (dibuang manual), otomatis (disisihkan kurasi), atau pakai."""
        if f.sampah:
            return "buang"
        return "otomatis" if KU.dipilih(self.kur, f.k) is False else "pakai"

    def _qa(self, f: Fr) -> dict | None:
        """Hasil pemeriksaan label yang masih berlaku (label belum disunting sejak diperiksa)."""
        q = self.qa.get(f.k)
        if not q or q.get("versi", 1) != qa_label.VERSI or abs(f.mt - q.get("waktu_label", 0)) > 1:
            return None
        return q

    def _waktu_kurasi(self) -> float:
        try:
            return datetime.fromisoformat(self.kur.get("dibuat", "")).timestamp()
        except (TypeError, ValueError):
            return float("inf")

    def _alasan(self, f: Fr) -> list[str]:
        """Alasan frame perlu dicek, tanpa kembar: tanda CV di qa_label dan di kurasi berasal dari CV yang sama."""
        q = self._qa(f)
        if q and q.get("ok"):
            return []                                    # sudah dilihat dan labelnya dinyatakan benar (K)
        asli = list(q.get("alasan") or []) if q else []
        dari_cv = any(a.startswith(("SARAN BUANG", "PERIKSA")) for a in asli)
        out = [a.split(": ", 1)[-1] if a.startswith(("SARAN BUANG", "PERIKSA")) else a for a in asli]
        # Tanda kurasi menilai label saat kurasi dihitung; label yang sudah disunting sesudahnya tidak ditandai lagi.
        v = self.kur.get("frame", {}).get(f.k) if f.mt <= self._t_kurasi + 1 else None
        for a in (v or {}).get("alasan", []):
            if a.startswith("perlu dilihat"):
                isi = a.split(": ", 1)[-1]
                if not (dari_cv and "model CV" in isi) and isi not in out:
                    out.append(isi)
        return out

    def _saran(self, f: Fr) -> str | None:
        q = self._qa(f)
        return q.get("saran") if q and not q.get("ok") else None

    def _tampak(self, f: Fr) -> str:
        """Status untuk warna: pakai, cek (dipakai tetapi perlu dicek), buang, otomatis."""
        s = self._status(f)
        return "cek" if s == "pakai" and (self._saran(f) == "buang" or self._alasan(f)) else s

    # ------------------------------------------------------------ pohon rekaman
    def _scene(self, r: str) -> list[dict]:
        return (self.catatan.get(r) or {}).get("scene", [])

    def _frames_iid(self, iid: str) -> list[Fr]:
        """Frame (termasuk yang dibuang) untuk satu baris pohon: rekaman, scene, tangga, atau rekaman di tangga."""
        if iid.startswith("tg::"):
            bag = iid.split("::")
            tid = None if bag[1] == "-" else bag[1]
            rek = [bag[2]] if len(bag) > 2 else self.rekaman
            return [f for r in rek for f in self.semua.get(r, []) if f.tg == tid]
        if "::" not in iid:
            return self.semua.get(iid, [])
        r, i = iid.split("::")
        sc = self._scene(r)[int(i)]
        return [f for f in self.semua.get(r, []) if int(sc["awal"]) <= (f.i if f.i is not None else -1) <= int(sc["akhir"])]

    def _rekaman_iid(self, iid: str) -> list[str]:
        if iid.startswith("tg::"):
            return sorted({f.r for f in self._frames_iid(iid)} | {x.split("::")[-1] for x in self.tree.get_children(iid)})
        return [iid.split("::")[0]]

    def _teks_waktu(self, r: str | None) -> str:
        w = self.waktu.get(r) if r else None
        return f"{w.day} {CR.BULAN[w.month - 1]} {w:%H.%M} {CR.periode(w.hour)}" if w else ""

    def _nilai_baris(self, iid: str) -> tuple:
        fr = self._frames_iid(iid)
        pakai = [f for f in fr if self._status(f) == "pakai"]
        cek = sum(1 for f in pakai if self._saran(f) == "buang" or self._alasan(f))
        if "::" not in iid:
            teks_set, r = LABEL_SET[self.data["rekaman"].get(iid)], iid
        else:
            set_ = {self._set(f) for f in pakai}
            teks_set = LABEL_SET[next(iter(set_))] if len(set_) == 1 else ("campur" if set_ else "–")
            r = iid.split("::")[2] if iid.startswith("tg::") and iid.count("::") == 2 else None
        return (teks_set, len(pakai), (len(fr) - len(pakai)) or "", cek or "", self._teks_waktu(r))

    def _tag_rekaman(self, r: str) -> list[str]:
        c = self.catatan.get(r) or {}
        w = CR.warna_rekaman(c, self.daftar_tangga)
        return ([str(self.data["rekaman"].get(r))] + ([f"stabilo_{w}"] if w else [])
                + ([] if c.get("selesai") else ["belum_selesai"]))

    def _isi_tree(self) -> None:
        self._abaikan_pilih_pohon = True
        try:
            pilih = set(self.tree.selection())
            buka = {r for r in self.rekaman if self.tree.exists(r) and self.tree.item(r, "open")}
            self.tree.delete(*self.tree.get_children())
            if self.per_tangga.get():
                self._isi_tree_tangga()
            else:
                for r in self.rekaman:
                    c = self.catatan.get(r) or {}
                    teks = (("✔ " if c.get("selesai") else "") + r.replace("TANGGA_NAIK_", "")
                            + ("" if c.get("selesai") else "  · belum ✔"))
                    self.tree.insert("", "end", iid=r, text=teks, values=self._nilai_baris(r),
                                     tags=self._tag_rekaman(r), open=r in buka)
                    for i, sc in enumerate(self._scene(r)):
                        iid = f"{r}::{i}"
                        w_sc = CR.warna_untuk_tangga(sc.get("tangga"), self.daftar_tangga)
                        self.tree.insert(r, "end", iid=iid, text=f"↳ {sc['nama']} ({sc['awal']}–{sc['akhir']})",
                                         values=self._nilai_baris(iid),
                                         tags=("scene",) + ((f"stabilo_{w_sc}",) if w_sc else ()))
            ada = [r for r in pilih if self.tree.exists(r)]
            if ada:
                self.tree.selection_set(ada)
        finally:
            self.induk.after_idle(lambda: setattr(self, "_abaikan_pilih_pohon", False))

    def _isi_tree_tangga(self) -> None:
        """Pohon tangga fisik -> rekaman yang merekamnya."""
        isi: dict = {}
        for r in self.rekaman:
            for f in self.semua.get(r, []):
                isi.setdefault(f.tg, set()).add(r)
        for tid in sorted(k for k in isi if k) + ([None] if None in isi else []):
            iid = f"tg::{tid or '-'}"
            info = self.daftar_tangga.get(tid or "", {})
            atr = " · ".join(x for x in (info.get("lokasi"), info.get("lingkungan"), info.get("bahan")) if x)
            teks = (("🪜 " + CR.label_tangga(tid, self.daftar_tangga) + (f"  [{atr}]" if atr else ""))
                    if tid else "❔ Belum ada tangga")
            w_t = CR.warna_untuk_tangga(tid, self.daftar_tangga)
            tag_t = [f"stabilo_{w_t}"] if w_t else []
            self.tree.insert("", "end", iid=iid, text=teks, values=self._nilai_baris(iid), open=True, tags=tag_t)
            for r in sorted(isi[tid]):
                self.tree.insert(iid, "end", iid=f"{iid}::{r}", text=r.replace("TANGGA_NAIK_", ""),
                                 values=self._nilai_baris(f"{iid}::{r}"), tags=tag_t)

    def _perbarui_tree(self, rekaman: set[str] | None = None) -> None:
        """Perbarui angka baris pohon; bila ``rekaman`` diberikan, hanya baris rekaman itu (dan tangganya)."""
        for iid in self._semua_iid():
            if rekaman is not None and not (iid.startswith("tg::") and iid.count("::") == 1):
                r = iid.split("::")[2] if iid.startswith("tg::") else iid.split("::")[0]
                if r not in rekaman:
                    continue
            self.tree.item(iid, values=self._nilai_baris(iid))
            if "::" not in iid:
                self.tree.item(iid, tags=self._tag_rekaman(iid))

    def _semua_iid(self) -> list[str]:
        out = []
        for a in self.tree.get_children(""):
            out.append(a)
            out += list(self.tree.get_children(a))
        return out

    def _pilihan_berubah(self) -> None:
        if self._abaikan_pilih_pohon:
            return
        pilih = self.tree.selection()
        self._tampil_catatan()
        if len(pilih) == 1 and not (pilih[0].startswith("tg::") and pilih[0].count("::") == 1):
            r = pilih[0].split("::")[-1] if pilih[0].startswith("tg::") else pilih[0].split("::")[0]
            self.studio._sesi_dari_split = self.akar / r       # dipilih juga di tab lain saat tab berganti
            self._sesi_awal = r
        if pilih:
            self._lompat_ke_baris(pilih[0])

    def _lompat_ke_baris(self, iid: str) -> None:
        sasaran = {f.k for f in self._frames_iid(iid)}
        j = next((n for n, f in enumerate(self.daftar) if f.k in sasaran), None)
        if j is None:
            self.lbl_jumlah.config(text=f"{len(self.daftar)} frame · baris ini tidak punya frame '{self.filter.get()}'")
            return
        self._fokus(j, sinkron_pohon=False)

    def _lompat_ke_rekaman(self, rek: list[str]) -> None:
        j = next((n for n, f in enumerate(self.daftar) if f.r in rek), None)
        if j is not None:
            self._fokus(j)

    def _tampil_catatan(self) -> None:
        sel = self.tree.selection()
        if len(sel) != 1:
            self.info_rek.config(text="", bg=PANEL); return
        if sel[0].startswith("tg::") and sel[0].count("::") == 1:
            tid = None if sel[0] == "tg::-" else sel[0][4:]
            info = self.daftar_tangga.get(tid or "", {})
            rek = self.tree.get_children(sel[0])
            teks = (f"🪜 {CR.label_tangga(tid, self.daftar_tangga)}" + (f" — {info['lokasi']}" if info.get("lokasi") else "")
                    + f"\n{len(rek)} rekaman: " + ", ".join(x.split("::")[-1][-15:] for x in rek)) if tid else \
                "Frame yang belum diberi tangga. Isi Tangga fisik di tab Tinjau."
            self.info_rek.config(text=teks, bg=PANEL); return
        r = sel[0].split("::")[-1] if sel[0].startswith("tg::") else sel[0].split("::")[0]
        c = self.catatan.get(r) or {}
        teks = ("📍 " + c["catatan"]) if c.get("catatan") else "Belum ada catatan (isi di tab Tinjau)."
        if c.get("tangga"):
            teks = f"🪜 {CR.label_tangga(c['tangga'], self.daftar_tangga)}\n" + teks
        if not c.get("selesai"):
            teks = "Belum ✔ di tab Tinjau: frame rekaman ini tidak dipakai melatih.\n" + teks
        self.info_rek.config(text=teks, bg=CR.WARNA.get(CR.warna_rekaman(c, self.daftar_tangga)) or PANEL)

    # ------------------------------------------------------------ daftar frame (strip)
    def _isi_daftar(self, pertahankan: str | None = None) -> None:
        """Bangun daftar strip menurut filter; frame yang lalu diubah statusnya tetap di daftar (diwarnai)."""
        kini = self._kini()
        kunci = pertahankan if pertahankan is not None else (kini.k if kini else None)
        filt = self.filter.get()
        semua = [f for r in self.rekaman for f in self.semua.get(r, [])]
        skor = lambda f: -((self._qa(f) or {}).get("skor") or 0)
        if filt == "perlu dicek":
            daftar = [f for f in semua if self._status(f) == "pakai" and (self._saran(f) == "buang" or self._alasan(f))]
            daftar.sort(key=lambda f: (self._saran(f) != "buang", skor(f)))
        elif filt == "saran buang":
            daftar = sorted((f for f in semua if self._saran(f) == "buang"), key=skor)
        elif filt == "dibuang":
            daftar = [f for f in semua if self._status(f) != "pakai"]
        elif filt == "latar (tanpa mask)":
            daftar = [f for f in semua if not f.mask and self._status(f) == "pakai"]
        else:
            daftar = semua
        self.daftar = daftar
        j = next((n for n, f in enumerate(daftar) if f.k == kunci), None) if kunci else None
        massal = filt in ("perlu dicek", "saran buang", "dibuang") and bool(daftar)
        self.btn_massal.configure(text="♻ Pulihkan semua yang tampil" if filt == "dibuang" else "🗑 Buang semua yang tampil",
                                  state="normal" if massal else "disabled")
        self._awal = -1
        self._fokus(j if j is not None else 0, sinkron_pohon=j is None)

    def _tab_aktif(self) -> bool:
        induk = getattr(self, "induk", None)
        try:
            return induk is not None and self.studio.tabs.select() == str(induk)
        except (AttributeError, tk.TclError):
            return False

    def _kini(self) -> Fr | None:
        return self.daftar[self.i] if 0 <= self.i < len(self.daftar) else None

    def _fokus(self, i: int, sinkron_pohon: bool = True, pertahankan_pilih: bool = False) -> None:
        self.i = min(max(0, i), max(0, len(self.daftar) - 1))
        if not pertahankan_pilih:
            self.pilih = {self.i} if self.daftar else set()
            self._jangkar = self.i
        f = self._kini()
        if f is not None:
            if self._tab_aktif():
                # Hanya saat pemakai menelusuri di tab ini: tab Label lalu membuka frame yang sama. Dulu fokus awal
                # saat Studio dibuka ikut dikirim, sehingga tab Label pindah sendiri ke rekaman lain.
                self.studio._frame_dari_split = f.d
            if sinkron_pohon:
                self._sorot_pohon(f)
        self._gambar()
        self._gambar_strip()
        self._minta_muka()

    def _sorot_pohon(self, f: Fr) -> None:
        iid = f"tg::{f.tg or '-'}::{f.r}" if self.per_tangga.get() else f.r
        if not self.tree.exists(iid) or self.tree.selection() == (iid,):
            return
        self._abaikan_pilih_pohon = True
        try:
            self.tree.selection_set(iid)
            self.tree.see(iid)
            self._tampil_catatan()
            if self._tab_aktif():
                self.studio._sesi_dari_split = self.akar / f.r
            self._sesi_awal = f.r
        finally:
            self.induk.after_idle(lambda: setattr(self, "_abaikan_pilih_pohon", False))

    def geser(self, arah: int) -> None:
        # Menahan tombol mengirim tekan-ulang ~30x per detik; yang tiba < 40 ms sesudah frame terakhir
        # selesai digambar dibuang agar frame tidak terus bergeser setelah tombol dilepas.
        if time.monotonic() - self._geser_terakhir < 0.04 or not self.daftar:
            return
        try:
            self._fokus(self.i + arah)
        finally:
            self._geser_terakhir = time.monotonic()

    def _klik_strip(self, e, mode: str) -> None:
        j = self._slot_di(e.x)
        if j is None:
            return
        if mode == "rentang":
            a, b = sorted((self._jangkar, j))
            self.pilih = set(range(a, b + 1))
            self._fokus(j, pertahankan_pilih=True)
        elif mode == "balik":
            self.pilih ^= {j}
            self._fokus(j, pertahankan_pilih=True)
        else:
            self._fokus(j)
        self.strip.focus_set()

    def _n_slot(self) -> int:
        return max(3, (max(self.strip.winfo_width(), 200) - 6) // (TW + 8))

    def _slot_di(self, x: int) -> int | None:
        s = int((x - 4) // (TW + 8))
        j = self._awal + s
        return j if 0 <= s < self._n_slot() and 0 <= j < len(self.daftar) else None

    def _gambar_strip(self) -> None:
        c = self.strip
        c.delete("all")
        n = len(self.daftar)
        k = self._n_slot()
        if not n:
            c.create_text(10, (TH + 30) // 2, anchor="w", fill="#BBBBBB",
                          text="Tidak ada frame. Ganti 'Tampilkan', atau ✔ rekaman di tab Tinjau.")
            return
        if not (self._awal <= self.i < self._awal + k):
            self._awal = max(0, min(self.i - k // 3, n - k))
        cerah = bool(self.cerah.get())
        for s in range(k):
            j = self._awal + s
            if j >= n:
                break
            f = self.daftar[j]
            x = 4 + s * (TW + 8)
            tampak = self._tampak(f)
            warna = WARNA_STATUS[tampak]
            foto = self._kecil.get((f.d, cerah))
            if foto is not None:
                c.create_image(x + 2, 4, anchor="nw", image=foto)
            else:
                c.create_rectangle(x + 2, 4, x + 2 + TW, 4 + TH, fill="#3A312E", outline="")
                c.create_text(x + 2 + TW // 2, 4 + TH // 2, text="memuat…", fill="#999999")
            c.create_rectangle(x + 1, 3, x + 3 + TW, 5 + TH, outline=warna, width=4 if j == self.i else 2)
            if tampak in ("buang", "otomatis"):
                c.create_line(x + 4, 6, x + TW, 2 + TH, fill=warna, width=2)
            if j in self.pilih and len(self.pilih) > 1:
                c.create_rectangle(x + 5, 7, x + TW - 1, TH + 1, outline="#FFD54F", width=2)
            if j == self.i:
                c.create_rectangle(x - 1, 1, x + 5 + TW, 7 + TH, outline="#FFD54F", width=1)
            ket = {"buang": "DIBUANG", "otomatis": "otomatis", "cek": "BUANG?" if self._saran(f) == "buang" else "cek"}
            c.create_text(x + 2, TH + 9, anchor="nw", fill=warna if tampak != "pakai" else "#DDDDDD",
                          font=("Segoe UI", 8, "bold"),
                          text=f"{f.r[-6:]} {f.d.name[-6:]}" + (f"  {ket[tampak]}" if tampak in ket else ""))
        self._minta_kecil(range(self._awal, min(n, self._awal + 2 * k)), cerah)

    # ------------------------------------------------------------ pratinjau
    def _gambar(self) -> None:
        k = self.kanvas
        k.delete("all")
        f = self._kini()
        n = len(self.daftar)
        self.lbl_jumlah.config(text=f"{self.i + 1 if n else 0} / {n} frame"
                                    + (f" · {len(self.pilih)} terpilih" if len(self.pilih) > 1 else ""))
        if f is None:
            self.judul.config(text="Tidak ada frame pada tampilan ini"); self.lencana.config(text="", bg=PANEL)
            self.info_cek.config(text=""); return
        tampak = self._tampak(f)
        self.judul.config(text=f"{f.r.replace('TANGGA_NAIK_', '')} / {f.d.name}")
        self.lencana.config(text=f"{TEKS_STATUS[tampak]} · {LABEL_SET[self._set(f)]}", bg=WARNA_STATUS[tampak])
        alasan = "; ".join(self._alasan(f))
        if tampak == "buang":
            teks = "Dibuang: tidak dilatih. Del memulihkan."
        elif tampak == "otomatis":
            teks = ("Disisihkan otomatis (" + "; ".join(self.kur.get("frame", {}).get(f.k, {}).get("alasan", []))
                    + "): tidak dilatih. Del memulihkan bila frame ini memang perlu.")
        elif self._saran(f) == "buang":
            teks = f"🗑 Saran buang: {alasan}. Perbaiki labelnya (Enter → tab Label) bila bisa, atau Del untuk membuang."
        elif alasan:
            teks = f"🔎 Perlu dicek: {alasan}. Label benar → biarkan (→ frame berikutnya); salah → Enter; tak berguna → Del."
        else:
            teks = ""
        self.info_cek.config(text=teks)
        v = self._ambil_frame(f.d)
        if v is None:
            k.create_text(12, 12, anchor="nw", fill="white", text="Berkas frame tidak lengkap."); return
        rgb, sem = v
        cw, ch = max(1, k.winfo_width()), max(1, k.winfo_height())
        skala = min(cw / rgb.shape[1], ch / rgb.shape[0])
        ukuran = (max(1, int(rgb.shape[1] * skala)), max(1, int(rgb.shape[0] * skala)))
        out = cv2.resize(rgb, ukuran, interpolation=cv2.INTER_AREA if skala < 1 else cv2.INTER_LINEAR)
        if self.cerah.get():
            out = _cerahkan(out)
        if self.tampil_mask.get():
            out = _tumpang_mask(out, cv2.resize(sem, ukuran, interpolation=cv2.INTER_NEAREST))
        self._foto = ImageTk.PhotoImage(Image.fromarray(out))
        k.create_image(cw // 2, ch // 2, image=self._foto)
        x0, y0 = cw // 2 - ukuran[0] // 2, ch // 2 - ukuran[1] // 2
        if sem.any():
            luas = sem.size
            pita = f"riser {100 * (sem == 1).sum() / luas:.0f}% · tread {100 * (sem == 2).sum() / luas:.0f}%"
            warna = "#1E8E3E"
        else:
            pita, warna = "TANPA MASK · LATAR — pastikan memang tidak ada tangga", "#E8710A"
        if bool(sem.any()) != f.mask:
            pita, warna = pita + "  ⚠ mask PNG tidak cocok dengan poligon label", "#C5221F"
        t = k.create_text(x0 + 10, y0 + 8, anchor="nw", text=pita, fill="white", font=("Segoe UI", 10, "bold"))
        a, b, c, d = k.bbox(t)
        k.tag_raise(t, k.create_rectangle(a - 6, b - 3, c + 6, d + 3, fill=warna, outline=""))
        if tampak in ("buang", "otomatis"):
            k.create_rectangle(x0 + 3, y0 + 3, x0 + ukuran[0] - 3, y0 + ukuran[1] - 3,
                               outline=WARNA_STATUS[tampak], width=6)

    def _tampilan_berubah(self) -> None:
        self._gambar()
        self._gambar_strip()

    def _ambil_frame(self, d: Path):
        with self._kunci:
            v = self._cache_frame.get(d)
            if v is not None:
                self._cache_frame.move_to_end(d)
                return v
        v = _baca_frame(d)
        if v is not None:
            with self._kunci:
                self._cache_frame[d] = v
                while len(self._cache_frame) > 10:
                    self._cache_frame.popitem(last=False)
        return v

    # ------------------------------------------------------------ pekerja gambar latar
    def _minta_kecil(self, indeks, cerah: bool) -> None:
        tugas = [("kecil", self.daftar[j].d, cerah) for j in indeks if (self.daftar[j].d, cerah) not in self._kecil]
        with self._kunci:
            self._tugas = [t for t in self._tugas if t[0] == "muka"] + tugas
        if tugas:
            self._ada_tugas.set()

    def _minta_muka(self) -> None:
        """Baca lebih dulu frame sesudah/sebelum frame ini agar panah terasa seketika."""
        tugas = [("muka", self.daftar[j].d, None) for j in (self.i + 1, self.i + 2, self.i - 1)
                 if 0 <= j < len(self.daftar)]
        with self._kunci:
            self._tugas = tugas + [t for t in self._tugas if t[0] != "muka"]
        self._ada_tugas.set()

    def _pekerja(self) -> None:
        while True:
            self._ada_tugas.wait()
            with self._kunci:
                if not self._tugas:
                    self._ada_tugas.clear()
                    continue
                jenis, d, cerah = self._tugas.pop(0)
            try:
                if jenis == "muka":
                    self._ambil_frame(d)
                elif (d, cerah) not in self._kecil:
                    arr = _gambar_kecil(d, cerah)
                    if arr is not None:
                        self._q.put(("kecil", ((d, cerah), arr)))
            except Exception:                                # noqa: BLE001 -- gambar gagal: kotak tetap kosong
                pass

    # ------------------------------------------------------------ buang / pulihkan
    def _tulis_sampah(self, f: Fr, nilai: bool) -> None:
        p = f.d / "frame_state.json"
        try:
            st_ = json.loads(p.read_text()) if p.exists() else {}
        except (OSError, ValueError):
            st_ = {}
        st_["di_sampah"] = nilai
        st_["waktu_ubah_iso"] = datetime.now().isoformat(timespec="seconds")
        p.write_text(json.dumps(st_, indent=1, ensure_ascii=False), encoding="utf-8")
        f.sampah = nilai

    def _ubah(self, frames: list[Fr], buang: bool) -> None:
        kurasi_berubah = False
        for f in frames:
            if buang:
                if not f.sampah:
                    self._tulis_sampah(f, True)
            else:
                if f.sampah:
                    self._tulis_sampah(f, False)
                if KU.dipilih(self.kur, f.k) is False:   # disisihkan kurasi: keputusan manual "pakai"
                    self.kur.setdefault("manual", {})[f.k] = True
                    kurasi_berubah = True
        if kurasi_berubah:
            self._simpan_kurasi_nanti()
        rek = {f.r for f in frames}
        self._hitung_frames(rek)
        self._perbarui_tree(rek)
        self._jadwalkan_ringkas()

    def buang_pulihkan(self) -> None:
        """Del: buang frame terpilih; bila semuanya sudah dibuang, pulihkan."""
        if not self.daftar:
            return
        indeks = sorted(self.pilih) if len(self.pilih) > 1 else [self.i]
        frames = [self.daftar[j] for j in indeks if j < len(self.daftar)]
        pulihkan = all(self._status(f) != "pakai" for f in frames)
        self._ubah(frames, buang=not pulihkan)
        if len(frames) > 1:                              # tanpa popup konfirmasi: bisa dibatalkan dari pemberitahuan
            self._kabar_batal(f"{len(frames)} frame {'dipulihkan' if pulihkan else 'dibuang'}.", frames, pulihkan)
        else:
            self.ringkas.config(text=f"{frames[0].d.name} " + ("dipulihkan." if pulihkan else "dibuang (Del lagi memulihkan)."))
        if len(frames) == 1 and not pulihkan and self.i + 1 < len(self.daftar):
            self._fokus(self.i + 1)                      # lanjut ke frame berikutnya, seperti tab Label
        else:
            self._fokus(self.i, pertahankan_pilih=len(frames) > 1)

    def tandai_ok(self) -> None:
        """K: label frame terpilih sudah benar -> tanda "perlu dicek" hilang sampai labelnya disunting lagi."""
        if not self.daftar:
            return
        indeks = sorted(self.pilih) if len(self.pilih) > 1 else [self.i]
        frames = [self.daftar[j] for j in indeks if j < len(self.daftar)]
        for f in frames:
            self.qa[f.k] = {"versi": qa_label.VERSI, "skor": 0, "alasan": [], "waktu_label": f.mt, "ok": True,
                            "sumber": f"dicek manual {datetime.now():%Y-%m-%d}"}
        self._simpan_qa_nanti()
        self._perbarui_tree({f.r for f in frames})
        self._jadwalkan_ringkas()
        self.ringkas.config(text=f"{len(frames)} frame dinyatakan benar; tanda cek hilang sampai labelnya disunting.")
        if len(frames) == 1 and self.i + 1 < len(self.daftar):
            self._fokus(self.i + 1)
        else:
            self._fokus(self.i, pertahankan_pilih=len(frames) > 1)

    def _simpan_qa_nanti(self) -> None:
        if self._jadwal_qa is not None:
            self.induk.after_cancel(self._jadwal_qa)

        def simpan():
            self._jadwal_qa = None
            teks = json.dumps(self.qa, ensure_ascii=False)

            def tulis():
                tmp = self.berkas_qa.with_suffix(".json.tmp")
                tmp.write_text(teks, encoding="utf-8")
                tmp.replace(self.berkas_qa)
            threading.Thread(target=tulis, daemon=True).start()
        self._jadwal_qa = self.induk.after(800, simpan)

    def buang_semua_tampil(self) -> None:
        filt = self.filter.get()
        if filt not in ("perlu dicek", "saran buang", "dibuang"):
            return                                       # "semua"/"latar": terlalu luas untuk sekali tekan
        pulihkan = filt == "dibuang"
        frames = [f for f in self.daftar if (self._status(f) != "pakai") == pulihkan]
        if not frames:
            messagebox.showinfo("Pilah frame", "Tidak ada frame yang perlu diubah di tampilan ini."); return
        per_rek: dict[str, int] = {}
        for f in frames:
            per_rek[f.r[-15:]] = per_rek.get(f.r[-15:], 0) + 1
        rinci = ", ".join(f"{r[-6:]} {n}" for r, n in sorted(per_rek.items(), key=lambda x: -x[1])[:5])
        self._ubah(frames, buang=not pulihkan)
        self._fokus(self.i)
        self._kabar_batal(f"{len(frames)} frame '{filt}' {'dipulihkan' if pulihkan else 'dibuang'} ({rinci}). "
                          "Simpan sebagai versi data lewat ⋯ Lainnya bila perlu.", frames, pulihkan)

    def _kabar_batal(self, teks: str, frames: list[Fr], pulihkan: bool) -> None:
        def batal():
            self._ubah(frames, buang=pulihkan)
            self._fokus(self.i, pertahankan_pilih=True)
            self.studio.kabar(f"Dibatalkan: {len(frames)} frame dikembalikan seperti semula.")
        self.studio.kabar(teks, aksi=("↶ Batalkan", batal))

    def _simpan_kurasi_nanti(self) -> None:
        if self._jadwal_kurasi is not None:
            self.induk.after_cancel(self._jadwal_kurasi)

        def simpan():
            self._jadwal_kurasi = None
            data = {**self.kur, "manual": dict(self.kur.get("manual", {}))}
            threading.Thread(target=KU.tulis, args=(self.akar_data, data), daemon=True).start()
        self._jadwal_kurasi = self.induk.after(800, simpan)

    def periksa_ulang(self) -> None:
        """Hitung ulang kurasi (frame mirip, depth kosong, potongan mask, model CV) untuk semua frame siap latih."""
        if self._sibuk:
            return
        frames = [f.d for r in self.rekaman for f in self.semua.get(r, []) if not f.sampah]
        if not frames:
            messagebox.showinfo("Periksa ulang", "Tidak ada frame siap latih (rekaman ✔ dengan label terperiksa)."); return
        self.studio.kabar(f"Periksa ulang {len(frames)} frame berjalan di latar (beberapa menit); keputusan manual "
                          "tetap dipakai. Hasilnya muncul di 'perlu dicek'.")
        self._sibuk = True
        dice = KU.dice_cv_dari(akar_aplikasi() / BERKAS_CV)

        def kerja():
            try:
                data = KU.kurasi(self.akar_data, frames, dice,
                                 progres=lambda n, t: self._q.put(("status", f"Memeriksa ulang {n}/{t} frame…")))
                self._q.put(("kurasi_selesai", data))
            except Exception as e:                           # noqa: BLE001
                self._q.put(("galat", f"Periksa ulang gagal: {e}"))
        threading.Thread(target=kerja, daemon=True).start()

    # ------------------------------------------------------------ set train/val/test
    def atur_rekaman(self, s: str | None) -> None:
        """Set untuk baris terpilih di pohon; baris scene/tangga dicatat per frame di rentangnya."""
        pilih = list(self.tree.selection())
        if not pilih:
            messagebox.showinfo("Pilih rekaman", "Pilih satu atau beberapa rekaman/scene di pohon dahulu."); return
        for iid in pilih:
            if iid.startswith("tg::"):
                self._terapkan_set(self._frames_iid(iid), s)
            elif "::" in iid:
                r = iid.split("::")[0]
                for f in self._frames_iid(iid):
                    if s is None or s == self.data["rekaman"].get(r):
                        self.data["frame"].pop(f.k, None)
                    else:
                        self.data["frame"][f.k] = s
            elif s is None:
                self.data["rekaman"].pop(iid, None)
            else:
                self.data["rekaman"][iid] = s
        self._simpan()
        self._perbarui_tree()
        self._gambar()
        self._jadwalkan_ringkas()
        rek = sorted({r for iid in pilih for r in self._rekaman_iid(iid)})
        belum = [r for r in rek if not (self.catatan.get(r) or {}).get("selesai")]
        if belum:
            self.ringkas.config(text=f"Set disimpan. {len(belum)} rekaman belum ✔ di tab Tinjau (tidak dilatih).")

    def _terapkan_set(self, frames: list[Fr], s: str | None) -> None:
        """Rekaman yang seluruh framenya terpilih diberi set rekaman; selebihnya pengecualian per frame."""
        per_rek: dict[str, list[Fr]] = {}
        for f in frames:
            per_rek.setdefault(f.r, []).append(f)
        for r, fr in per_rek.items():
            if len(fr) == len(self.semua.get(r, [])):
                for f in fr:
                    self.data["frame"].pop(f.k, None)
                if s is None:
                    self.data["rekaman"].pop(r, None)
                else:
                    self.data["rekaman"][r] = s
            else:
                for f in fr:
                    if s is None or s == self.data["rekaman"].get(r):
                        self.data["frame"].pop(f.k, None)
                    else:
                        self.data["frame"][f.k] = s

    def _tombol(self, e):
        if self.studio.tabs.select() != str(self.induk):
            return None
        if isinstance(self.studio.focus_get(), (tk.Entry, tk.Spinbox, ttk.Entry, ttk.Combobox)):
            return None
        k = e.keysym
        if k == "Delete":
            self.buang_pulihkan(); return "break"
        if k in ("Return", "KP_Enter"):
            self.buka_di_label(); return "break"
        peta = {"1": "train", "2": "val", "3": "test", "BackSpace": None}
        if k in peta:
            self.atur_rekaman(peta[k]); return "break"
        if k in ("Right", "space"):
            self.geser(1); return "break"
        if k in ("Left", "d", "D"):                       # D = sebelumnya, sama dengan tab Label
            self.geser(-1); return "break"
        if k in ("Prior", "Next"):
            self.geser((-1 if k == "Prior" else 1) * self._n_slot()); return "break"
        if k in ("Home", "End"):
            self._fokus(0 if k == "Home" else len(self.daftar) - 1); return "break"
        if k in ("k", "K"):
            self.tandai_ok(); return "break"
        if k in ("m", "M", "c", "C"):
            var = self.tampil_mask if k in ("m", "M") else self.cerah
            var.set(not var.get())
            self._tampilan_berubah(); return "break"
        return None

    # ------------------------------------------------------------ ringkasan & kalkulator
    def _jadwalkan_ringkas(self) -> None:
        if self._jadwal_ringkas is not None:
            self.induk.after_cancel(self._jadwal_ringkas)
        self._jadwal_ringkas = self.induk.after(300, self._perbarui_ringkas)

    def _perbarui_ringkas(self) -> None:
        """Baris atas: berapa frame dipakai per set, berapa perlu dicek dan dibuang; peringatan hanya bila ada."""
        self._jadwal_ringkas = None
        per_set = {s: [] for s in self.sd.SET}
        belum = 0
        bocor: dict[str, set] = {}
        for r in self.rekaman:
            for f in self.frames.get(r, []):
                s = self._set(f)
                if s in per_set:
                    per_set[s].append(f.d)
                    if f.tg:
                        bocor.setdefault(f.tg, set()).add(s)
                elif s is None:
                    belum += 1
        rk = self.sd.ringkasan(per_set)
        semua = [f for r in self.rekaman for f in self.semua.get(r, [])]
        dibuang = sum(1 for f in semua if f.sampah)
        otomatis = sum(1 for f in semua if not f.sampah and self._status(f) == "otomatis")
        cek = sum(1 for r in self.rekaman for f in self.frames.get(r, []) if self._tampak(f) == "cek")
        n = sum(len(v) for v in self.frames.values())
        bagian = " · ".join(f"{LABEL_SET[s]} {rk['frame'][s]:,}" for s in self.sd.SET).replace(",", ".")
        self.lbl_atas.config(text=f"Dipakai melatih {n:,} frame ({bagian}".replace(",", ".")
                                  + (f" · belum dibagi {belum}" if belum else "") + f")   ·   perlu dicek {cek}"
                                  + f"   ·   dibuang {dibuang} + {otomatis} otomatis")
        peringatan = []
        if rk["rekaman_terbagi"]:
            peringatan.append(f"⚠ {len(rk['rekaman_terbagi'])} rekaman terbagi ke beberapa set (frame berdekatan hampir "
                              "identik, angka val/test terlalu optimistis): "
                              + ", ".join(r[-6:] for r in rk["rekaman_terbagi"]))
        bocor = {t: v for t, v in bocor.items() if len(v) > 1}
        if bocor:
            peringatan.append("⚠ Tangga yang sama ada di lebih dari satu set: "
                              + "; ".join(f"{t} di {'/'.join(sorted(v))}" for t, v in sorted(bocor.items())))
        if n and not rk["frame"]["val"]:
            peringatan.append("⚠ Set VAL kosong: perintah latih memakai --pelatihan-penuh (tanpa validasi).")
        self.lbl_peringatan.config(text="\n".join(peringatan))
        self._perbarui_kalkulator(rk, belum)

    def _rasio(self) -> dict[str, float]:
        try:
            r = {"train": int(self.rasio_train.get()), "val": int(self.rasio_val.get()), "test": int(self.rasio_test.get())}
        except (tk.TclError, ValueError):
            r = {"train": 70, "val": 15, "test": 15}
        tot = sum(r.values()) or 1
        return {k: v / tot for k, v in r.items()}

    def _atur_rasio(self, tr: int, va: int, te: int) -> None:
        self.rasio_train.set(tr); self.rasio_val.set(va); self.rasio_test.set(te)
        self._rasio_berubah()

    def _rasio_berubah(self) -> None:
        if self.rekaman:
            self._jadwalkan_ringkas()

    def _grup(self) -> list[tuple[str, list[Fr]]]:
        """Satuan pembagian: tangga fisik lintas rekaman, lalu scene, lalu sisa rekaman."""
        grup = []
        per_tangga: dict[str, list[Fr]] = {}
        for r in self.rekaman:
            if self.data["rekaman"].get(r) == "abaikan":
                continue
            for f in self.frames.get(r, []):
                if f.tg:
                    per_tangga.setdefault(f.tg, []).append(f)
        grup += [(f"tg::{t}", fr) for t, fr in sorted(per_tangga.items())]
        bertangga = {f.k for fr in per_tangga.values() for f in fr}
        for r in self.rekaman:
            if self.data["rekaman"].get(r) == "abaikan" or not self.frames.get(r):
                continue
            pakai = {f.k for f in self.frames[r]}
            sisa = [f for f in self.frames[r] if f.k not in bertangga]
            if not sisa:
                continue
            for i, _sc in enumerate(self._scene(r)):
                isi = [f for f in self._frames_iid(f"{r}::{i}") if f.k in pakai and f.k not in bertangga]
                if isi:
                    grup.append((f"{r}::{i}", isi))
                    ambil = {f.k for f in isi}
                    sisa = [f for f in sisa if f.k not in ambil]
            if sisa:
                grup.append((r if len(sisa) == len(self.frames[r]) else f"{r}::sisa", sisa))
        return grup

    def _rencana_grup(self) -> dict[str, str]:
        """Serakah: grup terbesar lebih dulu ke set yang paling jauh di bawah targetnya;
        val dan test yang rasionya > 0 dijamin kebagian minimal satu grup bila grup cukup."""
        rasio = self._rasio()
        grup = sorted(self._grup(), key=lambda g: -len(g[1]))
        n = sum(len(g[1]) for g in grup)
        target = {k: rasio[k] * n for k in rasio}
        isi = {k: 0 for k in rasio}
        rencana: dict[str, str] = {}
        wajib = [k for k in ("val", "test") if rasio[k] > 0]
        for gid, fr in grup:
            k = max(target, key=lambda k: (target[k] - isi[k]) / max(target[k], 1e-9) if target[k] > 0 else -1e9)
            rencana[gid] = k
            isi[k] += len(fr)
        if len(grup) > len(wajib):
            for k in wajib:
                if isi[k] == 0:
                    calon = [g for g in sorted(grup, key=lambda g: len(g[1])) if rencana[g[0]] == "train"]
                    if len(calon) > 1:
                        rencana[calon[0][0]] = k
                        isi[k] += len(calon[0][1]); isi["train"] -= len(calon[0][1])
        return rencana

    def _perbarui_kalkulator(self, rk: dict, belum: int) -> None:
        for w in self.tabel.winfo_children():
            w.destroy()
        rasio = self._rasio()
        n = sum(rk["frame"].values()) + belum
        for j, t in enumerate(("Set", "Target %", "Target frame", "Sekarang", "Sekarang %", "Rekaman")):
            tk.Label(self.tabel, text=t, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).grid(
                row=0, column=j, sticky="e" if j else "w", padx=4)
        for i, k in enumerate(self.sd.SET, 1):
            sek = rk["frame"][k]
            nilai = (LABEL_SET[k], f"{rasio[k] * 100:.0f}%", f"{round(rasio[k] * n)}", f"{sek}",
                     f"{(sek / n * 100 if n else 0):.0f}%", f"{rk['rekaman'][k]}")
            for j, t in enumerate(nilai):
                tk.Label(self.tabel, text=t, bg=PANEL, fg=WARNA_SET[k] if j == 0 else INK,
                         font=("Segoe UI", 9, "bold" if j == 0 else "normal")).grid(
                    row=i, column=j, sticky="e" if j else "w", padx=4)
        tk.Label(self.tabel, text=f"Total dipakai: {n} frame   •   belum dibagi: {belum}", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).grid(row=4, column=0, columnspan=6, sticky="w", padx=4, pady=(2, 0))
        g = len(self._grup())
        n_tangga = len({f.tg for r in self.rekaman for f in self.frames.get(r, [])} - {None})
        saran = [f"Dari {n} frame dengan rasio ini: train {round(rasio['train'] * n)}, "
                 f"val {round(rasio['val'] * n)}, test {round(rasio['test'] * n)}."]
        if n_tangga:
            saran.append(f"{n_tangga} tangga fisik terdaftar: bagi otomatis memakai tangga sebagai satuan, "
                         "sehingga tangga yang sama tidak muncul di train dan test.")
        if g < 5:
            saran.append(f"Hanya {g} rekaman/scene: pembagian sulit tepat rasio. Tandai scene di tab Tinjau "
                         "untuk memecah rekaman.")
        self.rekomendasi.config(text="\n".join(saran))

    def bagi_otomatis(self, _mode: str = "grup") -> None:
        if not self.rekaman:
            return
        rasio = self._rasio()
        rencana = self._rencana_grup()
        hitung = {k: 0 for k in self.sd.SET}
        for gid, fr in self._grup():
            hitung[rencana[gid]] += len(fr)
        n = sum(hitung.values())
        if not n:
            messagebox.showinfo("Bagi otomatis", "Tidak ada frame yang dipakai."); return
        ringkas = " · ".join(f"{LABEL_SET[k]} {hitung[k]} ({hitung[k] / n * 100:.0f}%)" for k in self.sd.SET)
        lama = {"rekaman": dict(self.data["rekaman"]), "frame": dict(self.data["frame"])}
        per_frame = {f.k: (f, rencana[gid]) for gid, fr in self._grup() for f in fr}
        per_rek: dict[str, dict[str, int]] = {}
        for f, k in per_frame.values():
            per_rek.setdefault(f.r, {}); per_rek[f.r][k] = per_rek[f.r].get(k, 0) + 1
        for r, n_set in per_rek.items():
            self.data["rekaman"][r] = max(n_set, key=n_set.get)       # set mayoritas jadi set rekaman
            for f in self.semua.get(r, []):
                self.data["frame"].pop(f.k, None)
        for f, k in per_frame.values():
            if k != self.data["rekaman"][f.r]:
                self.data["frame"][f.k] = k
        self._simpan()
        self._isi_tree()
        self._gambar()
        self._perbarui_ringkas()

        def batal():
            self.data["rekaman"], self.data["frame"] = lama["rekaman"], lama["frame"]
            self._simpan()
            self._isi_tree()
            self._gambar()
            self._perbarui_ringkas()
            self.studio.kabar("Pembagian dikembalikan seperti sebelum Bagi otomatis.")
        self.studio.kabar(f"Dibagi otomatis: {ringkas}.", aksi=("↶ Batalkan", batal))

    # ------------------------------------------------------------ aksi lain
    def buka_di_label(self) -> None:
        """Buka frame yang dilihat di tab Label untuk diperbaiki."""
        f = self._kini()
        if f is None:
            return
        st = self.studio
        st.sesi = f.d.parent.parent.parent
        st._muat_catatan_ui()
        st.tabs.select(st.tab_label)
        st.muat_frame()
        if f.d in st.frame_paths:
            st._pilih_indeks_frame(st.frame_paths.index(f.d))

    def buka_versi(self) -> None:
        """Jendela versi data: potret dataset latih (v1, v2, ...) dan hasil CV tiap versi (jendela_versi.py)."""
        try:
            from .jendela_versi import JendelaVersi
        except ImportError:
            from jendela_versi import JendelaVersi
        lama = getattr(self, "_jendela_versi", None)
        if lama is not None and lama.winfo_exists():
            lama.lift(); return
        self._jendela_versi = JendelaVersi(self.studio)

    def _perintah_latih(self) -> str:
        app = akar_aplikasi()
        per_val = sum(1 for r in self.rekaman for f in self.frames.get(r, []) if self._set(f) == "val")
        awal = app / "runs" / "latih_ulang_384" / "pra_latih" / "best.pt"
        bagian = [f'cd "{app}" &&', "./jalankan.sh", f'"{sys.executable}"', "-m rgbd_convnext.latih.d435",
                  f'--akar "{self.akar}"', f'--split "{self.berkas}"']
        if (self.akar_data / "kurasi.json").exists():
            bagian.append(f'--kurasi "{self.akar_data / "kurasi.json"}"')
        if self.hanya_diperiksa.get():
            bagian.append("--hanya-diperiksa")
        if not per_val:
            bagian.append("--pelatihan-penuh")
        if awal.exists():
            bagian.append(f'--bobot-awal "{awal.relative_to(app)}"')
        bagian += [f"--keluar runs/latih_split_{datetime.now():%Y%m%d_%H%M}", "--epochs 60", "--ukuran 384"]
        return " ".join(bagian)

    def _salin(self, teks: str, judul: str) -> None:
        self.studio.clipboard_clear(); self.studio.clipboard_append(teks); self.studio.update_idletasks()
        self.studio.kabar(f"{judul} disalin ke papan klip; tempel di terminal.")

    def salin_latih(self) -> None:
        self._salin(self._perintah_latih(), "Perintah latih")

    def salin_uji(self) -> None:
        app = akar_aplikasi()
        teks = (f'cd "{app}" && ./jalankan.sh "{sys.executable}" skrip/utama/evaluasi_segmentasi_akhir.py '
                f'--akar "{self.akar}" --split "{self.berkas}" --set test'
                + (" --hanya-diperiksa" if self.hanya_diperiksa.get() else "")
                + ' --bobot <folder-hasil-latih>/best.pt --keluar <folder-hasil-latih>/uji_test.json')
        self._salin(teks, "Perintah uji")

    def ringkasan_csv(self) -> None:
        """Tabel per rekaman untuk naskah: waktu, lokasi, lux, kecerahan gambar, jumlah frame per set,
        ukuran berkas, dan hasil cek sinkron RGB-depth-IR."""
        if self._sibuk or not self.rekaman:
            return
        self._sibuk = True
        rekaman, n_ekspor, catatan = list(self.rekaman), dict(self.n_ekspor), dict(self.catatan)
        per_rek = {r: {k: sum(1 for f in self.frames.get(r, []) if self._set(f) == k) for k in ("train", "val", "test")}
                   for r in rekaman}
        n_pakai = {r: len(self.frames.get(r, [])) for r in rekaman}
        set_rek = dict(self.data["rekaman"])
        tangga = dict(self.daftar_tangga)
        tujuan = self.akar_data / "ringkasan_rekaman.csv"

        def kerja():
            import csv
            try:
                baris, waktu_semua, periode_hitung = [], [], {}
                for no, r in enumerate(rekaman, 1):
                    self._q.put(("status", f"Menyusun ringkasan: {no}/{len(rekaman)} rekaman"))
                    sesi = self.akar / r
                    w = CR.waktu_rekaman(sesi)
                    mulai, selesai = w if w else (None, None)
                    semua_ekspor = sorted(d for d in (sesi / "exports" / "frames").glob("frame_*") if d.is_dir()) \
                        if (sesi / "exports" / "frames").exists() else []
                    terang = []
                    for d in semua_ekspor[::max(1, len(semua_ekspor) // 20)][:20]:
                        g = cv2.imread(str(d / "color_raw.png"))
                        if g is not None:
                            terang.append(CR.kecerahan(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))[0])
                    per = per_rek[r]
                    c = catatan.get(r) or {}
                    u = CR.ukuran_rekaman(sesi)
                    sk = sinkron.baca(sesi).get("ringkasan") or {}
                    mb = lambda b: round(b / 1e6, 1)
                    if mulai:
                        waktu_semua.append(mulai)
                        periode_hitung[CR.periode(mulai.hour)] = periode_hitung.get(CR.periode(mulai.hour), 0) + 1
                    baris.append({
                        "rekaman": r, "tanggal": f"{mulai:%Y-%m-%d}" if mulai else "",
                        "jam_mulai": f"{mulai:%H:%M:%S}" if mulai else "", "jam_selesai": f"{selesai:%H:%M:%S}" if selesai else "",
                        "durasi_detik": int((selesai - mulai).total_seconds()) if mulai and selesai else "",
                        "periode": CR.periode(mulai.hour) if mulai else "", "lokasi_catatan": c.get("catatan", ""),
                        "lux_terukur": c.get("lux", ""), "sumber_lux": c.get("lux_sumber", ""),
                        "kecerahan_gambar_median_0_255": round(float(np.median(terang)), 1) if terang else "",
                        "frame_ekspor": n_ekspor.get(r, 0), "frame_siap_latih": n_pakai[r],
                        "train": per["train"], "val": per["val"], "test": per["test"],
                        "set_rekaman": set_rek.get(r, ""), "scene": len(c.get("scene", [])),
                        "tangga": c.get("tangga", ""), "selesai": "ya" if c.get("selesai") else "",
                        "nama_tangga": (tangga.get(c.get("tangga", "")) or {}).get("nama", ""),
                        "tangga_scene": ";".join(sorted({sc.get("tangga") for sc in c.get("scene", []) if sc.get("tangga")})),
                        "ukuran_total_mb": mb(u["total"][0]), "mentah_mb": mb(u["mentah"][0]),
                        "turunan_mb": mb(u["turunan"][0]), "ekspor_mb": mb(u["ekspor"][0]),
                        "jumlah_berkas": u["total"][1],
                        "ekspor_mb_per_frame": mb(u["ekspor"][0] / u["frame_ekspor"]) if u["frame_ekspor"] else "",
                        "sinkron_frame_diperiksa": sk.get("frame", ""),
                        "sinkron_satu_jepretan": sk.get("satu_jepretan", ""),
                        "selisih_rgb_depth_maks_ms": sk.get("selisih_maks_ms", ""),
                    })
                with tujuan.open("w", newline="", encoding="utf-8") as f:
                    wtr = csv.DictWriter(f, fieldnames=list(baris[0]))
                    wtr.writeheader(); wtr.writerows(baris)
                ket = ""
                if waktu_semua:
                    a, b = min(waktu_semua), max(waktu_semua)
                    ket = (f"{len(rekaman)} rekaman, {a:%d-%m-%Y} s.d. {b:%d-%m-%Y}; "
                           f"jam mulai {min(w.strftime('%H.%M') for w in waktu_semua)}-"
                           f"{max(w.strftime('%H.%M') for w in waktu_semua)}; "
                           + ", ".join(f"{k} {v}" for k, v in sorted(periode_hitung.items())))
                self._q.put(("csv_selesai", (str(tujuan), ket)))
            except Exception as e:                           # noqa: BLE001
                self._q.put(("galat", f"Ringkasan gagal: {e}"))
        threading.Thread(target=kerja, daemon=True).start()

    def tarik_hf(self) -> None:
        """Pulihkan frame ekspor + label dari HF agar bisa dilabel ulang tanpa video mentah."""
        if not unggah_hf.konfigurasi()["token"] and not masuk_hf(self.studio):
            return
        if self._sibuk:
            return
        k = unggah_hf.konfigurasi()
        if not messagebox.askyesno("Tarik dataset", f"Unduh dataset dari huggingface.co/datasets/{k['repo']}\n"
                                   "dan pulihkan frame ekspornya di laptop ini?\n\n"
                                   "Gambar yang sudah ada tidak ditimpa. Label lokal yang lebih baru\n"
                                   "dipertahankan; label dari HF yang lebih baru dipakai."):
            return
        self._sibuk = True

        def kerja():
            try:
                self._q.put(("tarik_selesai", unggah_hf.tarik(self.akar, self.berkas,
                                                               lambda t: self._q.put(("status", t)))))
            except Exception as e:                           # noqa: BLE001
                self._q.put(("galat", f"Tarik dari Hugging Face gagal: {e}"))
        threading.Thread(target=kerja, daemon=True).start()

    def push_hf(self) -> None:
        k = unggah_hf.konfigurasi()
        if not k["token"]:
            if not masuk_hf(self.studio):
                return
            k = unggah_hf.konfigurasi()
        if self._sibuk:
            return
        if not messagebox.askyesno("Push dataset", f"Unggah seluruh frame ekspor + label (Parquet) ke\n"
                                                   f"huggingface.co/datasets/{k['repo']}?\n\n"
                                                   f"Isi repo akan diganti dengan keadaan dataset sekarang."
                                                   + ("\nRepo baru dibuat PRIVAT." if k["privat"] else "")):
            return
        self._sibuk = True

        def kerja():
            try:
                url = unggah_hf.push(self.akar, self.berkas, lambda t: self._q.put(("status", t)))
                self._q.put(("hf_selesai", url))
            except Exception as e:                           # noqa: BLE001
                self._q.put(("galat", f"Push ke Hugging Face gagal: {e}"))
        threading.Thread(target=kerja, daemon=True).start()


def masuk_hf(induk) -> bool:
    """Dialog masuk Hugging Face untuk laptop yang belum punya token.

    Token diperiksa ke Hugging Face lalu disimpan di penyimpanan kredensial HF
    (``~/.cache/huggingface/token``), di luar folder proyek, jadi tidak pernah
    ikut Git. Cukup sekali per laptop.
    """
    dlg = tk.Toplevel(induk); dlg.title("Masuk Hugging Face"); dlg.configure(bg=PANEL)
    dlg.transient(induk); dlg.resizable(False, False)
    hasil = {"ok": False}
    tk.Label(dlg, bg=PANEL, fg=INK, justify="left", wraplength=440, text=(
        "Laptop ini belum masuk Hugging Face.\n\n"
        "1. Tekan tombol di bawah: browser membuka halaman pembuatan token (masuk akun bila diminta).\n"
        "2. Beri nama bebas, pilih jenis Write, tekan Create token, lalu salin tokennya.\n"
        "3. Tempel di kotak ini dan tekan Simpan.\n\n"
        "Token disimpan di ~/.cache/huggingface/token (di luar folder proyek), tidak ikut Git.")
             ).pack(padx=16, pady=(14, 8), anchor="w")
    tk.Button(dlg, text="🌐 Buka halaman token Hugging Face",
              command=lambda: webbrowser.open(unggah_hf.URL_TOKEN_BARU)).pack(padx=16, fill="x")
    isian = tk.Entry(dlg, show="•", width=52); isian.pack(padx=16, pady=10, fill="x"); isian.focus_set()
    info = tk.Label(dlg, text="", bg=PANEL, fg="#AA5A55", wraplength=440, justify="left"); info.pack(padx=16, anchor="w")

    def simpan(_e=None):
        t = isian.get().strip()
        if not t.startswith("hf_"):
            info.config(text="Token Hugging Face diawali hf_."); return
        dlg.config(cursor="watch"); dlg.update()
        try:
            nama = unggah_hf.simpan_token(t)
        except Exception as e:                               # noqa: BLE001
            dlg.config(cursor=""); info.config(text=f"Token ditolak: {e}"); return
        hasil["ok"] = True
        messagebox.showinfo("Hugging Face", f"Berhasil masuk sebagai {nama}.", parent=dlg)
        dlg.destroy()
    isian.bind("<Return>", simpan)
    baris = tk.Frame(dlg, bg=PANEL); baris.pack(fill="x", padx=16, pady=(6, 14))
    tk.Button(baris, text="Simpan & lanjut", command=simpan).pack(side="right")
    tk.Button(baris, text="Batal", command=dlg.destroy).pack(side="right", padx=6)
    jadikan_modal(dlg, induk)
    isian.focus_set()
    dlg.wait_window()
    return hasil["ok"]
