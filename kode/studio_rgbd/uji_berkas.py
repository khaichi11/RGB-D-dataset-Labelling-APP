"""Panel uji model pada berkas: gambar, video, atau frame ekspor RGB-D.

Berbeda tujuan dari panel uji realtime. Panel itu menguji model pada aliran
kamera langsung, sedangkan panel ini menguji pada berkas yang sudah ada,
sehingga hasilnya dapat diulang persis dan dibandingkan antar-model.

Tiga jenis masukan didukung, dan perbedaannya penting:

  frame ekspor RGB-D   punya kedalaman asli, jadi model bekerja penuh
  rekaman .bag         sama, dibaca berurutan lewat pyrealsense2
  gambar / video biasa TANPA kedalaman

Masukan ketiga tetap dilayani karena model dilatih dengan depth dropout 0,2,
yaitu sebagian batch sengaja dijalankan tanpa kedalaman sama sekali. Model
karena itu tidak runtuh ketika kedalaman hilang, hanya kehilangan sebagian
ketepatan. Keadaan itu ditandai jelas pada tampilan agar hasilnya tidak
disalahartikan sebagai kinerja penuh.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

BG, PANEL, INK, MUTED, ACCENT, LINE = "#F7F3F0", "#FFFFFF", "#2B2622", "#7A6E66", "#C1613C", "#E4DAD3"
HIJAU, MERAH = (60, 200, 60), (70, 70, 235)          # BGR: tapakan, bidang tegak
MIN_M, MAKS_M, UKURAN = 0.2, 4.0, 512
BG_K, RISER, TREAD = 0, 1, 2


class UjiBerkas:
    """Panel pengujian model terlatih pada berkas gambar, video, atau rekaman."""

    def __init__(self, induk: tk.Frame, studio) -> None:
        self.induk, self.studio = induk, studio
        self.model = None
        self.nama_bobot = tk.StringVar(value="belum dimuat")
        self.jalur_bobot: Path | None = None
        self.sumber = tk.StringVar(value="belum ada berkas dipilih")
        self.jenis = tk.StringVar(value="-")
        self.info = tk.StringVar(value="Pilih bobot lalu pilih berkas untuk diuji.")
        self.alpha = tk.DoubleVar(value=0.45)
        self.jalan = False
        self._frames: list = []
        self._idx = 0
        self._thread = None
        self._bangun()

    # ------------------------------------------------------------- tata letak
    def _kartu(self, induk, judul):
        luar = tk.Frame(induk, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        luar.pack(fill="x", pady=(0, 10))
        tk.Label(luar, text=judul, bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(9, 2))
        dalam = tk.Frame(luar, bg=PANEL); dalam.pack(fill="x", padx=12, pady=(0, 10))
        return dalam

    def _tombol(self, induk, teks, perintah, warna=ACCENT, fg="#FFFFFF"):
        b = tk.Button(induk, text=teks, command=perintah, bg=warna, fg=fg, relief="flat",
                      font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10, pady=6,
                      activebackground=warna, activeforeground=fg, bd=0)
        return b

    def _bangun(self) -> None:
        kiri = tk.Frame(self.induk, bg=BG, width=340); kiri.pack(side="left", fill="y", padx=(16, 8), pady=12)
        kiri.pack_propagate(False)
        kanan = tk.Frame(self.induk, bg=BG); kanan.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=12)

        k = self._kartu(kiri, "1. Bobot model")
        tk.Label(k, textvariable=self.nama_bobot, bg=PANEL, fg=MUTED, wraplength=300,
                 justify="left").pack(anchor="w", pady=(0, 6))
        self._tombol(k, "Muat bobot terbaik otomatis", self.muat_otomatis).pack(fill="x", pady=(0, 4))
        self._tombol(k, "Pilih berkas bobot…", self.pilih_bobot, "#6E8CA8").pack(fill="x")

        k = self._kartu(kiri, "2. Berkas yang diuji")
        tk.Label(k, textvariable=self.sumber, bg=PANEL, fg=MUTED, wraplength=300,
                 justify="left").pack(anchor="w", pady=(0, 6))
        self._tombol(k, "Gambar…", self.pilih_gambar, "#7FA96B").pack(fill="x", pady=(0, 4))
        self._tombol(k, "Video…", self.pilih_video, "#7FA96B").pack(fill="x", pady=(0, 4))
        self._tombol(k, "Folder frame ekspor RGB-D…", self.pilih_folder, "#7FA96B").pack(fill="x")
        self._tombol(k, "Pasangkan kedalaman manual…", self.pilih_depth_manual, "#6E8CA8").pack(fill="x", pady=(4, 0))
        tk.Label(k, textvariable=self.jenis, bg=PANEL, fg=ACCENT, wraplength=300,
                 justify="left", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(6, 0))

        k = self._kartu(kiri, "3. Jalankan")
        self.tombol_main = self._tombol(k, "▶  Jalankan", self.toggle)
        self.tombol_main.pack(fill="x", pady=(0, 6))
        tk.Scale(k, from_=0, to=0.85, resolution=.05, orient="horizontal", variable=self.alpha,
                 label="Opasitas mask", bg=PANEL, fg=INK, highlightthickness=0).pack(fill="x")
        self._tombol(k, "Simpan bingkai saat ini…", self.simpan_bingkai, "#6E8CA8").pack(fill="x", pady=(6, 0))

        tk.Label(kiri, textvariable=self.info, bg=BG, fg=MUTED, wraplength=310,
                 justify="left").pack(anchor="w", pady=(4, 0))

        self.kanvas = tk.Label(kanan, bg="#181513"); self.kanvas.pack(fill="both", expand=True)
        self.bar = ttk.Progressbar(kanan, mode="determinate"); self.bar.pack(fill="x", pady=(8, 0))

    # --------------------------------------------------------------- bobot
    def muat_otomatis(self) -> None:
        try:
            from .segmentasi_convnext_depth import cari_bobot
        except ImportError:
            from studio_rgbd.segmentasi_convnext_depth import cari_bobot
        try:
            nama, jalur = cari_bobot()
        except FileNotFoundError as e:
            messagebox.showinfo("Bobot belum ada", str(e), parent=self.induk); return
        self._muat(jalur, nama)

    def pilih_bobot(self) -> None:
        f = filedialog.askopenfilename(title="Pilih bobot .pt",
                                       filetypes=[("Checkpoint PyTorch", "*.pt")])
        if f:
            self._muat(Path(f), Path(f).stem)

    def _muat(self, jalur: Path, nama: str) -> None:
        try:
            import sys, torch
            akar = Path(__file__).resolve().parents[1]
            if str(akar) not in sys.path:
                sys.path.insert(0, str(akar))
            from stair_fusion_atto.model_kandidat import (StairFusionAttoKandidat,
                                                          stride_dari_checkpoint,
                                                          varian_dari_checkpoint)
            ckpt = torch.load(jalur, map_location='cpu', weights_only=False)
            self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            m = StairFusionAttoKandidat(line_kernel=tuple(ckpt.get('kernel', (5, 5))),
                                        varian=varian_dari_checkpoint(ckpt),
                                        semantic_stride=stride_dari_checkpoint(ckpt),
                                        timm_pretrained=False).to(self.dev).eval()
            m.load_state_dict(ckpt['model'])
            self.model, self.jalur_bobot = m, jalur
            n = sum(p.numel() for p in m.parameters()) / 1e6
            self.nama_bobot.set(f"{nama}\n{n:.2f} juta parameter · {self.dev.type}")
            self.info.set("Bobot dimuat. Pilih berkas yang akan diuji.")
        except Exception as e:                                    # noqa: BLE001
            messagebox.showerror("Gagal memuat bobot", str(e), parent=self.induk)

    # -------------------------------------------------------------- sumber
    def _cari_depth(self, gambar: Path):
        """Cari berkas kedalaman yang berpasangan dengan sebuah gambar.

        Frame ekspor studio menyimpan color_raw.png dan
        depth_aligned_to_color.npy berdampingan dalam satu folder, sehingga
        memilih gambarnya saja sudah cukup untuk menemukan kedalamannya.
        Pola bernama juga dicoba agar berkas dari sumber lain tetap terlayani.
        """
        import json
        d = gambar.parent
        kandidat = [d / 'depth_aligned_to_color.npy',
                    d / f'{gambar.stem}_depth.npy',
                    d / f'{gambar.stem}.npy',
                    d / 'depth_aligned_to_color.png',
                    d / f'{gambar.stem}_depth.png']
        for c in kandidat:
            if not c.exists():
                continue
            if c.suffix == '.npy':
                z = np.load(c)
            else:
                z = cv2.imread(str(c), cv2.IMREAD_UNCHANGED)
                if z is None:
                    continue
            skala = 0.001
            meta = d / 'frame.json'
            if meta.exists():
                skala = float(json.loads(meta.read_text()).get('depth_scale', 0.001))
            return z.astype(np.float32) * skala, c.name
        return None, None

    def pilih_gambar(self) -> None:
        f = filedialog.askopenfilename(title="Pilih gambar",
                                       filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not f:
            return
        bgr = cv2.imread(f)
        if bgr is None:
            messagebox.showerror("Gagal membaca", f, parent=self.induk); return
        dm, nama_depth = self._cari_depth(Path(f))
        if dm is not None and dm.shape[:2] != bgr.shape[:2]:
            # Ukuran tidak cocok berarti pasangannya keliru; lebih baik ditolak
            # daripada diregangkan diam-diam dan menghasilkan geometri palsu.
            dm, nama_depth = None, None
        self._frames = [(bgr, dm)]
        self._depth_manual = None
        if dm is None:
            self._siapkan(Path(f).name, "gambar tunggal TANPA kedalaman")
        else:
            self._siapkan(f"{Path(f).name}  +  {nama_depth}",
                          "gambar tunggal DENGAN kedalaman (pasangan ditemukan otomatis)")

    def pilih_depth_manual(self) -> None:
        """Pasangkan berkas kedalaman secara manual untuk gambar yang sedang dipilih."""
        if not self._frames or self._frames[0][0] is None or isinstance(self._frames[0][0], str):
            messagebox.showinfo("Pilih gambar dahulu",
                                "Pemasangan kedalaman manual hanya berlaku untuk gambar tunggal.",
                                parent=self.induk)
            return
        f = filedialog.askopenfilename(title="Pilih berkas kedalaman",
                                       filetypes=[("Kedalaman", "*.npy *.png *.tiff")])
        if not f:
            return
        c = Path(f)
        z = np.load(c) if c.suffix == '.npy' else cv2.imread(str(c), cv2.IMREAD_UNCHANGED)
        if z is None:
            messagebox.showerror("Gagal membaca", f, parent=self.induk); return
        bgr = self._frames[0][0]
        if z.shape[:2] != bgr.shape[:2]:
            messagebox.showerror("Ukuran tidak cocok",
                                 f"Kedalaman {z.shape[:2]} tidak sama dengan gambar {bgr.shape[:2]}.",
                                 parent=self.induk)
            return
        self._frames = [(bgr, z.astype(np.float32) * 0.001)]
        self._siapkan(f"{self.sumber.get().split('  +  ')[0]}  +  {c.name}",
                      "gambar tunggal DENGAN kedalaman (dipasangkan manual)")

    def pilih_video(self) -> None:
        f = filedialog.askopenfilename(title="Pilih video",
                                       filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if not f:
            return
        self._frames = [('video', f)]
        self._siapkan(Path(f).name, "video TANPA kedalaman")

    def pilih_folder(self) -> None:
        d = filedialog.askdirectory(title="Pilih folder exports/frames")
        if not d:
            return
        akar = Path(d)
        frames = sorted(x for x in akar.glob('*')
                        if x.is_dir() and (x / 'color_raw.png').exists()
                        and (x / 'depth_aligned_to_color.npy').exists())
        if not frames:
            messagebox.showinfo("Tidak ada frame",
                                "Folder harus berisi subfolder frame dengan color_raw.png "
                                "dan depth_aligned_to_color.npy.", parent=self.induk)
            return
        self._frames = [('rgbd', f) for f in frames]
        self._siapkan(f"{akar.name}  ({len(frames)} frame)", "frame ekspor DENGAN kedalaman")

    def _siapkan(self, nama: str, jenis: str) -> None:
        self._idx = 0
        self.sumber.set(nama)
        self.jenis.set(jenis)
        if "TANPA" in jenis:
            self.info.set("Berkas ini tidak memuat kedalaman. Model tetap berjalan karena "
                          "dilatih dengan depth dropout, tetapi ketepatannya lebih rendah "
                          "daripada saat kedalaman tersedia.")
        else:
            self.info.set("Kedalaman tersedia; model bekerja pada kondisi penuh.")

    # ------------------------------------------------------------- inferensi
    def _baca_rgbd(self, f: Path):
        import json
        bgr = cv2.imread(str(f / 'color_raw.png'))
        z16 = np.load(f / 'depth_aligned_to_color.npy')
        skala = 0.001
        meta = f / 'frame.json'
        if meta.exists():
            skala = float(json.loads(meta.read_text()).get('depth_scale', 0.001))
        return bgr, z16.astype(np.float32) * skala

    def _prediksi(self, bgr: np.ndarray, dm: np.ndarray | None) -> np.ndarray:
        import torch
        h, w = bgr.shape[:2]
        s = UKURAN / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        dx, dy = (UKURAN - nw) // 2, (UKURAN - nh) // 2

        def lb(a, interp):
            out = np.zeros((UKURAN, UKURAN) + a.shape[2:], a.dtype)
            out[dy:dy + nh, dx:dx + nw] = cv2.resize(a, (nw, nh), interpolation=interp)
            return out

        rgb_l = lb(bgr, cv2.INTER_LINEAR)
        if dm is None:
            # Kedalaman tidak ada: kirim nol dengan peta validitas nol. Ini
            # keadaan yang sama persis dengan depth dropout saat pelatihan,
            # bukan keadaan asing bagi model.
            norm_l = np.zeros((UKURAN, UKURAN), np.float32)
            sah_l = np.zeros((UKURAN, UKURAN), np.float32)
        else:
            norm_l = lb(np.clip((dm - MIN_M) / (MAKS_M - MIN_M), 0, 1).astype(np.float32), cv2.INTER_NEAREST)
            sah_l = lb((dm > 0).astype(np.float32), cv2.INTER_NEAREST)
        x_rgb = cv2.cvtColor(rgb_l, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
        x_dep = np.stack([norm_l * 2 - 1, sah_l]).astype(np.float32)
        with torch.no_grad():
            o = self.model(torch.from_numpy(x_rgb)[None].to(self.dev),
                           torch.from_numpy(x_dep)[None].to(self.dev))
        sem = o['semantic'].argmax(1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(sem[dy:dy + nh, dx:dx + nw], (w, h), interpolation=cv2.INTER_NEAREST)

    def _tempel(self, bgr: np.ndarray, sem: np.ndarray) -> np.ndarray:
        out = bgr.copy(); a = float(self.alpha.get())
        for kelas, warna in ((TREAD, HIJAU), (RISER, MERAH)):
            m = sem == kelas
            if m.any():
                out[m] = (out[m] * (1 - a) + np.array(warna) * a).astype(np.uint8)
        return out

    # --------------------------------------------------------------- kendali
    def toggle(self) -> None:
        self.berhenti() if self.jalan else self.mulai()

    def mulai(self) -> None:
        if self.model is None:
            messagebox.showinfo("Bobot belum dimuat", "Muat bobot lebih dahulu.", parent=self.induk); return
        if not self._frames:
            messagebox.showinfo("Belum ada berkas", "Pilih gambar, video, atau folder frame.", parent=self.induk); return
        self.jalan = True
        self.tombol_main.config(text="■  Berhenti", bg="#8A4A3C")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def berhenti(self) -> None:
        self.jalan = False
        self.tombol_main.config(text="▶  Jalankan", bg=ACCENT)

    def _loop(self) -> None:
        jenis = self._frames[0][0]
        try:
            if jenis == 'video':
                cap = cv2.VideoCapture(self._frames[0][1])
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                i = 0
                while self.jalan:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    self._satu(bgr, None, i, total); i += 1
                cap.release()
            elif jenis == 'rgbd':
                total = len(self._frames)
                for i, (_, f) in enumerate(self._frames):
                    if not self.jalan:
                        break
                    bgr, dm = self._baca_rgbd(f)
                    self._satu(bgr, dm, i, total)
            else:
                bgr, dm = self._frames[0]
                self._satu(bgr, dm, 0, 1)
        except Exception as e:                                    # noqa: BLE001
            self.induk.after(0, lambda: messagebox.showerror("Gagal menjalankan", str(e), parent=self.induk))
        finally:
            self.induk.after(0, self.berhenti)

    def _satu(self, bgr, dm, i, total) -> None:
        t0 = time.perf_counter()
        sem = self._prediksi(bgr, dm)
        ms = (time.perf_counter() - t0) * 1000
        gab = self._tempel(bgr, sem)
        luas = 100.0 * float((sem > 0).mean())
        tanpa = " · TANPA kedalaman" if dm is None else ""
        cv2.rectangle(gab, (0, 0), (gab.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(gab, f"{i+1}/{total}   {ms:.1f} ms   piksel tangga {luas:.1f}%{tanpa}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
        self._terakhir = gab
        self.induk.after(0, lambda g=gab, n=i + 1, t=total: self._tampil(g, n, t))

    def _tampil(self, bgr, n, total) -> None:
        h, w = bgr.shape[:2]
        lw = max(320, self.kanvas.winfo_width()); lh = max(240, self.kanvas.winfo_height())
        s = min(lw / w, lh / h)
        kecil = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        import PIL.Image, PIL.ImageTk
        img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(cv2.cvtColor(kecil, cv2.COLOR_BGR2RGB)))
        self.kanvas.configure(image=img); self.kanvas.image = img
        self.bar['maximum'] = total; self.bar['value'] = n

    def simpan_bingkai(self) -> None:
        if getattr(self, '_terakhir', None) is None:
            messagebox.showinfo("Belum ada bingkai", "Jalankan pengujian lebih dahulu.", parent=self.induk); return
        f = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")], title="Simpan bingkai")
        if f:
            cv2.imwrite(f, self._terakhir)
            self.info.set(f"Bingkai disimpan: {Path(f).name}")
