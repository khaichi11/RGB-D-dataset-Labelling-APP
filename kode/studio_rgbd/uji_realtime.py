"""Tab uji realtime StairFusion — terpisah dari alur pengambilan dataset.

Tab 1-4 di Studio adalah alur PEMBUATAN dataset (rekam, tinjau, ekspor, label).
Tab ini beda tujuan: menjalankan model yang sudah dilatih pada aliran kamera
langsung, untuk melihat mask, garis, dan ukuran H/W secara realtime.

Sengaja dipisah sebagai modul sendiri supaya alur dataset tidak ikut terbebani
impor torch, dan supaya Studio tetap bisa dipakai melabeli di mesin tanpa GPU:
torch baru diimpor saat tab ini benar-benar dipakai.
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np

BG, PANEL, INK, MUTED, ACCENT, LINE = "#F7F3F0", "#FFFFFF", "#2B2622", "#7A6E66", "#C1613C", "#E4DAD3"
GREEN, BLUE = "#3C7A5A", "#3C5F7A"

AKAR = Path(__file__).resolve().parents[2]
BOBOT_BAWAAN = AKAR / "bobot/kandidat/stairfusion_finetune_d435/run2/best.pt"


class UjiRealtime:
    """Panel uji model realtime. Dipasang ke sebuah frame Tk milik Studio."""

    def __init__(self, induk: tk.Frame, studio) -> None:
        self.induk, self.studio = induk, studio
        self.model = None
        self.pipeline = None
        self.jalan = False
        self.thread: threading.Thread | None = None
        self.foto = None

        self.bobot = tk.StringVar(value=str(BOBOT_BAWAAN) if BOBOT_BAWAAN.exists() else "")
        self.ambang = tk.DoubleVar(value=0.70)
        self.tampil_mask = tk.BooleanVar(value=True)
        self.tampil_garis = tk.BooleanVar(value=True)
        self.hitung_geometri = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Model belum dimuat.")
        self.fps = tk.StringVar(value="-")
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

    def _bangun(self) -> None:
        f = tk.Frame(self.induk, bg=BG)
        f.pack(fill="both", expand=True, padx=24, pady=24)
        kiri = tk.Frame(f, bg=BG)
        kiri.pack(side="left", fill="both", expand=True, padx=(0, 12))
        kanan = tk.Frame(f, bg=BG, width=360)
        kanan.pack(side="left", fill="y")
        kanan.pack_propagate(False)

        b, i = self._kartu(kiri, "Tampilan langsung")
        b.pack(fill="both", expand=True)
        self.kanvas = tk.Label(i, bg="#111111")
        self.kanvas.pack(fill="both", expand=True)

        b, i = self._kartu(kanan, "Model")
        b.pack(fill="x")
        tk.Entry(i, textvariable=self.bobot, bg="#FAF7F5", fg=INK,
                 relief="flat", highlightbackground=LINE, highlightthickness=1).pack(fill="x", pady=(0, 6))
        self._tombol(i, "Pilih berkas bobot (.pt)", self.pilih_bobot, "#E8DDD5", INK).pack(fill="x", pady=2)
        self._tombol(i, "Muat model", self.muat_model, BLUE).pack(fill="x", pady=2)

        b, i = self._kartu(kanan, "Kendali")
        b.pack(fill="x", pady=(10, 0))
        self.btn_mulai = self._tombol(i, "▶  Mulai uji realtime", self.toggle, GREEN)
        self.btn_mulai.pack(fill="x", pady=2)
        tk.Label(i, text="Ambang garis", bg=PANEL, fg=MUTED).pack(anchor="w", pady=(8, 0))
        tk.Scale(i, from_=0.30, to=0.95, resolution=0.05, orient="horizontal",
                 variable=self.ambang, bg=PANEL, fg=INK, highlightthickness=0,
                 troughcolor="#EFE7E2").pack(fill="x")
        for teks, var in (("Tampilkan mask", self.tampil_mask),
                          ("Tampilkan garis", self.tampil_garis),
                          ("Hitung ukuran H/W", self.hitung_geometri)):
            tk.Checkbutton(i, text=teks, variable=var, bg=PANEL, fg=INK, selectcolor=PANEL,
                           activebackground=PANEL, anchor="w").pack(fill="x")

        b, i = self._kartu(kanan, "Hasil")
        b.pack(fill="x", pady=(10, 0))
        tk.Label(i, textvariable=self.fps, bg=PANEL, fg=INK,
                 font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x")
        tk.Label(i, textvariable=self.ukuran, bg=PANEL, fg=GREEN,
                 font=("Consolas", 11), justify="left", anchor="w").pack(fill="x", pady=(4, 0))
        tk.Label(i, textvariable=self.status, bg=PANEL, fg=MUTED, wraplength=310,
                 justify="left", anchor="w").pack(fill="x", pady=(8, 0))

        b, i = self._kartu(kanan, "Catatan")
        b.pack(fill="x", pady=(10, 0))
        tk.Label(i, text="Tab ini memakai kamera langsung dan TIDAK merekam apa pun. "
                         "Angka cm hanya sah bila kamera menghadap tangga dan depth "
                         "cukup rapat; bila tidak, kolom ukuran dibiarkan kosong.",
                 bg=PANEL, fg=MUTED, wraplength=310, justify="left").pack(fill="x")

    # -------------------------------------------------------------- aksi

    def pilih_bobot(self) -> None:
        p = filedialog.askopenfilename(title="Pilih bobot model",
                                       filetypes=[("PyTorch", "*.pt"), ("Semua", "*.*")],
                                       initialdir=str(AKAR / "bobot"))
        if p:
            self.bobot.set(p)

    def muat_model(self) -> None:
        try:
            import torch
            from stair_fusion_atto.model_kandidat import StairFusionAttoKandidat
        except Exception as e:                                   # torch/paket belum ada
            self.status.set(f"Gagal impor torch: {e}")
            return
        p = Path(self.bobot.get())
        if not p.exists():
            self.status.set("Berkas bobot tidak ditemukan.")
            return
        try:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ck = torch.load(p, map_location=dev, weights_only=False)
            m = StairFusionAttoKandidat(line_kernel=tuple(ck.get('kernel', (5, 5)))).to(dev).eval()
            m.load_state_dict(ck['model'])
            self.model, self.device = m, dev
            self.rentang = tuple(ck.get('depth_meter_range', (0.2, 4.0)))
            self.status.set(f"Model dimuat pada {dev}. Rentang depth {self.rentang[0]}-{self.rentang[1]} m.")
        except Exception as e:
            self.status.set(f"Gagal memuat model: {e}")

    def toggle(self) -> None:
        if self.jalan:
            self.berhenti()
        else:
            self.mulai()

    def mulai(self) -> None:
        if self.model is None:
            self.status.set("Muat model dulu.")
            return
        if getattr(self.studio, 'sedang_rekam', False):
            self.status.set("Sedang merekam dataset. Hentikan dulu agar kamera tidak berebut.")
            return
        # Studio memakai kamera untuk preview/rekam; keduanya tidak boleh
        # membuka device yang sama bersamaan.
        try:
            if getattr(self.studio, 'cam', None) is not None and self.studio.cam.hidup:
                self.studio.preview_diminta = False
                self.studio._hentikan_render()
                self.studio.cam.hentikan()
        except Exception:
            pass
        self.jalan = True
        self.btn_mulai.configure(text="■  Hentikan", bg="#A6452F")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def berhenti(self) -> None:
        self.jalan = False
        self.btn_mulai.configure(text="▶  Mulai uji realtime", bg=GREEN)

    # -------------------------------------------------------------- loop

    def _loop(self) -> None:
        import torch
        import pyrealsense2 as rs
        from stair_fusion_atto.stabil.garis import ekstrak_garis
        from stair_fusion_atto.stabil.geometri import Intrinsics
        from stair_fusion_atto.stabil.geometri_garis import SumbuAtas, garis_ke_3d, ukur_dari_garis
        from stair_fusion_atto.stabil.infer import (PALETTE, letterbox, saring_komponen,
                                                    unletterbox_map)
        from stair_fusion_atto.stabil.pelacak import PelacakGaris

        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        try:
            prof = pipe.start(cfg)
        except Exception as e:
            self.status.set(f"Kamera gagal dibuka: {e}")
            self.jalan = False
            return
        skala = float(prof.get_device().first_depth_sensor().get_depth_scale())
        align = rs.align(rs.stream.color)
        ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intr = Intrinsics(ci.fx, ci.fy, ci.ppx, ci.ppy, ci.width, ci.height)
        tracker, sumbu = PelacakGaris(), SumbuAtas()
        lo, hi = self.rentang
        t0, n = time.time(), 0
        riser, tread = [], []
        try:
            while self.jalan:
                try:
                    frames = align.process(pipe.wait_for_frames(2000))
                except Exception:
                    continue
                c, d = frames.get_color_frame(), frames.get_depth_frame()
                if not c or not d:
                    continue
                bgr = np.asanyarray(c.get_data())
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                z16 = np.asanyarray(d.get_data())
                depth_m = z16.astype(np.float32) * skala
                h, w = rgb.shape[:2]
                x_scan = w // 2

                rl, s, dx, dy = letterbox(rgb, 512, cv2.INTER_LINEAR)
                dn = letterbox(np.clip((depth_m - lo) / (hi - lo), 0, 1), 512, cv2.INTER_NEAREST)[0]
                vl = letterbox((z16 > 0).astype(np.float32), 512, cv2.INTER_NEAREST)[0]
                rt = torch.from_numpy(rl.transpose(2, 0, 1)).float()[None].to(self.device) / 127.5 - 1
                dt = torch.from_numpy(np.stack((dn * 2 - 1, vl))[None]).float().to(self.device)
                with torch.inference_mode():
                    out = self.model(rt, dt)
                    sem = out['semantic'].softmax(1)[0].cpu().numpy()
                    line = torch.sigmoid(out['line'])[0].cpu().numpy()
                kelas = saring_komponen(unletterbox_map(sem, s, dx, dy, w, h).argmax(0).astype(np.uint8))

                view = bgr.copy()
                if self.tampil_mask.get():
                    view = cv2.addWeighted(view, 0.68, PALETTE[kelas][:, :, ::-1], 0.32, 0)
                tracks = []
                if self.tampil_garis.get():
                    lu = np.stack([cv2.resize(line[k], (512, 512), interpolation=cv2.INTER_LINEAR)
                                   for k in range(2)])
                    pv = unletterbox_map(lu[0], s, dx, dy, w, h)
                    cv_ = unletterbox_map(lu[1], s, dx, dy, w, h)
                    tracks = tracker.update(ekstrak_garis(pv, cv_, float(self.ambang.get()),
                                                          min_length=max(40, w // 8)))
                    for t in tracks:
                        x1, y1, x2, y2 = np.asarray(t['points']).round().astype(int)
                        col = (92, 92, 255) if t['kelas'] == 1 else (255, 210, 0)
                        cv2.line(view, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

                if self.hitung_geometri.get() and tracks and n % 3 == 0:
                    g3 = garis_ke_3d(tracks, depth_m, intr, x_scan)
                    up = sumbu.perbarui(g3, depth_m, kelas, intr, lo, hi)
                    m = ukur_dari_garis(tracks, depth_m, intr, x_scan, sumbu_atas=up)
                    riser += [u['nilai_cm'] for u in m['tinggi_riser'] if 5 <= u['nilai_cm'] <= 30]
                    tread += [u['nilai_cm'] for u in m['panjang_tread'] if 15 <= u['nilai_cm'] <= 50]
                    riser, tread = riser[-60:], tread[-60:]
                    baris = []
                    if riser:
                        baris.append(f"tinggi riser  : {np.median(riser):5.1f} cm")
                    if tread:
                        baris.append(f"panjang tread : {np.median(tread):5.1f} cm")
                    self.ukuran.set("\n".join(baris) if baris else "ukuran: depth belum cukup")

                n += 1
                if n % 10 == 0:
                    self.fps.set(f"{n / max(1e-6, time.time() - t0):.1f} FPS   |   {len(tracks)} garis")
                self._tampilkan(view)
        finally:
            try:
                pipe.stop()
            except Exception:
                pass
            self.jalan = False
            try:
                self.btn_mulai.configure(text="▶  Mulai uji realtime", bg=GREEN)
            except Exception:
                pass

    def _tampilkan(self, bgr: np.ndarray) -> None:
        try:
            from PIL import Image, ImageTk
        except Exception:
            return
        lebar = max(320, self.kanvas.winfo_width())
        skala = min(1.0, lebar / bgr.shape[1])
        kecil = cv2.resize(bgr, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(kecil, cv2.COLOR_BGR2RGB)))
        self.foto = img                       # tahan referensi; tanpa ini gambar dikumpulkan GC
        self.kanvas.configure(image=img)
