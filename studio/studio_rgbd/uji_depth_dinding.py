"""Uji depth dinding: jarak depth D435 ke dinding datar dibandingkan dengan meteran.

Pada 4 tangga bermeteran (model/bukti/final_d435/galat_ukur_20261017/) anak tangga yang
sama terukur makin panjang saat kamera makin jauh: depth tampaknya makin terlalu jauh
dengan jarak, kira-kira z_terbaca = z (1 + k z) dengan k ~0,05 per m. Jendela ini
memastikannya langsung: kamera dihadapkan ke dinding datar, jarak kaca depan kamera ke
dinding diukur meteran, lalu Studio mengambil 30 frame depth, memasang bidang pada bagian
tengah citra, dan mencatat jarak tegak lurus kamera ke bidang itu.

Depth dibaca dari stream depth asli (bukan yang diselaraskan ke warna), dengan
intrinsik depth, jadi titik asalnya imager kiri D435. Titik nol depth D435 berada
4,2 mm di belakang kaca depan (lembar data Intel D400, "Depth Start Point"), jadi
jarak meteran dari kaca ditambah 0,42 cm.

Hasil dapat disalin sebagai teks dan tersimpan di
``<akar data>/uji_depth_dinding/<waktu>.json``.
"""
from __future__ import annotations

import json
import math
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import numpy as np

BG, PANEL, INK, MUTED, ACCENT, LINE = "#F7F3F0", "#FFFFFF", "#2B2622", "#7A6E66", "#C1613C", "#E4DAD3"
GREEN, RED = "#3C7A5A", "#A6452F"

ASAL_KACA_M = 0.0042        # titik nol depth D435 di belakang kaca depan
N_FRAME = 30
PORSI_TENGAH = 0.4          # lebar/tinggi bagian tengah citra yang dipakai untuk bidang
K_TANGGA = 0.05             # perkiraan dari tangga bermeteran, 17 Okt 2026


def ukur_bidang_dinding(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                        porsi: float = PORSI_TENGAH) -> dict | None:
    """Bidang pada bagian tengah depth: jarak tegak lurus dari kamera, kemiringan, depth titik tengah.

    Bidang dipasang dengan PCA lalu titik yang menyimpang (> 3 MAD, minimal 5 mm) dibuang tiga
    kali, agar benda kecil di dinding atau tepi yang bocor tidak menarik bidang.
    """
    h, w = depth_m.shape
    y0, y1 = int(h * (1 - porsi) / 2), int(h * (1 + porsi) / 2)
    x0, x1 = int(w * (1 - porsi) / 2), int(w * (1 + porsi) / 2)
    z = depth_m[y0:y1:2, x0:x1:2].astype(np.float64)
    v, u = np.mgrid[y0:y1:2, x0:x1:2]
    sah = (z > 0.1) & (z < 8.0)
    rasio = float(sah.mean()) if sah.size else 0.0
    if sah.sum() < 200:
        return None
    zs = z[sah]
    pts = np.stack([(u[sah] - cx) * zs / fx, (v[sah] - cy) * zs / fy, zs], axis=1)
    pakai = np.ones(len(pts), bool)
    for _ in range(4):
        p = pts[pakai]
        c = p.mean(0)
        _, _, vt = np.linalg.svd(p - c, full_matrices=False)
        n = vt[2] if vt[2][2] < 0 else -vt[2]           # normal mengarah ke kamera
        r = (pts - c) @ n
        med = np.median(r[pakai])
        mad = 1.4826 * np.median(np.abs(r[pakai] - med))
        pakai = np.abs(r - med) < max(3 * mad, 0.005)
        if pakai.sum() < 200:
            return None
    jarak = abs(float(n @ c))
    rms = float(np.sqrt(np.mean(r[pakai] ** 2)))
    tengah = depth_m[h // 2 - 4:h // 2 + 5, w // 2 - 4:w // 2 + 5]
    tengah = tengah[(tengah > 0.1) & (tengah < 8.0)]
    return {"jarak_m": jarak, "miring_deg": math.degrees(math.acos(min(1.0, abs(float(n[2]))))),
            "rms_mm": rms * 1000, "sah": rasio, "inlier": float(pakai.mean()),
            "tengah_m": float(np.median(tengah)) if tengah.size else float("nan")}


def perkiraan_k(baris: list[dict]) -> float | None:
    """k pada z_benar = z / (1 + k z) (sama dengan TANGGA_KOREKSI_DEPTH pelacak), kuadrat terkecil."""
    if not baris:
        return None
    zt = np.array([b["jarak_m"] for b in baris])
    rel = zt / np.array([b["benar_m"] for b in baris]) - 1
    return float(rel @ zt / (zt @ zt))


def teks_hasil(info: dict, baris: list[dict]) -> str:
    """Ringkasan yang bisa disalin apa adanya."""
    t = [f"UJI DEPTH DINDING D435 - {info.get('waktu', '')}",
         f"kamera: {info.get('kamera', '?')}, seri {info.get('seri', '?')}, firmware {info.get('firmware', '?')}",
         f"preset: {info.get('preset', '?')}, depth {info.get('resolusi', '?')}, suhu ASIC {info.get('suhu_asic', '?')} C, "
         f"proyektor {info.get('suhu_proyektor', '?')} C, laser {info.get('laser', '?')}",
         f"meteran dari kaca depan kamera; titik nol depth = kaca + {ASAL_KACA_M * 100:.2f} cm; "
         f"bidang = bagian tengah {int(PORSI_TENGAH * 100)} % citra, median {N_FRAME} frame",
         "no | meteran cm | benar cm | depth bidang cm | selisih cm | galat % | depth titik tengah cm | miring deg | rms mm | sah % | sebaran antar-frame mm"]
    for i, b in enumerate(baris, 1):
        sel = (b["jarak_m"] - b["benar_m"]) * 100
        t.append(f"{i} | {b['meteran_cm']:.1f} | {b['benar_m'] * 100:.2f} | {b['jarak_m'] * 100:.2f} | {sel:+.2f} | "
                 f"{sel / b['benar_m']:+.2f} | {b['tengah_m'] * 100:.2f} | {b['miring_deg']:.1f} | {b['rms_mm']:.1f} | "
                 f"{b['sah'] * 100:.0f} | {b['sebaran_mm']:.1f}")
    k = perkiraan_k(baris)
    if k is not None:
        t.append(f"k (z_benar = z / (1 + k z)) = {k:+.4f} per m dari {len(baris)} jarak "
                 f"[dari tangga bermeteran 17 Okt: ~{K_TANGGA:.2f}]")
    return "\n".join(t)


class JendelaUjiDepth(tk.Toplevel):
    """Jendela uji; memakai kamera Studio yang sudah hidup (preview tab Rekam), tidak membuka kamera sendiri."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.app = app
        self.title("Uji depth dinding (meteran)")
        self.configure(bg=BG)
        self.geometry("980x640")
        self.baris: list[dict] = []
        self.info: dict = {"waktu": datetime.now().strftime("%Y-%m-%d %H:%M")}
        self.berkas = Path(app.root_data) / "uji_depth_dinding" / f"{datetime.now():%Y%m%d_%H%M%S}.json"
        self._kumpul: list[dict] | None = None
        self._nomor_akhir = -1
        self._mulai_kumpul = 0.0
        self._buat_ui()
        self.protocol("WM_DELETE_WINDOW", self.tutup)
        self.after(200, self._detak)

    # ---------------- tata letak ----------------
    def _buat_ui(self) -> None:
        kiri = tk.Frame(self, bg=BG); kiri.pack(side="left", fill="y", padx=(16, 8), pady=16)
        kanan = tk.Frame(self, bg=BG); kanan.pack(side="left", fill="both", expand=True, padx=(8, 16), pady=16)
        petunjuk = ("1. Nyalakan preview kamera (tab Rekam).\n"
                    "2. Hadapkan kamera tegak lurus ke dinding datar; bagian tengah citra harus dinding saja.\n"
                    "3. Ukur jarak dari KACA DEPAN kamera ke dinding dengan meteran, tegak lurus dinding.\n"
                    "4. Isi jarak (cm), tekan Ukur. Ulangi pada ~0,5 / 1 / 1,5 / 2 / 2,5 / 3 m.\n"
                    "Usahakan miring < 5 derajat. Setelah selesai tekan Salin hasil.")
        tk.Label(kiri, text=petunjuk, bg=BG, fg=INK, justify="left", wraplength=330).pack(anchor="w")
        self.var_langsung = tk.StringVar(value="Kamera belum hidup.")
        tk.Label(kiri, textvariable=self.var_langsung, bg=PANEL, fg=INK, justify="left", font=("Consolas", 11),
                 relief="flat", padx=10, pady=8, width=36, anchor="w").pack(anchor="w", pady=(12, 8), fill="x")
        self.btn_kamera = tk.Button(kiri, text="Nyalakan kamera", command=self._nyalakan, bg="#E8DDD5", fg=INK,
                                    relief="flat")
        self.btn_kamera.pack(anchor="w", fill="x")
        baris = tk.Frame(kiri, bg=BG); baris.pack(anchor="w", pady=(14, 4), fill="x")
        tk.Label(baris, text="Jarak meteran (cm)", bg=BG, fg=MUTED).pack(side="left")
        self.var_meteran = tk.StringVar()
        isian = tk.Entry(baris, textvariable=self.var_meteran, width=10, bg="#FFF9F4", fg=INK, relief="flat")
        isian.pack(side="left", padx=8)
        isian.bind("<Return>", lambda _e: self.ukur())
        self.btn_ukur = tk.Button(kiri, text=f"Ukur ({N_FRAME} frame)", command=self.ukur, bg=ACCENT, fg="white",
                                  relief="flat")
        self.btn_ukur.pack(anchor="w", fill="x", pady=(4, 4))
        tk.Button(kiri, text="Hapus baris terakhir", command=self.hapus_terakhir, bg="#E8DDD5", fg=INK,
                  relief="flat").pack(anchor="w", fill="x", pady=(0, 4))
        tk.Button(kiri, text="Salin hasil", command=self.salin, bg=GREEN, fg="white",
                  relief="flat").pack(anchor="w", fill="x", pady=(10, 4))
        self.var_status = tk.StringVar(value=f"Tersimpan otomatis ke {self.berkas}")
        tk.Label(kiri, textvariable=self.var_status, bg=BG, fg=MUTED, justify="left",
                 wraplength=330).pack(anchor="w", pady=(6, 0))

        kolom = ("meteran", "depth", "selisih", "galat", "tengah", "miring", "rms", "sebaran")
        judul = ("meteran cm", "depth bidang cm", "selisih cm", "galat %", "titik tengah cm", "miring °", "rms mm",
                 "sebaran mm")
        self.tabel = ttk.Treeview(kanan, columns=kolom, show="headings", height=8)
        for k, j in zip(kolom, judul):
            self.tabel.heading(k, text=j)
            self.tabel.column(k, width=82, anchor="e")
        self.tabel.pack(fill="x")
        self.var_k = tk.StringVar(value="")
        tk.Label(kanan, textvariable=self.var_k, bg=BG, fg=INK, font=("Segoe UI", 11, "bold"),
                 justify="left").pack(anchor="w", pady=(8, 4))
        tk.Label(kanan, text="Teks untuk disalin (bisa juga dipilih manual):", bg=BG, fg=MUTED).pack(anchor="w")
        self.teks = tk.Text(kanan, height=14, bg=PANEL, fg=INK, relief="flat", font=("Consolas", 9), wrap="none")
        self.teks.pack(fill="both", expand=True)

    # ---------------- kamera ----------------
    def _kamera(self):
        cam = getattr(self.app, "cam", None)
        if cam is None or not cam.hidup or getattr(self.app, "sedang_rekam", False):
            return None
        return cam

    def _nyalakan(self) -> None:
        if self._kamera() is None and not getattr(self.app, "sedang_rekam", False):
            self.app.toggle_preview_kamera()

    def _frame_baru(self, cam):
        """(depth meter, intrinsik depth, nomor frame) dari frameset terakhir pompa, atau None bila belum baru."""
        fs = cam.frameset_terbaru() if cam.pompa_hidup else None
        if fs is None:
            return None
        depth = fs.get_depth_frame()
        if not depth:
            return None
        nomor = depth.get_frame_number()
        if nomor == self._nomor_akhir:
            return None
        self._nomor_akhir = nomor
        skala = float(cam.profile.get_device().first_depth_sensor().get_depth_scale())
        intr = depth.profile.as_video_stream_profile().intrinsics
        m = np.asanyarray(depth.get_data()).astype(np.float32) * skala
        return m, intr, nomor

    def _catat_info(self, cam, intr) -> None:
        try:
            import pyrealsense2 as rs
            dev = cam.profile.get_device()
            ds = dev.first_depth_sensor()
            self.info.update(kamera=dev.get_info(rs.camera_info.name), seri=dev.get_info(rs.camera_info.serial_number),
                             firmware=dev.get_info(rs.camera_info.firmware_version), preset=cam.preset_sekarang(),
                             resolusi=f"{intr.width}x{intr.height}", fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy)
            for kunci, opsi in (("suhu_asic", rs.option.asic_temperature),
                                ("suhu_proyektor", rs.option.projector_temperature),
                                ("laser", rs.option.laser_power)):
                if ds.supports(opsi):
                    self.info[kunci] = round(float(ds.get_option(opsi)), 1)
        except Exception:                                       # noqa: BLE001
            pass

    def _detak(self) -> None:
        if not self.winfo_exists():
            return
        cam = self._kamera()
        if cam is None:
            self.var_langsung.set("Kamera mati. Tekan Nyalakan kamera\n(tab Rekam harus terbuka, tidak sedang rekam).")
            self.btn_kamera.configure(state="normal")
            if self._kumpul is not None:
                self._kumpul = None
                self.btn_ukur.configure(state="normal", text=f"Ukur ({N_FRAME} frame)")
            self.after(400, self._detak)
            return
        self.btn_kamera.configure(state="disabled")
        f = self._frame_baru(cam)
        if f is not None:
            m, intr, _ = f
            hasil = ukur_bidang_dinding(m, intr.fx, intr.fy, intr.ppx, intr.ppy)
            if self._kumpul is not None:
                if hasil is not None:
                    self._kumpul.append(hasil)
                if len(self._kumpul) >= N_FRAME:
                    self._selesai_kumpul(cam, intr)
                elif time.monotonic() - self._mulai_kumpul > 8:
                    self.var_status.set("Gagal: depth dinding tidak cukup (terlalu dekat/jauh atau bukan dinding?).")
                    self._kumpul = None
                    self.btn_ukur.configure(state="normal", text=f"Ukur ({N_FRAME} frame)")
            if hasil is None:
                self.var_langsung.set("Bidang tidak terpasang:\nbagian tengah citra bukan dinding datar?")
            else:
                self.var_langsung.set(f"depth bidang  {hasil['jarak_m'] * 100:7.1f} cm\n"
                                      f"titik tengah  {hasil['tengah_m'] * 100:7.1f} cm\n"
                                      f"miring        {hasil['miring_deg']:7.1f} deg\n"
                                      f"rms bidang    {hasil['rms_mm']:7.1f} mm   sah {hasil['sah'] * 100:3.0f} %")
        self.after(40 if self._kumpul is not None else 150, self._detak)

    # ---------------- ukur ----------------
    def ukur(self) -> None:
        if self._kumpul is not None:
            return
        try:
            meteran = float(self.var_meteran.get().replace(",", "."))
        except ValueError:
            self.var_status.set("Isi jarak meteran dalam cm, mis. 150 atau 150,5.")
            return
        if not 20 <= meteran <= 600:
            self.var_status.set("Jarak meteran di luar 20-600 cm.")
            return
        if self._kamera() is None:
            self.var_status.set("Kamera belum hidup.")
            return
        self._meteran = meteran
        self._kumpul = []
        self._mulai_kumpul = time.monotonic()
        self.btn_ukur.configure(state="disabled", text="Mengukur… tahan kamera")

    def _selesai_kumpul(self, cam, intr) -> None:
        h = self._kumpul or []
        self._kumpul = None
        self.btn_ukur.configure(state="normal", text=f"Ukur ({N_FRAME} frame)")
        if "kamera" not in self.info:
            self._catat_info(cam, intr)
        jarak = np.array([x["jarak_m"] for x in h])
        b = {"meteran_cm": self._meteran, "benar_m": self._meteran / 100 + ASAL_KACA_M,
             "jarak_m": float(np.median(jarak)), "sebaran_mm": float(np.std(jarak) * 1000),
             "tengah_m": float(np.nanmedian([x["tengah_m"] for x in h])),
             "miring_deg": float(np.median([x["miring_deg"] for x in h])),
             "rms_mm": float(np.median([x["rms_mm"] for x in h])),
             "sah": float(np.median([x["sah"] for x in h])), "n_frame": len(h)}
        self.baris.append(b)
        self.var_meteran.set("")
        self._segarkan()
        if b["miring_deg"] > 5:
            self.var_status.set(f"Tercatat, tetapi miring {b['miring_deg']:.1f} derajat: pastikan meteran juga tegak lurus dinding.")

    def hapus_terakhir(self) -> None:
        if self.baris:
            self.baris.pop()
            self._segarkan()

    def _segarkan(self) -> None:
        self.tabel.delete(*self.tabel.get_children())
        for b in self.baris:
            sel = (b["jarak_m"] - b["benar_m"]) * 100
            self.tabel.insert("", "end", values=(f"{b['meteran_cm']:.1f}", f"{b['jarak_m'] * 100:.2f}", f"{sel:+.2f}",
                                                 f"{sel / b['benar_m']:+.2f}", f"{b['tengah_m'] * 100:.2f}",
                                                 f"{b['miring_deg']:.1f}", f"{b['rms_mm']:.1f}", f"{b['sebaran_mm']:.1f}"))
        k = perkiraan_k(self.baris)
        self.var_k.set("" if k is None else
                       f"k = {k:+.4f} per m  (dari tangga: ~{K_TANGGA:.2f}; 0 = depth tepat)")
        teks = teks_hasil(self.info, self.baris)
        self.teks.delete("1.0", "end")
        self.teks.insert("1.0", teks)
        try:
            self.berkas.parent.mkdir(parents=True, exist_ok=True)
            self.berkas.write_text(json.dumps({"info": self.info, "baris": self.baris, "k": k, "teks": teks},
                                              ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            self.var_status.set(f"Tersimpan: {self.berkas}")
        except OSError as e:
            self.var_status.set(f"Gagal menyimpan: {e}")

    def salin(self) -> None:
        teks = teks_hasil(self.info, self.baris)
        self.clipboard_clear()
        self.clipboard_append(teks)
        self.var_status.set("Hasil disalin. Tempel di chat.")

    def tutup(self) -> None:
        self._kumpul = None
        self.destroy()
