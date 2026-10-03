"""Kartu "Ukuran tangga (meteran)" di tab Label: nomor anak tangga dari video, ukuran sistem, isian meteran.

Pada frame ekspor (mis. tiap 10 frame) anak tangga ke-1, 2, 3 sulit dibedakan,
sedangkan pada video perpindahannya jelas. Tombol "Hitung nomor dari video"
menjalankan pelacak sistem (``rgbd_convnext.ukuran_sesi``, mekanisme yang sama
dengan perintah ``tangga``) pada SETIAP frame rekaman sesi; hasilnya:

- nomor ``#n R`` / ``#n T`` ditandai di atas gambar frame yang sedang dilabel
  (dicocokkan lewat cap waktu kamera warna);
- video bernomor ``derived/ukuran_sistem.mp4`` untuk memastikan urutannya;
- tinggi riser dan panjang tread sistem per anak tangga, di samping kolom
  isian ukuran meteran yang disimpan ``ukuran_meteran.py`` (per tangga fisik
  bila sesi/scene sudah dikaitkan ke tangga, selain itu per sesi).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path

try:
    from . import catatan_rekaman as CR
    from . import ukuran_meteran as UM
    from .uji_realtime import AKAR_APLIKASI
except ImportError:
    import catatan_rekaman as CR
    import ukuran_meteran as UM
    from uji_realtime import AKAR_APLIKASI

PANEL, INK, MUTED, ACCENT = "#FFFFFF", "#2B2622", "#7A6E66", "#C1613C"
WARNA_R, WARNA_T = "#FFB347", "#7CE0FF"        # nomor riser / tread di atas gambar


def _modul_ukuran():
    if str(AKAR_APLIKASI) not in sys.path:
        sys.path.insert(0, str(AKAR_APLIKASI))
    from rgbd_convnext import ukuran_sesi
    return ukuran_sesi


def _cm(blok) -> str:
    return f"{blok['cm']:.1f}" if blok and blok.get("cm") is not None else "-"


class PanelUkuran:
    def __init__(self, induk: tk.Frame, studio) -> None:
        self.induk, self.studio = induk, studio
        self.sesi: Path | None = None
        self.tid: str | None = None
        self.sistem: dict | None = None
        self._ke = 1                                # penomoran yang tampak di frame ini
        self._thread: threading.Thread | None = None
        self._progres = None
        self.tandai = tk.BooleanVar(value=True)
        self.geser = tk.IntVar(value=1)             # sistem #1 = anak tangga fisik ke-?
        self.lebar = tk.StringVar(value="")
        self.info = tk.StringVar(value="Buka sebuah frame ekspor.")
        self.status = tk.StringVar(value="")
        self._isian: dict[int, tuple[tk.StringVar, tk.StringVar]] = {}

        tk.Label(induk, textvariable=self.info, bg=PANEL, fg=INK, wraplength=300, justify="left",
                 font=("Segoe UI", 8)).pack(anchor="w")
        baris = tk.Frame(induk, bg=PANEL); baris.pack(fill="x", pady=(4, 0))
        self.btn_hitung = tk.Button(baris, text="Hitung nomor dari video", command=self.hitung, bg="#3C5F7A",
                                    fg="white", relief="flat", font=("Segoe UI", 8, "bold"), pady=4)
        self.btn_hitung.pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(baris, text="▶ Video bernomor", command=self.putar_video, bg="#E8DDD5", fg=INK, relief="flat",
                  font=("Segoe UI", 8, "bold"), pady=4).pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Checkbutton(induk, text="Tandai nomor anak tangga di gambar (dari video)", variable=self.tandai,
                       command=self._segarkan_tanda, bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))
        g = tk.Frame(induk, bg=PANEL); g.pack(fill="x", pady=(2, 0))
        tk.Label(g, text="Nomor sistem #1 = anak tangga fisik ke-", bg=PANEL, fg=INK,
                 font=("Segoe UI", 8)).pack(side="left")
        tk.Spinbox(g, from_=1, to=30, width=3, textvariable=self.geser, command=self._bangun_tabel,
                   font=("Segoe UI", 8)).pack(side="left")
        self.tabel = tk.Frame(induk, bg=PANEL); self.tabel.pack(fill="x", pady=(4, 0))
        lb = tk.Frame(induk, bg=PANEL); lb.pack(fill="x", pady=(2, 0))
        tk.Label(lb, text="Lebar tangga (cm, opsional)", bg=PANEL, fg=INK, font=("Segoe UI", 8)).pack(side="left")
        tk.Entry(lb, textvariable=self.lebar, width=7, font=("Segoe UI", 8)).pack(side="left", padx=(4, 0))
        tk.Button(induk, text="Simpan ukuran meteran", command=self.simpan, bg=ACCENT, fg="white", relief="flat",
                  font=("Segoe UI", 8, "bold"), pady=4).pack(fill="x", pady=(4, 0))
        tk.Label(induk, textvariable=self.status, bg=PANEL, fg=MUTED, wraplength=300, justify="left",
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

    # ------------------------------------------------------------ frame dibuka
    def frame_dibuka(self, p: Path) -> None:
        sesi = Path(p).parent.parent.parent
        catatan = CR.baca(sesi)
        tid = CR.tangga_untuk(catatan, CR.indeks_frame(Path(p).name))
        if sesi != self.sesi or tid != self.tid:
            self.sesi, self.tid = sesi, tid
            self.sistem = _modul_ukuran().baca(sesi) if self._ada_sistem(sesi) else None
            self._muat_meteran()
        self._segarkan_tanda()

    @staticmethod
    def _ada_sistem(sesi: Path) -> bool:
        return (sesi / "derived" / "ukuran_sistem.json").exists()

    def _waktu_frame(self) -> float | None:
        info = getattr(self.studio, "label_info", None) or {}
        for kunci in ("timestamp_rgb_ms", "timestamp_kamera_ms"):
            if info.get(kunci) is not None:
                return float(info[kunci])
        return None

    def _segarkan_tanda(self) -> None:
        kanvas = getattr(self.studio, "kanvas", None)
        tanda = []
        if self.sistem is not None:
            tanda = _modul_ukuran().tanda_frame(self.sistem, self._waktu_frame())
        ke_tampak = sorted({t[0] for t in tanda})
        ke = ke_tampak[0] if ke_tampak else 1
        if ke != self._ke:
            self._ke = ke
            self._muat_meteran()
        if kanvas is not None:
            geser = self.geser.get() - 1
            kanvas.nomor_anak_tangga = [
                (x, y, f"#{n + geser} {k}" + (f" (T{t})" if len(ke_tampak) > 1 else ""),
                 WARNA_R if k == "R" else WARNA_T)
                for t, n, k, x, y in tanda] if self.tandai.get() else []
            kanvas.render()
        self._perbarui_info(len(tanda))

    def _perbarui_info(self, n_tanda: int) -> None:
        if self.sesi is None:
            return
        daftar = CR.baca_tangga(self.studio.root_data)
        tempat = (f"tangga fisik {CR.label_tangga(self.tid, daftar)} (dipakai semua rekaman tangga ini)"
                  if self.tid else "sesi ini saja (kaitkan sesi ke tangga fisik di tab Tinjau agar berlaku "
                                   "untuk semua rekamannya)")
        if self.sistem is None:
            nomor = "Nomor dari video belum dihitung."
        else:
            n = sum(len(t["anak_tangga"]) for t in self.sistem["tangga"])
            nomor = (f"Nomor dari video: {n} anak tangga, {len(self.sistem['tangga'])} penomoran; "
                     f"frame ini {n_tanda} bidang bernomor.")
        self.info.set(f"{self.sesi.name}\nUkuran meteran disimpan untuk {tempat}.\n{nomor}")

    # ------------------------------------------------------------ tabel
    def _muat_meteran(self) -> None:
        if self.sesi is None:
            return
        met = UM.baca(self.sesi, self.studio.root_data, self.tid, self._ke)
        self._isian = {}                            # isian sesi/tangga lain tidak boleh terbawa
        self.geser.set(met["geser"].get(self._ke, 0) + 1)
        self.lebar.set("" if met["lebar_cm"] in (None, "") else str(met["lebar_cm"]))
        self._meteran = met["anak_tangga"]
        self._bangun_tabel()

    def _bangun_tabel(self) -> None:
        for w in self.tabel.winfo_children():
            w.destroy()
        lama = {k: (r.get(), t.get()) for k, (r, t) in self._isian.items()}
        self._isian = {}
        try:
            geser = int(self.geser.get()) - 1
        except (tk.TclError, ValueError):
            geser = 0
        sistem = {}
        if self.sistem is not None:
            for t in self.sistem["tangga"]:
                if t["ke"] == self._ke:
                    sistem = {a["nomor"] + geser: a for a in t["anak_tangga"]}
        fisik = sorted(set(sistem) | set(getattr(self, "_meteran", {})))
        for kol, teks in enumerate(("fisik", "sistem R/T cm", "meteran R", "meteran T")):
            tk.Label(self.tabel, text=teks, bg=PANEL, fg=MUTED, font=("Segoe UI", 7, "bold")).grid(
                row=0, column=kol, sticky="w", padx=1)
        if not fisik:
            tk.Label(self.tabel, text="(hitung nomor dari video dulu)", bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8)).grid(row=1, column=0, columnspan=4, sticky="w")
        for baris, nomor in enumerate(fisik, start=1):
            a = sistem.get(nomor)
            m = getattr(self, "_meteran", {}).get(nomor, {})
            r = tk.StringVar(value=lama.get(nomor, (None, None))[0] or ("" if m.get("tinggi_riser_cm") is None
                                                                       else str(m["tinggi_riser_cm"])))
            t = tk.StringVar(value=lama.get(nomor, (None, None))[1] or ("" if m.get("panjang_tread_cm") is None
                                                                       else str(m["panjang_tread_cm"])))
            self._isian[nomor] = (r, t)
            tk.Label(self.tabel, text=f"#{nomor}", bg=PANEL, fg=INK, font=("Segoe UI", 8, "bold")).grid(
                row=baris, column=0, sticky="w", padx=1)
            tk.Label(self.tabel, text=(f"{_cm(a['tinggi_riser'])} / {_cm(a['panjang_tread'])}" if a else "-"),
                     bg=PANEL, fg=INK, font=("Consolas", 8)).grid(row=baris, column=1, sticky="w", padx=1)
            tk.Entry(self.tabel, textvariable=r, width=6, font=("Segoe UI", 8)).grid(row=baris, column=2, padx=1)
            tk.Entry(self.tabel, textvariable=t, width=6, font=("Segoe UI", 8)).grid(row=baris, column=3, padx=1)

    def simpan(self) -> None:
        if self.sesi is None:
            self.status.set("Buka sebuah frame ekspor dulu.")
            return
        anak, salah = {}, []
        for nomor, (r, t) in self._isian.items():
            isi = {}
            for kunci, var in (("tinggi_riser_cm", r), ("panjang_tread_cm", t)):
                teks = var.get().strip().replace(",", ".")
                if teks:
                    try:
                        isi[kunci] = float(teks)
                    except ValueError:
                        salah.append(f"#{nomor}")
            if isi:
                anak[nomor] = isi
        if salah:
            self.status.set(f"Angka tidak terbaca pada {', '.join(sorted(set(salah)))}; pakai titik/koma desimal.")
            return
        lebar = self.lebar.get().strip().replace(",", ".")
        try:
            lebar_cm = float(lebar) if lebar else None
        except ValueError:
            self.status.set("Lebar tangga tidak terbaca.")
            return
        met = UM.baca(self.sesi, self.studio.root_data, self.tid, self._ke)
        geser = dict(met["geser"])
        geser[self._ke] = int(self.geser.get()) - 1
        tempat = UM.tulis(self.sesi, self.studio.root_data, self.tid, anak, lebar_cm, geser, self._ke)
        self._meteran = UM._bersihkan({str(k): v for k, v in anak.items()})
        beda = []
        if self.sistem is not None:
            for t in self.sistem["tangga"]:
                if t["ke"] != self._ke:
                    continue
                for a in t["anak_tangga"]:
                    m = anak.get(a["nomor"] + geser[self._ke], {})
                    for kunci, besaran, huruf in (("tinggi_riser_cm", "tinggi_riser", "R"),
                                                  ("panjang_tread_cm", "panjang_tread", "T")):
                        if m.get(kunci) is not None and (a.get(besaran) or {}).get("cm") is not None:
                            beda.append(f"#{a['nomor'] + geser[self._ke]}{huruf} {a[besaran]['cm'] - m[kunci]:+.1f}")
        self.status.set(f"Tersimpan ke {tempat}." + (f" Selisih sistem - meteran (cm): {', '.join(beda)}" if beda else ""))

    # ------------------------------------------------------------ hitung / video
    def hitung(self) -> None:
        if self.sesi is None:
            self.status.set("Buka sebuah frame ekspor dulu.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        sesi = self.sesi
        self.btn_hitung.configure(state="disabled", text="Menghitung…")
        self.status.set("Menjalankan pelacak pada seluruh rekaman (model akhir, mekanisme tangga)…")
        self._progres, self._galat = 0, None

        def kerja():
            try:
                _modul_ukuran().hitung(sesi, video=True, progres=lambda n: setattr(self, "_progres", n))
            except Exception as e:                                  # noqa: BLE001
                baris = [b for b in str(e).strip().splitlines() if b.strip()]
                self._galat = f"{type(e).__name__}: {baris[-1][:200] if baris else ''}"

        self._thread = threading.Thread(target=kerja, daemon=True)
        self._thread.start()
        self.induk.after(500, self._pantau)

    def _pantau(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.status.set(f"Menghitung… {self._progres or 0} frame")
            self.induk.after(500, self._pantau)
            return
        self.btn_hitung.configure(state="normal", text="Hitung nomor dari video")
        if self._galat:
            self.status.set(f"Gagal: {self._galat}")
            return
        self.sistem = _modul_ukuran().baca(self.sesi) if self.sesi else None
        self._muat_meteran()
        self._segarkan_tanda()
        self.status.set("Selesai. Nomor ditandai di gambar; cek urutannya dengan ▶ Video bernomor.")

    def putar_video(self) -> None:
        if self.sesi is None:
            return
        video = self.sesi / "derived" / "ukuran_sistem.mp4"
        if not video.exists():
            self.status.set("Video bernomor belum ada; tekan Hitung nomor dari video.")
            return
        try:
            subprocess.Popen(["xdg-open", str(video)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            self.status.set(f"Tidak dapat membuka pemutar video: {e}")
