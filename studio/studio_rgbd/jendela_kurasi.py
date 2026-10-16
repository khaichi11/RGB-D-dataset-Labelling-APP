"""Jendela "Kurasi dataset": periksa frame yang dipilih/dibuang kurasi (kurasi.py) sebelum dilatih.

Dibuka dari tab Split (tombol "🧹 Kurasi dataset…"). Frame tampil sebagai kartu
bergambar dengan mask riser (merah) dan tread (hijau), dibagi per halaman agar
ringan walau ribuan frame.

- Bingkai hijau = dipilih untuk latih, merah = dibuang (alasan di bawah gambar),
  oranye = dipilih tetapi perlu dilihat (model CV sangat berbeda dengan label).
- Klik kartu = balik keputusan (tersimpan sebagai keputusan manual, bertanda ✋);
  klik kanan = kembalikan ke keputusan aturan; klik dua kali = buka di tab Label.
- "Hitung ulang" menjalankan kurasi pada semua frame siap latih (rekaman ✔, label
  terperiksa); keputusan manual dipertahankan.
"""
from __future__ import annotations

import collections
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from . import kurasi as KU
except ImportError:
    import kurasi as KU

BG, PANEL, INK, MUTED, LINE = "#F3EEE7", "#FFFDFC", "#382D29", "#786962", "#DED2C8"
HIJAU, MERAH, ORANYE, BIRU = "#1E8E3E", "#C5221F", "#E8710A", "#1A73E8"
W, H, KOLOM, BARIS = 208, 117, 6, 4
FILTER = ("semua", "dipilih", "dibuang", "buram", "mirip", "perlu dilihat", "latar, mungkin ada tangga",
          "label mungkin tak sengaja", "latar (tanpa tangga)", "bertangga", "diubah manual")


def _gambar(d: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(d / "color_raw.png"))
    if bgr is None:
        return None
    im = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    if im.mean() < 40:                                     # frame malam dicerahkan agar isinya terlihat
        im = np.clip(im.astype(np.float32) * (90.0 / max(im.mean(), 5.0)), 0, 255).astype(np.uint8)
    for nama, warna in (("mask_acuan.png", (60, 200, 60)), ("mask_objek.png", (230, 60, 60))):
        m = cv2.imread(str(d / nama), 0)
        if m is not None:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0
            im[m] = (im[m] * 0.5 + np.array(warna) * 0.5).astype(np.uint8)
    return im


class JendelaKurasi(tk.Toplevel):
    def __init__(self, studio, frames_siap) -> None:
        super().__init__(studio)
        self.studio, self.akar = studio, Path(studio.root_data)
        self.rek_dir = self.akar / "rekaman" / "tangga_naik"
        self.frames_siap = frames_siap                     # fungsi -> daftar frame siap latih (tab Split)
        self.title("Kurasi dataset latih")
        self.configure(bg=BG)
        self.geometry(f"{min(1380, self.winfo_screenwidth() - 40)}x{min(900, self.winfo_screenheight() - 80)}")
        self.data = KU.baca(self.akar)
        self.kunci: list[str] = []
        self.halaman = 0
        self._foto: dict[str, ImageTk.PhotoImage] = {}
        self._kartu: dict[str, dict] = {}
        self.filter = tk.StringVar(value="semua")
        self.rek = tk.StringVar(value="semua rekaman")
        self.waktu = tk.StringVar(value="semua waktu")
        self.info = tk.StringVar()
        self.status = tk.StringVar()
        self._q: queue.Queue = queue.Queue()            # utas latar -> utas Tk (gambar, progres, hasil)
        self._ui()
        self.after(100, self._poll)
        self.bind("<Prior>", lambda e: self.geser_halaman(-1))
        self.bind("<Next>", lambda e: self.geser_halaman(1))
        if not self.data.get("frame"):
            self.status.set("Belum ada kurasi. Tekan ↻ Hitung ulang kurasi.")
        self.saring()

    # ------------------------------------------------------------ tata letak
    def _tombol(self, induk, teks, perintah, warna="#E8DDD5", fg=INK, lebar=150):
        if hasattr(self.studio, "tombol_ringkas"):
            return self.studio.tombol_ringkas(induk, teks, perintah, warna, fg, width=lebar)
        return tk.Button(induk, text=teks, command=perintah, bg=warna, fg=fg, relief="flat")

    def _ui(self) -> None:
        atas = tk.Frame(self, bg=BG)
        atas.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(atas, text="Hijau = dipilih untuk latih · Merah = dibuang · Oranye = dipilih, perlu dilihat · ✋ = "
                            "keputusanmu. Klik = balik keputusan, klik kanan = kembali ke aturan, klik dua kali = "
                            "buka di tab Label. PageUp/PageDown = halaman.",
                 bg=BG, fg=MUTED, wraplength=1300, justify="left").pack(anchor="w")
        tk.Label(atas, textvariable=self.info, bg=BG, fg=INK, font=("Segoe UI", 9, "bold"), justify="left",
                 wraplength=1300).pack(anchor="w", pady=(4, 0))
        baris = tk.Frame(self, bg=BG)
        baris.pack(fill="x", padx=10, pady=4)
        for var, nilai, lebar in ((self.filter, FILTER, 18), (self.waktu, ("semua waktu", "pagi", "siang", "malam"), 11)):
            cb = ttk.Combobox(baris, textvariable=var, values=nilai, state="readonly", width=lebar)
            cb.pack(side="left", padx=(0, 6)); cb.bind("<<ComboboxSelected>>", lambda e: self.saring())
        self.cb_rek = ttk.Combobox(baris, textvariable=self.rek, state="readonly", width=26)
        self.cb_rek.pack(side="left", padx=(0, 6)); self.cb_rek.bind("<<ComboboxSelected>>", lambda e: self.saring())
        self._tombol(baris, "◀", lambda: self.geser_halaman(-1), lebar=40).pack(side="left")
        self.lbl_hal = tk.Label(baris, text="", bg=BG, fg=INK, width=16)
        self.lbl_hal.pack(side="left")
        self._tombol(baris, "▶", lambda: self.geser_halaman(1), lebar=40).pack(side="left")
        self._tombol(baris, "↻ Hitung ulang kurasi", self.hitung_ulang, "#6B8B63", "white", 180).pack(side="right")
        tk.Label(self, textvariable=self.status, bg=BG, fg=MUTED).pack(anchor="w", padx=10)
        self.isi = tk.Frame(self, bg=BG)
        self.isi.pack(fill="both", expand=True, padx=10, pady=6)

    def _poll(self) -> None:
        try:
            while True:
                jenis, isi = self._q.get_nowait()
                if jenis == "gambar":
                    self._pasang(*isi)
                elif jenis == "status":
                    self.status.set(isi)
                elif jenis == "selesai":
                    self._selesai(isi)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    # ------------------------------------------------------------ data
    def _final(self, k: str) -> bool | None:
        return KU.dipilih(self.data, k)

    def _cocok(self, k: str, v: dict) -> bool:
        f = self.filter.get()
        akhir = self._final(k)
        alasan = " ".join(v.get("alasan", []))
        if f == "dipilih" and not akhir: return False
        if f == "dibuang" and akhir: return False
        if f == "buram" and "buram" not in alasan: return False
        if f == "mirip" and "mirip" not in alasan: return False
        if f == "perlu dilihat" and "perlu dilihat" not in alasan: return False
        if f == "latar, mungkin ada tangga" and "latar, model CV melihat tangga" not in alasan: return False
        if f == "label mungkin tak sengaja" and not any(s in alasan for s in (
                "tidak melihat tangga", "mask sangat kecil", "potongan mask kecil")): return False
        if f == "latar (tanpa tangga)" and not v["info"].get("latar"): return False
        if f == "bertangga" and v["info"].get("latar"): return False
        if f == "diubah manual" and k not in self.data.get("manual", {}): return False
        if self.waktu.get() != "semua waktu" and v["info"].get("waktu") != self.waktu.get(): return False
        if self.rek.get() != "semua rekaman" and not k.startswith("TANGGA_NAIK_" + self.rek.get() + "/"): return False
        return True

    def saring(self) -> None:
        frame = self.data.get("frame", {})
        rek = sorted({k.split("/")[0].replace("TANGGA_NAIK_", "") for k in frame})
        self.cb_rek.configure(values=["semua rekaman"] + rek)
        self.kunci = [k for k in sorted(frame) if self._cocok(k, frame[k])]
        self.halaman = 0
        self._ringkas()
        self.gambar_halaman()

    def _ringkas(self) -> None:
        frame = self.data.get("frame", {})
        if not frame:
            self.info.set("")
            return
        pilih = [k for k in frame if self._final(k)]
        alasan = collections.Counter(("mirip" if a.startswith("mirip") else a.split(":")[0])
                                     for v in frame.values() for a in v["alasan"])
        waktu = collections.Counter(frame[k]["info"]["waktu"] for k in pilih)
        latar = sum(1 for k in pilih if frame[k]["info"]["latar"])
        self.info.set(f"{len(pilih)} dari {len(frame)} frame dipilih untuk latih · dibuang: {alasan.get('buram', 0)} "
                      f"buram, {alasan.get('mirip', 0)} mirip frame sebelumnya · {alasan.get('perlu dilihat', 0)} perlu "
                      f"dilihat · {len(self.data.get('manual', {}))} keputusan manual · terpilih per waktu: "
                      + ", ".join(f"{w} {n}" for w, n in sorted(waktu.items()))
                      + f" · latar {latar} ({100 * latar / max(len(pilih), 1):.0f}%) · tampil: {len(self.kunci)}")

    def geser_halaman(self, arah: int) -> None:
        n_hal = max(1, -(-len(self.kunci) // (KOLOM * BARIS)))
        self.halaman = min(max(0, self.halaman + arah), n_hal - 1)
        self.gambar_halaman()

    def gambar_halaman(self) -> None:
        for w in self.isi.winfo_children():
            w.destroy()
        self._kartu.clear()
        per = KOLOM * BARIS
        n_hal = max(1, -(-len(self.kunci) // per))
        self.lbl_hal.configure(text=f"hal {self.halaman + 1}/{n_hal}")
        tampil = self.kunci[self.halaman * per:(self.halaman + 1) * per]
        for i, k in enumerate(tampil):
            self._buat_kartu(i, k)
        threading.Thread(target=self._muat_gambar, args=(list(tampil),), daemon=True).start()

    def _buat_kartu(self, i: int, k: str) -> None:
        kartu = tk.Frame(self.isi, bg=PANEL, highlightthickness=4, cursor="hand2")
        kartu.grid(row=i // KOLOM, column=i % KOLOM, padx=4, pady=4, sticky="n")
        gambar = tk.Label(kartu, bg="#222222", width=28, height=7, text="memuat…", fg="#999999")
        if k in self._foto:
            gambar.configure(image=self._foto[k], width=W, height=H)
        gambar.pack()
        teks = tk.Label(kartu, text="", bg=PANEL, fg=INK, font=("Segoe UI", 8), anchor="w", justify="left",
                        wraplength=W - 4, width=28)
        teks.pack(fill="x", padx=3, pady=(2, 3))
        self._kartu[k] = {"kartu": kartu, "gambar": gambar, "teks": teks}
        for w in (kartu, gambar, teks):
            w.bind("<Button-1>", lambda e, k=k: self.balik(k))
            w.bind("<Button-3>", lambda e, k=k: self.kembali_aturan(k))
            w.bind("<Double-Button-1>", lambda e, k=k: self.buka_di_label(k))
        self._warnai(k)

    def _warnai(self, k: str) -> None:
        v = self.data["frame"].get(k, {"alasan": [], "info": {}})
        akhir = self._final(k)
        manual = k in self.data.get("manual", {})
        perlu = any(a.startswith("perlu dilihat") for a in v["alasan"])
        warna = (ORANYE if perlu else HIJAU) if akhir else MERAH
        rek, nama = k.split("/")
        info = v.get("info", {})
        baris2 = ("✋ " if manual else "") + ("DIPILIH" if akhir else "DIBUANG") + (" · latar" if info.get("latar") else "")
        self._kartu[k]["kartu"].configure(highlightbackground=warna, highlightcolor=warna)
        self._kartu[k]["teks"].configure(text=f"{rek[-15:]} {nama[-6:]} · {info.get('waktu', '')}\n{baris2}"
                                         + (f"\n{'; '.join(v['alasan'])}" if v["alasan"] else ""))

    def _muat_gambar(self, daftar: list[str]) -> None:
        for k in daftar:
            if k in self._foto:
                continue
            rek, nama = k.split("/")
            try:
                img = _gambar(self.rek_dir / rek / "exports" / "frames" / nama)
            except Exception:                                    # noqa: BLE001 -- kartu tetap tanpa gambar
                img = None
            if img is None:
                continue
            self._q.put(("gambar", (k, img)))

    def _pasang(self, k: str, img: np.ndarray) -> None:
        if not self.winfo_exists():
            return
        self._foto[k] = ImageTk.PhotoImage(Image.fromarray(img))
        if k in self._kartu:
            self._kartu[k]["gambar"].configure(image=self._foto[k], width=W, height=H)

    # ------------------------------------------------------------ aksi
    def balik(self, k: str) -> None:
        self.data.setdefault("manual", {})[k] = not bool(self._final(k))
        KU.tulis(self.akar, self.data)
        if k in self._kartu:
            self._warnai(k)
        self._ringkas()

    def kembali_aturan(self, k: str) -> None:
        if self.data.get("manual", {}).pop(k, None) is not None:
            KU.tulis(self.akar, self.data)
            if k in self._kartu:
                self._warnai(k)
            self._ringkas()

    def buka_di_label(self, k: str) -> None:
        rek, nama = k.split("/")
        d = self.rek_dir / rek / "exports" / "frames" / nama
        st = self.studio
        st.sesi = self.rek_dir / rek
        st._muat_catatan_ui()
        st.tabs.select(st.tab_label)
        st.muat_frame()
        if d in st.frame_paths:
            st._pilih_indeks_frame(st.frame_paths.index(d))
        st.lift()

    def hitung_ulang(self) -> None:
        frames = self.frames_siap()
        if not frames:
            self.status.set("Tidak ada frame siap latih (rekaman ✔ dengan label terperiksa).")
            return
        self.status.set(f"Menghitung kurasi {len(frames)} frame…")
        # Dice out-of-fold CV terakhir (bila ada) untuk menandai "perlu dilihat".
        dice = KU.dice_cv_dari(Path(__file__).resolve().parents[2]
                               / "model/aplikasi_tangga/runs/cv_publik_gabung/hasil_cv.json")

        def kerja():
            try:
                data = KU.kurasi(self.akar, frames, dice,
                                 progres=lambda n, t: self._q.put(("status", f"Menghitung kurasi {n}/{t}…")))
                self._q.put(("selesai", data))
            except Exception as e:                               # noqa: BLE001
                self._q.put(("status", f"Kurasi gagal: {e}"))
        threading.Thread(target=kerja, daemon=True).start()

    def _selesai(self, data: dict) -> None:
        if not self.winfo_exists():
            return
        self.data = data
        self.status.set(f"Kurasi selesai: {data.get('dibuat', '')}. Tersimpan di {self.akar / 'kurasi.json'}")
        self.saring()
