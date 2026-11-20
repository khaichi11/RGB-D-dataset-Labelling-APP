"""Jendela "Versi data": potret dataset latih (v1, v2, ...) beserta hasil CV tiap versi.

Dibuka dari tab Split (tombol "📦 Versi data…"). Inti logikanya di repo model
(``rgbd_convnext/data/versi_data.py``), sama dengan yang dipakai skrip latih dan CV.

- Tabel: versi, tanggal, frame latih, latar, abaikan, dibuang kurasi, hasil CV (Dice out-of-fold), model.
- "Simpan versi sekarang": potret frame latih saat ini (rekaman ✔, label terperiksa, tanpa abaikan, tanpa
  rekaman uji, dikurangi frame yang dibuang Kurasi bila dicentang) + arsip label, agar bisa dilatih/diulang
  persis dan dibandingkan.
- "Bandingkan": pilih dua versi (Ctrl+klik), atau satu versi untuk dibanding keadaan sekarang.
- Hasil CV diisi sesudah pelatihan (``bukti/final_d435/versi_data/hasil.json``); meteran uji tidak pernah
  dipakai memilih versi.
"""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, simpledialog, ttk

try:
    from .segmentasi_convnext_depth import akar_aplikasi
except ImportError:
    from segmentasi_convnext_depth import akar_aplikasi

BG, PANEL, INK, MUTED = "#F3EEE7", "#FFFDFC", "#382D29", "#786962"
KOLOM = (("versi", "Versi", 130), ("dibuat", "Dibuat", 130), ("latih", "Frame latih", 80), ("latar", "Latar", 60),
         ("abaikan", "Abaikan", 60), ("kurasi", "Dibuang kurasi", 95), ("dice", "Dice CV", 65),
         ("bertangga", "Dice bertangga", 95), ("model", "Model", 230))


def _V():
    akar = str(akar_aplikasi())
    if akar not in sys.path:
        sys.path.insert(0, akar)
    from rgbd_convnext.data import versi_data
    return versi_data


class JendelaVersi(tk.Toplevel):
    def __init__(self, studio) -> None:
        super().__init__(studio)
        self.studio = studio
        self.title("Versi data latih")
        self.configure(bg=BG)
        self.geometry(f"{min(1150, self.winfo_screenwidth() - 40)}x{min(680, self.winfo_screenheight() - 80)}")
        self.V = _V()
        self._q: queue.Queue = queue.Queue()
        self.pakai_kurasi = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="")
        self._ui()
        self.muat()
        self.after(150, self._poll)

    def _tombol(self, induk, teks, perintah, warna="#E8DDD5", fg=INK, lebar=170):
        if hasattr(self.studio, "tombol_ringkas"):
            return self.studio.tombol_ringkas(induk, teks, perintah, warna, fg, width=lebar)
        return tk.Button(induk, text=teks, command=perintah, bg=warna, fg=fg, relief="flat")

    def _ui(self) -> None:
        tk.Label(self, text="Satu versi = potret frame latih + label saat itu. Latih tiap versi dengan CV yang sama, "
                            "lalu bandingkan Dice-nya di sini. Uji ukur (meteran) hanya dijalankan sekali untuk versi "
                            "yang terpilih, tidak dipakai memilih.",
                 bg=BG, fg=MUTED, wraplength=1100, justify="left").pack(anchor="w", padx=10, pady=(10, 4))
        baris = tk.Frame(self, bg=BG)
        baris.pack(fill="x", padx=10, pady=4)
        self._tombol(baris, "💾 Simpan versi sekarang…", self.simpan, "#6B8B63", "white", 200).pack(side="left")
        tk.Checkbutton(baris, text="kurangi frame yang dibuang Kurasi", variable=self.pakai_kurasi, bg=BG,
                       fg=INK, activebackground=BG, selectcolor=PANEL).pack(side="left", padx=8)
        self._tombol(baris, "⇄ Bandingkan", self.bandingkan, lebar=130).pack(side="left", padx=4)
        self._tombol(baris, "🔍 Label berubah sejak versi", self.cek_berubah, lebar=210).pack(side="left", padx=4)
        self._tombol(baris, "↻ Muat ulang", self.muat, lebar=110).pack(side="right")
        f = tk.Frame(self, bg=BG)
        f.pack(fill="both", expand=True, padx=10, pady=4)
        self.tree = ttk.Treeview(f, columns=[k for k, _, _ in KOLOM], show="headings", height=8, selectmode="extended")
        for k, judul, lebar in KOLOM:
            self.tree.heading(k, text=judul)
            self.tree.column(k, width=lebar, anchor="w" if k in ("versi", "dibuat", "model") else "center")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._rinci())
        self.teks = tk.Text(f, height=14, bg=PANEL, fg=INK, wrap="word", relief="flat", font=("DejaVu Sans Mono", 9))
        self.teks.pack(fill="both", expand=True, pady=(6, 0))
        tk.Label(self, textvariable=self.status, bg=BG, fg=MUTED).pack(anchor="w", padx=10, pady=(0, 8))

    def _tulis(self, isi: str) -> None:
        self.teks.configure(state="normal")
        self.teks.delete("1.0", "end")
        self.teks.insert("1.0", isi)
        self.teks.configure(state="disabled")

    def muat(self) -> None:
        self.versi = {v["versi"]: v for v in self.V.daftar(self.studio.root_data)}
        self.tree.delete(*self.tree.get_children())
        for n, v in self.versi.items():
            r, h = v["ringkas"], v["hasil"] or {}
            self.tree.insert("", "end", iid=n, values=(
                n, v["dibuat"].replace("T", " ")[:16], r["frame_latih"], r["latar_latih"], r["abaikan"],
                r["dibuang_kurasi"], h.get("dice_oof", "belum dilatih"), h.get("dice_bertangga", "-"),
                h.get("model", "-")))
        self.status.set(f"{len(self.versi)} versi. Arsip: {self.studio.root_data}/versi_data/")

    def _rinci(self) -> None:
        pilih = self.tree.selection()
        if len(pilih) != 1:
            return
        v = self.versi[pilih[0]]
        h = v["hasil"] or {}
        isi = [f"{v['versi']} ({v['dibuat']})", f"Catatan: {v['catatan']}", f"Ringkas: {v['ringkas']}",
               f"Arsip label: {'ada' if v['arsip'] else 'tidak ada di komputer ini (hanya manifest di repo model)'}"]
        if h:
            isi += ["", "Hasil:"] + [f"  {k}: {x}" for k, x in h.items()]
        self._tulis("\n".join(isi))

    def simpan(self) -> None:
        bawaan = f"v{len(self.versi) + 1}_{datetime.now():%Y%m%d}"
        nama = simpledialog.askstring("Simpan versi", "Nama versi:", initialvalue=bawaan, parent=self)
        if not nama:
            return
        catatan = simpledialog.askstring("Simpan versi", "Catatan (apa yang berubah dibanding versi sebelumnya):",
                                         parent=self)
        if catatan is None:
            return
        kur = self.pakai_kurasi.get()
        self.status.set(f"Menyimpan versi {nama}…")

        def kerja():
            try:
                self._q.put(("tersimpan", self.V.simpan(nama.strip(), catatan.strip(), self.studio.root_data, kur)))
            except Exception as e:                                   # noqa: BLE001
                self._q.put(("galat", str(e)))
        threading.Thread(target=kerja, daemon=True).start()

    def bandingkan(self) -> None:
        pilih = list(self.tree.selection())
        if len(pilih) not in (1, 2):
            messagebox.showinfo("Bandingkan", "Pilih dua versi (Ctrl+klik), atau satu versi untuk dibanding "
                                               "keadaan sekarang.", parent=self)
            return
        self.status.set("Membandingkan…")
        kur = self.pakai_kurasi.get()          # variabel Tk hanya dibaca di utas utama

        def kerja():
            try:
                a = self.V.baca(pilih[0], self.studio.root_data)
                if len(pilih) == 2:
                    b, nb = self.V.baca(pilih[1], self.studio.root_data), pilih[1]
                else:
                    b, nb = {"frame": self.V.frame_latih(self.studio.root_data, kur)["frame"]}, "sekarang"
                self._q.put(("teks", self.V.teks_banding(pilih[0], nb, self.V.banding(a, b))))
            except Exception as e:                                   # noqa: BLE001
                self._q.put(("galat", str(e)))
        threading.Thread(target=kerja, daemon=True).start()

    def cek_berubah(self) -> None:
        pilih = self.tree.selection()
        if len(pilih) != 1:
            messagebox.showinfo("Label berubah", "Pilih satu versi.", parent=self)
            return

        def kerja():
            try:
                ubah = self.V.label_berubah_sejak(pilih[0], self.studio.root_data)
                self._q.put(("teks", f"{len(ubah)} frame berlabel lain dibanding potret {pilih[0]}"
                                     + (":\n" + "\n".join(ubah[:200]) if ubah else ".")))
            except Exception as e:                                   # noqa: BLE001
                self._q.put(("galat", str(e)))
        threading.Thread(target=kerja, daemon=True).start()

    def _poll(self) -> None:
        if not self.winfo_exists():
            return
        try:
            while True:
                jenis, isi = self._q.get_nowait()
                if jenis == "tersimpan":
                    tolak = "".join(f"\n  tidak ikut: {r} ({a})" for r, a in isi["ditolak"])
                    self._tulis(f"Versi {isi['versi']} tersimpan.\n{isi['ringkas']}{tolak}")
                    self.muat()
                elif jenis == "teks":
                    self._tulis(isi)
                    self.status.set("")
                elif jenis == "galat":
                    self.status.set(isi)
                    messagebox.showerror("Versi data", isi, parent=self)
        except queue.Empty:
            pass
        self.after(150, self._poll)
