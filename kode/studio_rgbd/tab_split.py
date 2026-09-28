"""Tab Studio "7. Split Dataset": bagi rekaman/frame ke train, val, dan test.

Yang ditampilkan di sini adalah persis yang akan dilatih: daftar frame berasal
dari ``frame_bersih`` (frame lengkap, bukan sampah, dan bila dicentang hanya
yang sudah diperiksa), dan mask pratinjau dibentuk seperti ``FrameD435``
(tread lalu riser menimpa). Pembagian disimpan otomatis ke
``<dataset>/studio_rgbd/split_dataset.json`` dan dibaca skrip latih lewat
``--split``.

Pintasan saat tab ini aktif: 1 Train, 2 Val, 3 Test, 0 Abaikan, Backspace
kosongkan (berlaku pada daftar yang terakhir diklik: rekaman atau frame);
panah kanan/Space frame berikutnya, panah kiri sebelumnya.
"""
from __future__ import annotations

import queue
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from .segmentasi_convnext_depth import akar_aplikasi
    from . import unggah_hf
except ImportError:
    from segmentasi_convnext_depth import akar_aplikasi
    import unggah_hf

BG, PANEL, INK, MUTED = "#F3EEE7", "#FFFDFC", "#382D29", "#786962"
WARNA_SET = {"train": "#6B8B63", "val": "#C98A2B", "test": "#407EA3", "abaikan": "#9A8F8A", None: "#CFC6C0"}
LABEL_SET = {"train": "TRAIN", "val": "VAL", "test": "TEST", "abaikan": "ABAIKAN", None: "belum"}
RISER_RGB, TREAD_RGB = (255, 90, 90), (90, 169, 255)       # sama dengan kanvas label


def _pustaka():
    akar = str(akar_aplikasi())
    if akar not in sys.path:
        sys.path.insert(0, akar)
    from rgbd_convnext.data import split_dataset
    from rgbd_convnext.data.dataset_d435 import frame_bersih, letterbox
    return split_dataset, frame_bersih, letterbox


def _cerahkan(rgb: np.ndarray) -> np.ndarray:
    """Hanya untuk mata: model tetap menerima citra asli."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(l, (1, 99.5))
    l = np.clip((l - lo) / max(hi - lo, 8.0), 0, 1) ** 0.5
    lab[..., 0] = cv2.createCLAHE(3.0, (8, 8)).apply((l * 255).astype(np.uint8))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class TabSplit:
    def __init__(self, induk: tk.Frame, studio) -> None:
        self.studio = studio
        self.sd, self.frame_bersih, self.letterbox = _pustaka()
        self.akar = Path(studio.root_data) / "rekaman" / "tangga_naik"
        self.berkas = Path(studio.root_data) / "split_dataset.json"
        self.data = self.sd.baca(self.berkas)
        self.rekaman: list[str] = []
        self.frames: dict[str, list[Path]] = {}          # rekaman -> frame siap latih
        self.n_ekspor: dict[str, int] = {}
        self.daftar: list[Path] = []                     # frame pada daftar (setelah filter)
        self.kini: Path | None = None
        self._fokus = "rekaman"
        self._q: queue.Queue = queue.Queue()
        self._foto = None
        self._sibuk = False

        self.hanya_diperiksa = tk.BooleanVar(value=True)
        self.filter_set = tk.StringVar(value="semua")
        self.opasitas = tk.DoubleVar(value=0.45)
        self.garis = tk.BooleanVar(value=True)
        self.cerah = tk.BooleanVar(value=False)
        self.seperti_model = tk.BooleanVar(value=False)
        self._ui(induk)
        studio.bind_all("<KeyPress>", self._tombol, add="+")
        self.after = induk.after
        self.induk = induk
        self.after(100, self._poll)
        self.muat_ulang()

    # ------------------------------------------------------------------ UI
    def _ui(self, induk: tk.Frame) -> None:
        induk.columnconfigure(1, weight=1)
        induk.rowconfigure(0, weight=1)
        kiri = tk.Frame(induk, bg=BG); kiri.grid(row=0, column=0, sticky="ns", padx=(0, 10), pady=6)
        kanan = tk.Frame(induk, bg=BG); kanan.grid(row=0, column=1, sticky="nsew", pady=6)
        bawah = tk.Frame(induk, bg=PANEL); bawah.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        tk.Label(kiri, text="Rekaman (Shift/Ctrl+klik untuk memblok)", bg=BG, fg=INK,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        f = tk.Frame(kiri, bg=BG); f.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(f, columns=("set", "siap", "rincian"), show="tree headings",
                                 selectmode="extended", height=8)
        self.tree.heading("#0", text="Rekaman"); self.tree.column("#0", width=210)
        self.tree.heading("set", text="Set"); self.tree.column("set", width=70, anchor="center")
        self.tree.heading("siap", text="Siap latih"); self.tree.column("siap", width=70, anchor="e")
        self.tree.heading("rincian", text="Per set"); self.tree.column("rincian", width=120)
        for s, w in WARNA_SET.items():
            self.tree.tag_configure(str(s), foreground=w)
        gulir = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=gulir.set)
        self.tree.pack(side="left", fill="both", expand=True); gulir.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._isi_daftar())
        self.tree.bind("<Button-1>", lambda e: self._set_fokus("rekaman"), add="+")
        self._baris_tombol(kiri, "Atur REKAMAN terpilih:", self.atur_rekaman)

        tk.Label(kiri, text="Frame siap latih pada rekaman terpilih", bg=BG, fg=INK,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        baris = tk.Frame(kiri, bg=BG); baris.pack(fill="x")
        tk.Label(baris, text="Tampilkan:", bg=BG, fg=MUTED).pack(side="left")
        cb = ttk.Combobox(baris, textvariable=self.filter_set, state="readonly", width=10,
                          values=("semua", "train", "val", "test", "abaikan", "belum"))
        cb.pack(side="left", padx=4); cb.bind("<<ComboboxSelected>>", lambda e: self._isi_daftar())
        f = tk.Frame(kiri, bg=BG); f.pack(fill="both", expand=True)
        self.lb = tk.Listbox(f, height=9, selectmode="extended", exportselection=False,
                             bg="#FFF9F4", fg=INK, relief="flat", font=("DejaVu Sans Mono", 9))
        gulir = ttk.Scrollbar(f, orient="vertical", command=self.lb.yview)
        self.lb.configure(yscrollcommand=gulir.set)
        self.lb.pack(side="left", fill="both", expand=True); gulir.pack(side="left", fill="y")
        self.lb.bind("<<ListboxSelect>>", lambda e: self._pilih_frame())
        self.lb.bind("<Button-1>", lambda e: self._set_fokus("frame"), add="+")
        self._baris_tombol(kiri, "Atur FRAME terpilih (pengecualian):", self.atur_frame, ikut=True)

        kepala = tk.Frame(kanan, bg=BG); kepala.pack(fill="x")
        self.judul = tk.Label(kepala, text="Pilih rekaman di kiri", bg=BG, fg=INK, font=("Segoe UI", 11, "bold"))
        self.judul.pack(side="left")
        self.lencana = tk.Label(kepala, text="", fg="white", font=("Segoe UI", 10, "bold"), padx=8)
        self.lencana.pack(side="left", padx=8)
        self.kanvas = tk.Canvas(kanan, bg="#1E1A18", highlightthickness=0)
        self.kanvas.pack(fill="both", expand=True, pady=4)
        self.kanvas.bind("<Configure>", lambda e: self._gambar())
        kontrol = tk.Frame(kanan, bg=BG); kontrol.pack(fill="x")
        tk.Button(kontrol, text="◀ Sebelumnya", command=lambda: self.geser(-1)).pack(side="left")
        tk.Button(kontrol, text="Berikutnya ▶", command=lambda: self.geser(1)).pack(side="left", padx=4)
        tk.Scale(kontrol, from_=0, to=1, resolution=.05, orient="horizontal", variable=self.opasitas,
                 label="Opacity mask", length=160, bg=BG, highlightthickness=0,
                 command=lambda _: self._gambar()).pack(side="left", padx=8)
        for teks, var in (("Garis mask", self.garis), ("Cerahkan (tampilan saja)", self.cerah),
                          ("Seperti masukan model (letterbox 384)", self.seperti_model)):
            tk.Checkbutton(kontrol, text=teks, variable=var, bg=BG, command=self._gambar).pack(side="left")

        self.ringkas = tk.Label(bawah, text="", bg=PANEL, fg=INK, justify="left", anchor="w",
                                font=("Segoe UI", 9))
        self.ringkas.pack(side="left", fill="x", expand=True, padx=8, pady=4)
        tk.Checkbutton(bawah, text="Hanya label diperiksa", variable=self.hanya_diperiksa, bg=PANEL,
                       command=self.muat_ulang).pack(side="left")
        for teks, cmd in (("↻ Muat ulang", self.muat_ulang), ("📋 Perintah latih", self.salin_latih),
                          ("📋 Perintah uji (test)", self.salin_uji), ("☁ Push dataset ke HF", self.push_hf)):
            tk.Button(bawah, text=teks, command=cmd).pack(side="left", padx=2, pady=4)

    def _baris_tombol(self, induk, judul, aksi, ikut=False) -> None:
        tk.Label(induk, text=judul, bg=BG, fg=MUTED).pack(anchor="w", pady=(4, 0))
        baris = tk.Frame(induk, bg=BG); baris.pack(fill="x")
        pilihan = [("1 Train", "train"), ("2 Val", "val"), ("3 Test", "test"), ("0 Abaikan", "abaikan")]
        pilihan.append(("Ikut rekaman", None) if ikut else ("Kosongkan", None))
        for teks, s in pilihan:
            tk.Button(baris, text=teks, bg=WARNA_SET[s], fg="white" if s else INK, relief="flat",
                      command=lambda s=s: aksi(s)).pack(side="left", expand=True, fill="x", padx=1)

    def _set_fokus(self, nama: str) -> None:
        self._fokus = nama

    # ------------------------------------------------------------ data
    def muat_ulang(self) -> None:
        """Hitung frame siap latih tiap rekaman di thread latar (disk dataset lambat)."""
        if self._sibuk:
            return
        self._sibuk = True
        self.ringkas.config(text="Membaca daftar frame…")
        hanya = self.hanya_diperiksa.get()

        def kerja():
            try:
                rek = sorted(p.name for p in self.akar.iterdir() if p.is_dir()) if self.akar.exists() else []
                frames, n_ekspor = {}, {}
                for r in rek:
                    frames[r] = [d for d in self.frame_bersih(self.akar, [r], hanya_diperiksa=hanya)
                                 if d.parent.parent.parent.name == r]
                    ek = self.akar / r / "exports" / "frames"
                    n_ekspor[r] = sum(1 for d in ek.iterdir() if d.is_dir()) if ek.exists() else 0
                self._q.put(("muat", (rek, frames, n_ekspor)))
            except Exception as e:                           # noqa: BLE001
                self._q.put(("galat", f"Gagal membaca dataset: {e}"))
        threading.Thread(target=kerja, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                jenis, isi = self._q.get_nowait()
                if jenis == "muat":
                    self.rekaman, self.frames, self.n_ekspor = isi
                    self._sibuk = False
                    self._isi_tree()
                    self._isi_daftar()
                    self._perbarui_ringkas()
                elif jenis == "status":
                    self.ringkas.config(text=isi)
                elif jenis == "galat":
                    self._sibuk = False
                    self.ringkas.config(text=isi)
                    messagebox.showerror("Split dataset", isi)
                elif jenis == "hf_selesai":
                    self._sibuk = False
                    self.ringkas.config(text=f"Dataset terunggah: {isi}")
                    messagebox.showinfo("Hugging Face", f"Dataset terunggah:\n{isi}")
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def _set_frame(self, d: Path) -> str | None:
        return self.sd.set_frame(self.data, d)

    def _simpan(self) -> None:
        self.sd.tulis(self.berkas, self.data)
        self.data = self.sd.baca(self.berkas)

    # ------------------------------------------------------------ tampilan daftar
    def _rincian(self, r: str) -> str:
        n = {}
        for d in self.frames.get(r, []):
            s = self._set_frame(d)
            n[s] = n.get(s, 0) + 1
        if len(n) <= 1:
            return ""
        return " ".join(f"{LABEL_SET[s][0]}{v}" for s, v in n.items())

    def _isi_tree(self) -> None:
        pilih = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        for r in self.rekaman:
            s = self.data["rekaman"].get(r)
            siap = len(self.frames.get(r, []))
            self.tree.insert("", "end", iid=r, text=r.replace("TANGGA_NAIK_", ""),
                             values=(LABEL_SET[s], f"{siap}/{self.n_ekspor.get(r, 0)}", self._rincian(r)),
                             tags=(str(s),))
        ada = [r for r in pilih if self.tree.exists(r)]
        if ada:
            self.tree.selection_set(ada)

    def _perbarui_tree(self, rekaman: list[str]) -> None:
        for r in rekaman:
            if self.tree.exists(r):
                s = self.data["rekaman"].get(r)
                self.tree.item(r, values=(LABEL_SET[s], f"{len(self.frames.get(r, []))}/{self.n_ekspor.get(r, 0)}",
                                          self._rincian(r)), tags=(str(s),))

    def _isi_daftar(self, pertahankan: bool = False) -> None:
        kini = self.kini
        filt = self.filter_set.get()
        self.daftar = []
        for r in self.tree.selection():
            for d in self.frames.get(r, []):
                s = self._set_frame(d)
                if filt == "semua" or (filt == "belum" and s is None) or s == filt:
                    self.daftar.append(d)
        pos = self.lb.yview()
        self.lb.delete(0, "end")
        for i, d in enumerate(self.daftar):
            s = self._set_frame(d)
            ciri = "*" if self.data["frame"].get(self.sd.kunci_frame(d)) else " "
            self.lb.insert("end", f"{LABEL_SET[s]:>7}{ciri} {d.parent.parent.parent.name[-6:]} {d.name}")
            self.lb.itemconfig(i, fg=WARNA_SET[s])
        if pertahankan:
            self.lb.yview_moveto(pos[0])
            if kini in self.daftar:
                i = self.daftar.index(kini)
                self.lb.selection_set(i); self.lb.activate(i)
        elif self.daftar:
            self._buka(0)
        else:
            self.kini = None
            self._gambar()

    def _perbarui_ringkas(self) -> None:
        per_set = {s: [] for s in self.sd.SET}
        for r in self.rekaman:
            for d in self.frames.get(r, []):
                s = self._set_frame(d)
                if s in per_set:
                    per_set[s].append(d)
        rk = self.sd.ringkasan(per_set)
        belum = sum(1 for r in self.rekaman for d in self.frames.get(r, []) if self._set_frame(d) is None)
        teks = "   ".join(f"{LABEL_SET[s]}: {rk['frame'][s]} frame / {rk['rekaman'][s]} rekaman" for s in self.sd.SET)
        teks += f"   |   belum dibagi: {belum} frame"
        if rk["rekaman_terbagi"]:
            teks += (f"\n⚠ {len(rk['rekaman_terbagi'])} rekaman terbagi ke beberapa set "
                     f"(frame berdekatan hampir identik; angka val/test bisa terlalu optimistis): "
                     + ", ".join(r[-6:] for r in rk["rekaman_terbagi"]))
        if not rk["frame"]["val"]:
            teks += "\n⚠ Set VAL kosong: perintah latih akan memakai --pelatihan-penuh (tanpa validasi)."
        self.ringkas.config(text=teks)

    # ------------------------------------------------------------ aksi
    def atur_rekaman(self, s: str | None) -> None:
        rek = list(self.tree.selection())
        if not rek:
            messagebox.showinfo("Pilih rekaman", "Pilih satu atau beberapa rekaman dahulu."); return
        for r in rek:
            if s is None:
                self.data["rekaman"].pop(r, None)
            else:
                self.data["rekaman"][r] = s
        self._simpan()
        self._perbarui_tree(rek)
        self._isi_daftar(pertahankan=True)
        self._perbarui_ringkas()
        self._gambar()

    def atur_frame(self, s: str | None) -> None:
        pilih = [self.daftar[i] for i in self.lb.curselection()]
        if not pilih:
            messagebox.showinfo("Pilih frame", "Pilih satu atau beberapa frame dahulu."); return
        for d in pilih:
            k = self.sd.kunci_frame(d)
            if s is None or s == self.data["rekaman"].get(d.parent.parent.parent.name):
                self.data["frame"].pop(k, None)          # sama dengan set rekaman: tidak perlu pengecualian
            else:
                self.data["frame"][k] = s
        self._simpan()
        self._perbarui_tree(sorted({d.parent.parent.parent.name for d in pilih}))
        self._isi_daftar(pertahankan=True)
        self._perbarui_ringkas()
        self._gambar()

    def _tombol(self, e):
        if self.studio.tabs.select() != str(self.induk):
            return None
        if isinstance(self.studio.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox)):
            return None
        peta = {"1": "train", "2": "val", "3": "test", "0": "abaikan", "BackSpace": None}
        if e.keysym in peta:
            (self.atur_frame if self._fokus == "frame" else self.atur_rekaman)(peta[e.keysym])
            return "break"
        if e.keysym in ("Right", "space"):
            self.geser(1); return "break"
        if e.keysym == "Left":
            self.geser(-1); return "break"
        return None

    def geser(self, arah: int) -> None:
        if not self.daftar:
            return
        i = self.daftar.index(self.kini) + arah if self.kini in self.daftar else 0
        if 0 <= i < len(self.daftar):
            self._buka(i)

    def _buka(self, i: int) -> None:
        self.lb.selection_clear(0, "end"); self.lb.selection_set(i); self.lb.activate(i); self.lb.see(i)
        self.kini = self.daftar[i]
        self._gambar()

    def _pilih_frame(self) -> None:
        s = self.lb.curselection()
        if len(s) == 1:
            self.kini = self.daftar[s[0]]
            self._gambar()

    # ------------------------------------------------------------ pratinjau
    def _gambar(self) -> None:
        k = self.kanvas
        k.delete("all")
        d = self.kini
        if d is None:
            self.judul.config(text="Pilih rekaman di kiri"); self.lencana.config(text="", bg=BG)
            return
        s = self._set_frame(d)
        pengecualian = " (pengecualian frame)" if self.data["frame"].get(self.sd.kunci_frame(d)) else ""
        self.judul.config(text=f"{d.parent.parent.parent.name} / {d.name}"
                               f"   [{self.daftar.index(d) + 1}/{len(self.daftar)}]" if d in self.daftar else d.name)
        self.lencana.config(text=LABEL_SET[s] + pengecualian, bg=WARNA_SET[s])
        bgr = cv2.imread(str(d / "color_raw.png"))
        riser = cv2.imread(str(d / "mask_objek.png"), 0)
        tread = cv2.imread(str(d / "mask_acuan.png"), 0)
        if bgr is None or riser is None or tread is None:
            k.create_text(12, 12, anchor="nw", fill="white", text="Berkas frame tidak lengkap."); return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sem = np.zeros(riser.shape, np.uint8)
        sem[tread > 0] = 2
        sem[riser > 0] = 1                                 # sama dengan FrameD435
        if self.cerah.get():
            rgb = _cerahkan(rgb)
        if self.seperti_model.get():
            rgb = self.letterbox(rgb, 384, cv2.INTER_LINEAR)
            sem = self.letterbox(sem, 384, cv2.INTER_NEAREST)
        a = float(self.opasitas.get())
        out = rgb.astype(np.float32)
        for c, w in ((2, TREAD_RGB), (1, RISER_RGB)):
            m = sem == c
            out[m] = out[m] * (1 - a) + np.array(w, np.float32) * a
        out = out.astype(np.uint8)
        if self.garis.get():
            for c, w in ((2, TREAD_RGB), (1, RISER_RGB)):
                kontur, _ = cv2.findContours((sem == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(out, kontur, -1, w, 2 if not self.seperti_model.get() else 1)
        cw, ch = max(1, k.winfo_width()), max(1, k.winfo_height())
        skala = min(cw / out.shape[1], ch / out.shape[0])
        ukuran = (max(1, int(out.shape[1] * skala)), max(1, int(out.shape[0] * skala)))
        self._foto = ImageTk.PhotoImage(Image.fromarray(out).resize(
            ukuran, Image.NEAREST if self.seperti_model.get() else Image.BILINEAR))
        k.create_image(cw // 2, ch // 2, image=self._foto)

    # ------------------------------------------------------------ perintah & HF
    def _perintah_latih(self) -> str:
        app = akar_aplikasi()
        per_val = sum(1 for r in self.rekaman for d in self.frames.get(r, []) if self._set_frame(d) == "val")
        awal = app / "runs" / "latih_ulang_384" / "pra_latih" / "best.pt"
        bagian = [f'cd "{app}" &&', "./jalankan.sh", f'"{sys.executable}"', "-m rgbd_convnext.latih.d435",
                  f'--akar "{self.akar}"', f'--split "{self.berkas}"']
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
        messagebox.showinfo(judul, "Disalin ke papan klip. Tempel di terminal:\n\n" + teks)

    def salin_latih(self) -> None:
        self._salin(self._perintah_latih(), "Perintah latih")

    def salin_uji(self) -> None:
        app = akar_aplikasi()
        teks = (f'cd "{app}" && ./jalankan.sh "{sys.executable}" skrip/utama/evaluasi_segmentasi_akhir.py '
                f'--akar "{self.akar}" --split "{self.berkas}" --set test'
                + (" --hanya-diperiksa" if self.hanya_diperiksa.get() else "")
                + ' --bobot <folder-hasil-latih>/best.pt --keluar <folder-hasil-latih>/uji_test.json')
        self._salin(teks, "Perintah uji")

    def push_hf(self) -> None:
        k = unggah_hf.konfigurasi()
        if not k["token"]:
            if not masuk_hf(self.studio):
                return
            k = unggah_hf.konfigurasi()
        if self._sibuk:
            return
        if not messagebox.askyesno("Push dataset", f"Unggah seluruh frame berlabel (Parquet) ke\n"
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
    dlg.transient(induk); dlg.grab_set(); dlg.resizable(False, False)
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
    dlg.wait_window()
    return hasil["ok"]
