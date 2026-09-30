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
    from . import catatan_rekaman as CR, unggah_hf
    from .ui_bantu import kolom_gulir
except ImportError:
    from segmentasi_convnext_depth import akar_aplikasi
    import catatan_rekaman as CR
    import unggah_hf
    from ui_bantu import kolom_gulir

BG, PANEL, INK, MUTED = "#F3EEE7", "#FFFDFC", "#382D29", "#786962"      # sama dengan studio_dataset_rgbd
ACCENT, ACCENT_SOFT, GREEN, BLUE = "#754C3B", "#E9D8CC", "#6B8B63", "#407EA3"
WARNA_SET = {"train": "#6B8B63", "val": "#C98A2B", "test": "#407EA3", "abaikan": "#9A8F8A", None: "#E8DDD5"}
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
        self.catatan: dict[str, dict] = {}              # rekaman -> catatan.json (warna, catatan, scene)
        self.waktu: dict = {}                           # rekaman -> waktu mulai rekam (datetime lokal)
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
        self.rasio_train, self.rasio_val, self.rasio_test = (tk.IntVar(value=70), tk.IntVar(value=15),
                                                             tk.IntVar(value=15))
        self._ui(induk)
        studio.bind_all("<KeyPress>", self._tombol, add="+")
        self.after = induk.after
        self.induk = induk
        self.after(100, self._poll)
        self.muat_ulang()

    # ------------------------------------------------------------------ UI
    def _ui(self, induk: tk.Frame) -> None:
        """Gaya sama dengan tab lain: kartu Studio, tombol bulat, palet yang sama."""
        st = self.studio
        f = tk.Frame(induk, bg=BG); f.pack(fill="both", expand=True, padx=14, pady=14)
        kiri = kolom_gulir(f, 540, BG, padx=(0, 12))
        kanan = tk.Frame(f, bg=BG); kanan.pack(side="left", fill="both", expand=True)

        # --- kalkulator pembagian
        b, i = st.card(kiri, "Kalkulator pembagian"); b.pack(fill="x", pady=(0, 8))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x")
        tk.Label(baris, text="Rasio", bg=PANEL, fg=MUTED).pack(side="left")
        for teks, r in (("70/15/15", (70, 15, 15)), ("80/10/10", (80, 10, 10)), ("60/20/20", (60, 20, 20))):
            st.tombol_ringkas(baris, teks, lambda r=r: self._atur_rasio(*r), "#E8DDD5", INK, width=74).pack(side="left", padx=2)
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(6, 0))
        for nama, var in (("Train %", self.rasio_train), ("Val %", self.rasio_val), ("Test %", self.rasio_test)):
            tk.Label(baris, text=nama, bg=PANEL, fg=MUTED).pack(side="left", padx=(0, 3))
            sp = tk.Spinbox(baris, from_=0, to=100, width=4, textvariable=var, command=self._rasio_berubah)
            sp.pack(side="left", padx=(0, 10)); sp.bind("<KeyRelease>", lambda _e: self._rasio_berubah())
        self.tabel = tk.Frame(i, bg=PANEL); self.tabel.pack(fill="x", pady=(8, 0))
        self.rekomendasi = tk.Label(i, text="", bg=PANEL, fg=ACCENT, justify="left", anchor="w",
                                    wraplength=470, font=("Segoe UI", 9))
        self.rekomendasi.pack(fill="x", pady=(6, 0))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(6, 0))
        st.tombol_ringkas(baris, "⚖ Bagi otomatis per rekaman/scene", lambda: self.bagi_otomatis("grup"),
                          GREEN, width=250).pack(side="left", fill="x", expand=True, padx=(0, 3))
        st.tombol_ringkas(baris, "Bagi per blok frame", lambda: self.bagi_otomatis("blok"),
                          "#E8DDD5", INK, width=150).pack(side="left", fill="x", expand=True)
        tk.Checkbutton(i, text="Hanya label yang sudah divalidasi manual (disarankan)", variable=self.hanya_diperiksa,
                       bg=PANEL, fg=INK, selectcolor=PANEL, activebackground=PANEL,
                       command=self.muat_ulang).pack(anchor="w", pady=(6, 0))

        # --- rekaman & scene
        b, i = st.card(kiri, "Rekaman & scene  (Shift/Ctrl+klik memblok)"); b.pack(fill="x", pady=(0, 8))
        f2 = tk.Frame(i, bg=PANEL); f2.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(f2, columns=("set", "siap", "rincian", "waktu"), show="tree headings",
                                 selectmode="extended", height=8)
        self.tree.heading("#0", text="Rekaman"); self.tree.column("#0", width=170)
        self.tree.heading("set", text="Set"); self.tree.column("set", width=66, anchor="center")
        self.tree.heading("siap", text="Siap latih"); self.tree.column("siap", width=66, anchor="e")
        self.tree.heading("rincian", text="Per set"); self.tree.column("rincian", width=80)
        self.tree.heading("waktu", text="Direkam"); self.tree.column("waktu", width=120)
        for s_, w in WARNA_SET.items():
            self.tree.tag_configure(str(s_), foreground=w)
        for nama, w in CR.WARNA.items():
            self.tree.tag_configure(f"stabilo_{nama}", background=w)
        self.tree.tag_configure("scene", font=("Segoe UI", 9, "italic"))
        gulir = ttk.Scrollbar(f2, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=gulir.set)
        self.tree.pack(side="left", fill="both", expand=True); gulir.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda e: (self._isi_daftar(), self._tampil_catatan()))
        self.tree.bind("<Button-1>", lambda e: self._set_fokus("rekaman"), add="+")
        self.info_rek = tk.Label(i, text="", bg=PANEL, fg=INK, justify="left", anchor="w", wraplength=470,
                                 font=("Segoe UI", 9))
        self.info_rek.pack(fill="x", pady=(4, 0))
        self._baris_tombol(i, "Atur rekaman / scene terpilih", self.atur_rekaman)

        # --- frame
        b, i = st.card(kiri, "Frame siap latih"); b.pack(fill="x", pady=(0, 8))
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x")
        tk.Label(baris, text="Tampilkan", bg=PANEL, fg=MUTED).pack(side="left")
        cb = ttk.Combobox(baris, textvariable=self.filter_set, state="readonly", width=10,
                          values=("semua", "train", "val", "test", "abaikan", "belum"))
        cb.pack(side="left", padx=4); cb.bind("<<ComboboxSelected>>", lambda e: self._isi_daftar())
        f2 = tk.Frame(i, bg=PANEL); f2.pack(fill="both", expand=True, pady=(4, 0))
        self.lb = tk.Listbox(f2, height=9, selectmode="extended", exportselection=False, activestyle="none",
                             bg="#FFF9F4", fg=INK, relief="flat", selectbackground=ACCENT_SOFT,
                             selectforeground=INK, font=("DejaVu Sans Mono", 9))
        gulir = ttk.Scrollbar(f2, orient="vertical", command=self.lb.yview)
        self.lb.configure(yscrollcommand=gulir.set)
        self.lb.pack(side="left", fill="both", expand=True); gulir.pack(side="left", fill="y")
        self.lb.bind("<<ListboxSelect>>", lambda e: self._pilih_frame())
        self.lb.bind("<Button-1>", lambda e: self._set_fokus("frame"), add="+")
        self._baris_tombol(i, "Atur frame terpilih (pengecualian)", self.atur_frame, ikut=True)

        # --- kanan: aksi (dipasang dulu dari bawah) lalu pratinjau
        b, i = st.card(kanan, "Ringkasan & aksi"); b.pack(side="bottom", fill="x", pady=(8, 0))
        self.ringkas = tk.Label(i, text="", bg=PANEL, fg=INK, justify="left", anchor="w", font=("Segoe UI", 9))
        self.ringkas.pack(fill="x")
        baris = tk.Frame(i, bg=PANEL); baris.pack(fill="x", pady=(6, 0))
        for teks, cmd, w, warna, fg in (("↻ Muat ulang", self.muat_ulang, 100, "#E8DDD5", INK),
                                        ("📄 Ringkasan CSV", self.ringkasan_csv, 130, "#E8DDD5", INK),
                                        ("📋 Perintah latih", self.salin_latih, 130, "#E8DDD5", INK),
                                        ("📋 Perintah uji (test)", self.salin_uji, 150, "#E8DDD5", INK),
                                        ("⬇ Tarik dari HF", self.tarik_hf, 130, "#E8DDD5", INK),
                                        ("☁ Push dataset ke HF", self.push_hf, 160, BLUE, "white")):
            st.tombol_ringkas(baris, teks, cmd, warna, fg, width=w).pack(side="left", fill="x", expand=True, padx=2)

        b, i = st.card(kanan, "Pratinjau (persis seperti yang dilatih)"); b.pack(fill="both", expand=True)
        kepala = tk.Frame(i, bg=PANEL); kepala.pack(fill="x")
        self.judul = tk.Label(kepala, text="Pilih rekaman di kiri", bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold"))
        self.judul.pack(side="left")
        self.lencana = tk.Label(kepala, text="", bg=PANEL, fg="white", font=("Segoe UI", 9, "bold"), padx=8)
        self.lencana.pack(side="left", padx=8)
        centang = tk.Frame(i, bg=PANEL); centang.pack(side="bottom", fill="x")
        kontrol = tk.Frame(i, bg=PANEL); kontrol.pack(side="bottom", fill="x", pady=(4, 0))
        self.kanvas = tk.Canvas(i, bg="#27201E", highlightthickness=0, height=200)
        self.kanvas.pack(fill="both", expand=True, pady=(6, 0))
        self.kanvas.bind("<Configure>", lambda e: self._gambar())
        st.tombol_ringkas(kontrol, "◀ Sebelumnya", lambda: self.geser(-1), "#E8DDD5", INK, width=110).pack(side="left")
        st.tombol_ringkas(kontrol, "Berikutnya ▶", lambda: self.geser(1), "#E8DDD5", INK, width=110).pack(side="left", padx=4)
        tk.Scale(kontrol, from_=0, to=1, resolution=.05, orient="horizontal", variable=self.opasitas,
                 label="Opacity mask", length=170, bg=PANEL, fg=INK, highlightthickness=0,
                 command=lambda _: self._gambar()).pack(side="left", padx=8)
        for teks, var in (("Garis mask", self.garis), ("Cerahkan (tampilan saja)", self.cerah),
                          ("Seperti masukan model (letterbox 384)", self.seperti_model)):
            tk.Checkbutton(centang, text=teks, variable=var, bg=PANEL, fg=INK, selectcolor=PANEL,
                           activebackground=PANEL, command=self._gambar).pack(side="left")

    def _baris_tombol(self, induk, judul, aksi, ikut=False) -> None:
        tk.Label(induk, text=judul, bg=PANEL, fg=MUTED).pack(anchor="w", pady=(6, 2))
        baris = tk.Frame(induk, bg=PANEL); baris.pack(fill="x")
        pilihan = [("1 Train", "train"), ("2 Val", "val"), ("3 Test", "test"), ("0 Abaikan", "abaikan")]
        pilihan.append(("Ikut rekaman", None) if ikut else ("Kosongkan", None))
        for teks, s_ in pilihan:
            self.studio.tombol_ringkas(baris, teks, lambda s_=s_: aksi(s_), WARNA_SET[s_],
                                       "white" if s_ else INK, width=86).pack(side="left", expand=True, fill="x", padx=1)

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
                frames, n_ekspor, catatan, waktu = {}, {}, {}, {}
                for r in rek:
                    catatan[r] = CR.baca(self.akar / r)
                    w = CR.waktu_rekaman(self.akar / r)
                    waktu[r] = w[0] if w else None
                    frames[r] = [d for d in self.frame_bersih(self.akar, [r], hanya_diperiksa=hanya)
                                 if d.parent.parent.parent.name == r]
                    ek = self.akar / r / "exports" / "frames"
                    n_ekspor[r] = sum(1 for d in ek.iterdir() if d.is_dir()) if ek.exists() else 0
                self._q.put(("muat", (rek, frames, n_ekspor, catatan, waktu)))
            except Exception as e:                           # noqa: BLE001
                self._q.put(("galat", f"Gagal membaca dataset: {e}"))
        threading.Thread(target=kerja, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                jenis, isi = self._q.get_nowait()
                if jenis == "muat":
                    self.rekaman, self.frames, self.n_ekspor, self.catatan, self.waktu = isi
                    self._sibuk = False
                    self._isi_tree()
                    if not self.tree.selection() and self.rekaman:
                        self.tree.selection_set([self.rekaman[0]])     # pratinjau tidak kosong saat tab dibuka
                    self._isi_daftar()
                    self._perbarui_ringkas()
                elif jenis == "status":
                    self.ringkas.config(text=isi)
                elif jenis == "galat":
                    self._sibuk = False
                    self.ringkas.config(text=isi)
                    messagebox.showerror("Split dataset", isi)
                elif jenis == "csv_selesai":
                    self._sibuk = False
                    self.ringkas.config(text=f"Ringkasan rekaman ditulis: {isi[0]}")
                    messagebox.showinfo("Ringkasan rekaman", f"Ditulis ke:\n{isi[0]}\n\n{isi[1]}")
                elif jenis == "tarik_selesai":
                    self._sibuk = False
                    teks = (f"Tarik selesai: {isi['frame_baru']} frame baru, {isi['label_diperbarui']} label "
                            f"diperbarui dari HF, {isi['label_lokal_lebih_baru']} label lokal lebih baru dipertahankan.")
                    self.ringkas.config(text=teks)
                    messagebox.showinfo("Hugging Face", teks + "\n\nPilih rekamannya di tab 2. Tinjau untuk melabel.")
                    self.muat_ulang()
                    self.studio.muat_daftar()
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

    def _scene(self, r: str) -> list[dict]:
        return (self.catatan.get(r) or {}).get("scene", [])

    def _frames_iid(self, iid: str) -> list[Path]:
        """Frame siap latih untuk satu baris pohon: rekaman penuh atau satu scene."""
        if "::" not in iid:
            return self.frames.get(iid, [])
        r, i = iid.split("::")
        sc = self._scene(r)[int(i)]
        return [d for d in self.frames.get(r, [])
                if int(sc["awal"]) <= (CR.indeks_frame(d.name) or -1) <= int(sc["akhir"])]

    def _nilai_baris(self, iid: str) -> tuple:
        frames = self._frames_iid(iid)
        n: dict = {}
        for d in frames:
            s = self._set_frame(d)
            n[s] = n.get(s, 0) + 1
        rincian = " ".join(f"{LABEL_SET[s][0]}{v}" for s, v in n.items()) if len(n) > 1 else ""
        if "::" in iid:
            s = next(iter(n)) if len(n) == 1 else None
            return (LABEL_SET[s] if len(n) == 1 else ("campur" if n else "–"), f"{len(frames)}", rincian, "")
        s = self.data["rekaman"].get(iid)
        w = self.waktu.get(iid)
        waktu = f"{w.day} {CR.BULAN[w.month - 1]} {w:%H.%M} {CR.periode(w.hour)}" if w else "?"
        return (LABEL_SET[s], f"{len(frames)}/{self.n_ekspor.get(iid, 0)}", rincian, waktu)

    def _isi_tree(self) -> None:
        pilih = set(self.tree.selection())
        buka = {r for r in self.rekaman if self.tree.exists(r) and self.tree.item(r, "open")}
        self.tree.delete(*self.tree.get_children())
        for r in self.rekaman:
            c = self.catatan.get(r) or {}
            s = self.data["rekaman"].get(r)
            tag = [str(s)] + ([f"stabilo_{c['warna']}"] if c.get("warna") else [])
            teks = r.replace("TANGGA_NAIK_", "") + ("  📝" if c.get("catatan") else "")
            self.tree.insert("", "end", iid=r, text=teks, values=self._nilai_baris(r), tags=tag,
                             open=r in buka)
            for i, sc in enumerate(self._scene(r)):
                iid = f"{r}::{i}"
                self.tree.insert(r, "end", iid=iid, text=f"↳ {sc['nama']} ({sc['awal']}–{sc['akhir']})",
                                 values=self._nilai_baris(iid), tags=("scene",))
        ada = [r for r in pilih if self.tree.exists(r)]
        if ada:
            self.tree.selection_set(ada)

    def _perbarui_tree(self, _rekaman=None) -> None:
        for iid in [r for r in self.rekaman] + [f"{r}::{i}" for r in self.rekaman for i in range(len(self._scene(r)))]:
            if self.tree.exists(iid):
                self.tree.item(iid, values=self._nilai_baris(iid))
                if "::" not in iid:
                    s = self.data["rekaman"].get(iid)
                    c = self.catatan.get(iid) or {}
                    self.tree.item(iid, tags=[str(s)] + ([f"stabilo_{c['warna']}"] if c.get("warna") else []))

    def _tampil_catatan(self) -> None:
        sel = self.tree.selection()
        if len(sel) != 1:
            self.info_rek.config(text="", bg=BG); return
        r = sel[0].split("::")[0]
        c = self.catatan.get(r) or {}
        teks = ("📍 " + c["catatan"]) if c.get("catatan") else "Belum ada catatan (isi di tab 2. Tinjau)."
        if self._scene(r):
            teks += "\nScene: " + "; ".join(f"{sc['nama']} ({sc['awal']}–{sc['akhir']})" for sc in self._scene(r))
        self.info_rek.config(text=teks, bg=CR.WARNA.get(c.get("warna")) or BG)

    def _isi_daftar(self, pertahankan: bool = False) -> None:
        kini = self.kini
        filt = self.filter_set.get()
        self.daftar = []
        dilihat = set()
        for iid in self.tree.selection():
            for d in self._frames_iid(iid):
                if d in dilihat:
                    continue
                dilihat.add(d)
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
        self._perbarui_kalkulator(rk, belum)

    # ------------------------------------------------------------ kalkulator & bagi otomatis
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
            self._perbarui_ringkas()

    def _grup(self) -> list[tuple[str, list[Path]]]:
        """Satuan pembagian: scene bila rekaman punya scene (sisa frame menjadi grup
        sendiri), selain itu satu rekaman penuh. Rekaman berset 'abaikan' tidak ikut."""
        grup = []
        for r in self.rekaman:
            if self.data["rekaman"].get(r) == "abaikan" or not self.frames.get(r):
                continue
            sisa = list(self.frames[r])
            for i, _sc in enumerate(self._scene(r)):
                isi = self._frames_iid(f"{r}::{i}")
                if isi:
                    grup.append((f"{r}::{i}", isi))
                    sisa = [d for d in sisa if d not in isi]
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

    def _rencana_blok(self) -> dict[Path, str]:
        """Tiap rekaman dipotong berurutan waktu: awal train, tengah val, akhir test."""
        rasio = self._rasio()
        hasil: dict[Path, str] = {}
        for r in self.rekaman:
            if self.data["rekaman"].get(r) == "abaikan":
                continue
            fr = sorted(self.frames.get(r, []), key=lambda d: CR.indeks_frame(d.name) or 0)
            n = len(fr)
            a = round(n * rasio["train"]); b = a + round(n * rasio["val"])
            for i, d in enumerate(fr):
                hasil[d] = "train" if i < a else "val" if i < b else "test"
        return hasil

    def _perbarui_kalkulator(self, rk: dict, belum: int) -> None:
        for w in self.tabel.winfo_children():
            w.destroy()
        rasio = self._rasio()
        n = sum(rk["frame"].values()) + belum
        kolom = ("Set", "Target %", "Target frame", "Sekarang", "Sekarang %", "Rekaman")
        for j, t in enumerate(kolom):
            tk.Label(self.tabel, text=t, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).grid(row=0, column=j, sticky="e" if j else "w", padx=4)
        for i, k in enumerate(self.sd.SET, 1):
            sek = rk["frame"][k]
            nilai = (LABEL_SET[k], f"{rasio[k] * 100:.0f}%", f"{round(rasio[k] * n)}", f"{sek}",
                     f"{(sek / n * 100 if n else 0):.0f}%", f"{rk['rekaman'][k]}")
            for j, t in enumerate(nilai):
                tk.Label(self.tabel, text=t, bg=PANEL, fg=WARNA_SET[k] if j == 0 else INK,
                         font=("Segoe UI", 9, "bold" if j == 0 else "normal")).grid(row=i, column=j, sticky="e" if j else "w", padx=4)
        tk.Label(self.tabel, text=f"Total siap latih: {n} frame   •   belum dibagi: {belum}", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).grid(row=4, column=0, columnspan=6, sticky="w", padx=4, pady=(2, 0))
        g = len(self._grup())
        saran = [f"Dari {n} frame dengan rasio ini: train {round(rasio['train'] * n)}, "
                 f"val {round(rasio['val'] * n)}, test {round(rasio['test'] * n)}."]
        if n < 300:
            saran.append("Data masih sedikit (< 300): 80/10/10 memberi train lebih banyak, "
                         "tetapi val/test jadi kecil dan angkanya kurang stabil.")
        if g >= 5:
            saran.append(f"Ada {g} rekaman/scene: disarankan bagi PER REKAMAN/SCENE (tombol hijau), "
                         "idealnya lokasi tangga val/test berbeda dari train.")
        else:
            saran.append(f"Hanya {g} rekaman/scene: pembagian per rekaman sulit tepat rasio. Tandai scene di tab Tinjau "
                         "untuk memecah rekaman, atau bagi per blok frame (hati-hati: frame berdekatan mirip).")
        self.rekomendasi.config(text="\n".join(saran))

    def bagi_otomatis(self, mode: str) -> None:
        if not self.rekaman:
            return
        rasio = self._rasio()
        if mode == "grup":
            rencana = self._rencana_grup()
            hitung = {k: 0 for k in self.sd.SET}
            for gid, fr in self._grup():
                hitung[rencana[gid]] += len(fr)
            ket = "per rekaman/scene"
        else:
            rencana_f = self._rencana_blok()
            hitung = {k: sum(1 for v in rencana_f.values() if v == k) for k in self.sd.SET}
            ket = "per blok frame (awal train, tengah val, akhir test tiap rekaman)"
        n = sum(hitung.values())
        if not n:
            messagebox.showinfo("Bagi otomatis", "Tidak ada frame siap latih."); return
        ringkas = "\n".join(f"  {LABEL_SET[k]}: {hitung[k]} frame ({hitung[k] / n * 100:.0f}%, target {rasio[k] * 100:.0f}%)"
                           for k in self.sd.SET)
        if not messagebox.askyesno("Bagi otomatis", f"Bagi {n} frame {ket}:\n\n{ringkas}\n\n"
                                   "Pembagian sekarang (kecuali rekaman 'abaikan') akan diganti. Lanjut?"):
            return
        if mode == "grup":
            for gid, fr in self._grup():
                r = gid.split("::")[0]
                for d in self.frames.get(r, []):
                    self.data["frame"].pop(self.sd.kunci_frame(d), None)
            per_rek: dict[str, dict[str, int]] = {}
            for gid, fr in self._grup():
                per_rek.setdefault(gid.split("::")[0], {}).setdefault(rencana[gid], 0)
                per_rek[gid.split("::")[0]][rencana[gid]] += len(fr)
            for r, n_set in per_rek.items():
                self.data["rekaman"][r] = max(n_set, key=n_set.get)       # set mayoritas jadi set rekaman
            for gid, fr in self._grup():
                r = gid.split("::")[0]
                if rencana[gid] != self.data["rekaman"][r]:
                    for d in fr:
                        self.data["frame"][self.sd.kunci_frame(d)] = rencana[gid]
        else:
            for d, k in rencana_f.items():
                r = d.parent.parent.parent.name
                self.data["rekaman"].setdefault(r, "train")
                if self.data["rekaman"][r] == k:
                    self.data["frame"].pop(self.sd.kunci_frame(d), None)
                else:
                    self.data["frame"][self.sd.kunci_frame(d)] = k
        self._simpan()
        self._isi_tree()
        self._isi_daftar(pertahankan=True)
        self._perbarui_ringkas()
        self._gambar()

    # ------------------------------------------------------------ aksi
    def atur_rekaman(self, s: str | None) -> None:
        """Set untuk rekaman terpilih; untuk baris scene, set dicatat per frame di rentangnya."""
        pilih = list(self.tree.selection())
        if not pilih:
            messagebox.showinfo("Pilih rekaman", "Pilih satu atau beberapa rekaman/scene dahulu."); return
        for iid in pilih:
            if "::" in iid:
                r = iid.split("::")[0]
                for d in self._frames_iid(iid):
                    k = self.sd.kunci_frame(d)
                    if s is None or s == self.data["rekaman"].get(r):
                        self.data["frame"].pop(k, None)
                    else:
                        self.data["frame"][k] = s
            elif s is None:
                self.data["rekaman"].pop(iid, None)
            else:
                self.data["rekaman"][iid] = s
        self._simpan()
        self._perbarui_tree()
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
        self._perbarui_tree()
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

    def ringkasan_csv(self) -> None:
        """Tabel per rekaman untuk naskah: waktu, lokasi, lux, kecerahan gambar, jumlah frame per set."""
        if self._sibuk or not self.rekaman:
            return
        self._sibuk = True
        rekaman, frames, n_ekspor, catatan = list(self.rekaman), dict(self.frames), dict(self.n_ekspor), dict(self.catatan)
        tujuan = Path(self.studio.root_data) / "ringkasan_rekaman.csv"

        def kerja():
            import csv
            try:
                baris, waktu_semua, periode_hitung = [], [], {}
                for no, r in enumerate(rekaman, 1):
                    self._q.put(("status", f"Menyusun ringkasan: {no}/{len(rekaman)} rekaman"))
                    sesi = self.akar / r
                    w = CR.waktu_rekaman(sesi)
                    mulai, selesai = w if w else (None, None)
                    fr = frames.get(r, [])
                    semua_ekspor = sorted(d for d in (sesi / "exports" / "frames").glob("frame_*") if d.is_dir()) \
                        if (sesi / "exports" / "frames").exists() else []
                    contoh = semua_ekspor[::max(1, len(semua_ekspor) // 20)][:20]
                    terang = []
                    for d in contoh:
                        g = cv2.imread(str(d / "color_raw.png"))
                        if g is not None:
                            terang.append(CR.kecerahan(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))[0])
                    per = {k: sum(1 for d in fr if self._set_frame(d) == k) for k in ("train", "val", "test")}
                    c = catatan.get(r) or {}
                    if mulai:
                        waktu_semua.append(mulai)
                        periode_hitung[CR.periode(mulai.hour)] = periode_hitung.get(CR.periode(mulai.hour), 0) + 1
                    baris.append({
                        "rekaman": r, "tanggal": f"{mulai:%Y-%m-%d}" if mulai else "",
                        "jam_mulai": f"{mulai:%H:%M:%S}" if mulai else "", "jam_selesai": f"{selesai:%H:%M:%S}" if selesai else "",
                        "durasi_detik": int((selesai - mulai).total_seconds()) if mulai and selesai else "",
                        "periode": CR.periode(mulai.hour) if mulai else "", "lokasi_catatan": c.get("catatan", ""),
                        "lux_terukur": c.get("lux", ""), "sumber_lux": c.get("lux_sumber", ""),
                        "stabilo": c.get("warna") or "",
                        "kecerahan_gambar_median_0_255": round(float(np.median(terang)), 1) if terang else "",
                        "frame_ekspor": n_ekspor.get(r, 0), "frame_siap_latih": len(fr),
                        "train": per["train"], "val": per["val"], "test": per["test"],
                        "set_rekaman": self.data["rekaman"].get(r, ""), "scene": len(c.get("scene", [])),
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
