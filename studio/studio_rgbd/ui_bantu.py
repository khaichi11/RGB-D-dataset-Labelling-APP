"""Pembantu tata letak Tk yang dipakai beberapa tab Studio."""
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk


def kolom_gulir(induk, lebar: int, bg: str, **pack) -> tk.Frame:
    """Kolom selebar ``lebar`` yang bisa di-scroll; kembalikan bingkai isinya.

    Panel kontrol berukuran tetap kehabisan tempat pada layar laptop pendek, dan
    Tk menyembunyikan atau menjepit kontrol yang tidak kebagian tempat. Roda
    tetikus hanya menggulir kolom ini bila kursor berada di atasnya; daftar dan
    kotak teks di dalamnya tetap menggulir dirinya sendiri.
    """
    luar = tk.Frame(induk, bg=bg, width=lebar)
    luar.pack(**({"side": "left", "fill": "y"} | pack))
    luar.pack_propagate(False)
    kanvas = tk.Canvas(luar, bg=bg, highlightthickness=0)
    bilah = ttk.Scrollbar(luar, orient="vertical", command=kanvas.yview)
    kanvas.configure(yscrollcommand=bilah.set)
    bilah.pack(side="right", fill="y")
    kanvas.pack(side="left", fill="both", expand=True)
    isi = tk.Frame(kanvas, bg=bg)
    jendela = kanvas.create_window((0, 0), window=isi, anchor="nw")
    isi.bind("<Configure>", lambda _e: kanvas.configure(scrollregion=kanvas.bbox("all")))
    kanvas.bind("<Configure>", lambda e: kanvas.itemconfigure(jendela, width=e.width))
    akar = induk.winfo_toplevel()

    def gulir(e, langkah):
        w = akar.winfo_containing(e.x_root, e.y_root)
        if isinstance(w, (tk.Listbox, tk.Text)):
            return None
        while w is not None and w is not kanvas:
            w = getattr(w, "master", None)
        if w is None or kanvas.yview() == (0.0, 1.0):
            return None
        kanvas.yview_scroll(langkah, "units")
        return "break"
    akar.bind_all("<Button-4>", lambda e: gulir(e, -3), add="+")
    akar.bind_all("<Button-5>", lambda e: gulir(e, 3), add="+")
    akar.bind_all("<MouseWheel>", lambda e: gulir(e, -3 if e.delta > 0 else 3), add="+")
    return isi


def jadikan_modal(dlg: tk.Toplevel, induk) -> None:
    """Pusatkan dialog di atas induk, lalu kunci (grab) SETELAH dialog tampil.

    Di Xwayland, grab_set() pada jendela yang belum tampil gagal ("grab failed: window not
    viewable") dan dialog tertinggal kosong. Panggil sesudah semua isi dialog dibuat.
    """
    dlg.update_idletasks()
    w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
    try:
        x = induk.winfo_rootx() + (induk.winfo_width() - w) // 2
        y = induk.winfo_rooty() + (induk.winfo_height() - h) // 3
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
    except tk.TclError:
        pass
    try:
        dlg.wait_visibility()
        dlg.grab_set()
    except tk.TclError:
        pass
    dlg.lift()
    dlg.focus_force()


def tombol_menu(studio, induk, teks: str, item: list, width: int = 110, warna: str = "#E8DDD5",
                fg: str = "#382D29"):
    """Satu tombol yang membuka menu: tindakan yang jarang dipakai tetap ada tanpa memenuhi panel.

    ``item``: daftar (atau fungsi yang mengembalikan daftar, dibangun ulang tiap dibuka) berisi ``(teks, perintah)``,
    ``(teks, BooleanVar, perintah)`` untuk centang, atau ``None`` untuk garis.
    """
    menu = tk.Menu(induk, tearoff=False)

    def isi():
        menu.delete(0, "end")
        for it in (item() if callable(item) else item):
            if it is None:
                menu.add_separator()
            elif len(it) == 3:
                menu.add_checkbutton(label=it[0], variable=it[1], command=it[2])
            else:
                menu.add_command(label=it[0], command=it[1])
    isi()

    def tutup(_e=None):
        if menu.winfo_ismapped():
            menu.unpost()

    def tutup_klik(e):
        if e.widget is not tombol:                      # tombol menjalankan perintahnya saat ditekan (Button-1)
            tutup()

    def buka():
        if menu.winfo_ismapped():
            tutup()                                     # klik tombolnya lagi = tutup menu
            return
        if callable(item):
            isi()                                       # isi berubah (mis. daftar tangga): bangun ulang saat dibuka
        x, y = tombol.winfo_rootx(), tombol.winfo_rooty() + tombol.winfo_height()
        if sys.platform == "win32":
            try:
                menu.tk_popup(x, y)                     # Windows: modal, kembali sesudah menu ditutup
            finally:
                menu.grab_release()
        else:
            # Linux (X11/Xwayland): tk_popup tidak modal. grab_release() langsung sesudahnya (pola Windows) melepas
            # grab selagi menu masih terbuka, sehingga klik di luar tidak menutupnya dan menu "macet" di layar.
            menu.tk_popup(x, y)
    # Cadangan bila grab gagal (Xwayland): klik di mana pun di jendela, Esc, atau jendela berpindah menutup menu.
    akar = induk.winfo_toplevel()
    akar.bind("<Button-1>", tutup_klik, add="+")
    akar.bind("<Unmap>", tutup, add="+")
    menu.bind("<Escape>", tutup)
    tombol = studio.tombol_ringkas(induk, teks, buka, warna, fg, width=width)
    tombol.menu = menu
    return tombol


class Kabar:
    """Pemberitahuan di dalam jendela, menggantikan popup info/peringatan/galat yang harus diklik OK.

    Muncul di atas tengah jendela; info hilang sendiri, galat tetap sampai ditutup (✕). Bisa membawa satu
    tombol aksi, mis. "↶ Batalkan" sesudah membuang banyak frame.
    """
    # Pasangan warna yang juga dipetakan tema gelap (tema.PETA_LATAR/PETA_TINTA), agar tulisan tetap terbaca di kedua tema.
    WARNA = {"info": ("#1E5B2A", "#DCEFD8", "✔"), "peringatan": ("#8A5A12", "#FBEEDA", "⚠"),
             "galat": ("#B42318", "#FDE7E5", "✖")}
    DETIK = {"info": 5, "peringatan": 9, "galat": 0}

    def __init__(self, akar) -> None:
        self.akar = akar
        self.bingkai = tk.Frame(akar, bd=0, highlightthickness=1, padx=10, pady=6)
        self.ikon = tk.Label(self.bingkai, font=("Segoe UI", 11, "bold"))
        self.teks = tk.Label(self.bingkai, justify="left", anchor="w", font=("Segoe UI", 10))
        self.aksi = tk.Label(self.bingkai, font=("Segoe UI", 10, "bold"), cursor="hand2", padx=8)
        self.tutup = tk.Label(self.bingkai, text="✕", cursor="hand2", font=("Segoe UI", 10), padx=4)
        self.ikon.pack(side="left", padx=(0, 6))
        self.teks.pack(side="left")
        self.tutup.pack(side="right")
        self.tutup.bind("<Button-1>", lambda _e: self.sembunyi())
        self.teks.bind("<Button-1>", lambda _e: self.sembunyi())
        self._jadwal = None
        self._fungsi_aksi = None
        self.aksi.bind("<Button-1>", lambda _e: self._jalankan_aksi())

    def tampil(self, teks: str, jenis: str = "info", detik: float | None = None, aksi=None) -> None:
        fg, bg, ikon = self.WARNA.get(jenis, self.WARNA["info"])
        teks = " ".join(str(teks).split()) if len(str(teks)) < 160 else str(teks).strip()
        if len(teks) > 700:
            teks = teks[:700] + "…"
        lebar = max(320, int(self.akar.winfo_width() * 0.6))
        for w in (self.bingkai, self.ikon, self.teks, self.tutup, self.aksi):
            w.configure(bg=bg)
        self.bingkai.configure(highlightbackground=fg)
        self.ikon.configure(text=ikon, fg=fg)
        self.teks.configure(text=teks, fg=fg, wraplength=lebar)
        self.tutup.configure(fg=fg)
        self._fungsi_aksi = aksi[1] if aksi else None
        if aksi:
            self.aksi.configure(text=aksi[0], fg="#FFFFFF", bg=fg)
            self.aksi.pack(side="right", padx=(8, 4), before=self.tutup)
        else:
            self.aksi.pack_forget()
        self.bingkai.place(relx=0.5, y=6, anchor="n")
        self.bingkai.lift()
        if self._jadwal is not None:
            self.akar.after_cancel(self._jadwal)
            self._jadwal = None
        detik = self.DETIK.get(jenis, 5) if detik is None else detik
        if aksi and detik:
            detik = max(detik, 10)                      # beri waktu untuk menekan Batalkan
        if detik:
            self._jadwal = self.akar.after(int(detik * 1000), self.sembunyi)

    def _jalankan_aksi(self) -> None:
        f, self._fungsi_aksi = self._fungsi_aksi, None
        self.sembunyi()
        if f is not None:
            f()

    def sembunyi(self) -> None:
        if self._jadwal is not None:
            try:
                self.akar.after_cancel(self._jadwal)
            except tk.TclError:
                pass
            self._jadwal = None
        self.bingkai.place_forget()


def pasang_kabar(akar) -> Kabar:
    """Alihkan messagebox.showinfo/showwarning/showerror milik jendela utama ke :class:`Kabar`.

    Dialog yang dibuka dari jendela lain (``parent`` = Toplevel lain) tetap berupa popup di jendela itu.
    Pertanyaan ya/tidak (askyesno) tidak dialihkan: dipakai hanya untuk tindakan yang tidak bisa dibatalkan.
    """
    from tkinter import messagebox
    kabar = Kabar(akar)
    asli = {n: getattr(messagebox, n) for n in ("showinfo", "showwarning", "showerror")}

    def buat(nama: str, jenis: str):
        def f(title=None, message=None, **opsi):
            induk = opsi.get("parent")
            try:
                lain = induk is not None and induk.winfo_toplevel() is not akar
            except tk.TclError:
                lain = True
            if lain or not akar.winfo_exists():
                return asli[nama](title, message, **opsi)
            judul = (title or "").strip()
            isi = (message or "").strip()
            kabar.tampil(f"{judul}: {isi}" if judul and isi and len(isi) < 300 else (isi or judul), jenis)
            return "ok"
        return f
    messagebox.showinfo = buat("showinfo", "info")
    messagebox.showwarning = buat("showwarning", "peringatan")
    messagebox.showerror = buat("showerror", "galat")
    return kabar
