"""Gambar ikon Studio Dataset RGB-D dari nol (bentuk geometris sendiri, tanpa aset pihak lain).

Motif: tangga naik dengan riser merah dan tread biru (warna mask di aplikasi),
titik sudut poligon label, dan latar titik-titik seperti pola proyektor IR yang
diwarnai gradasi jarak.

    kode/.venv/bin/python desktop/buat_ikon.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4                      # digambar 4x lalu diperkecil: tepi halus
N = 512 * S
AKAR = Path(__file__).resolve().parent.parent


def _p(*xy):
    return [(x * S, y * S) for x, y in zip(xy[::2], xy[1::2])]


def _gradasi(warna_atas, warna_bawah, tinggi):
    t = np.linspace(0, 1, tinggi)[:, None]
    a, b = np.array(warna_atas, float), np.array(warna_bawah, float)
    return (a * (1 - t) + b * t).astype(np.uint8)


def _turbo(t: float) -> tuple[int, int, int]:
    """Pendekatan polinomial peta warna turbo (rumus terbuka, Google 2019)."""
    t = min(max(t, 0.0), 1.0)
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
    return tuple(int(255 * min(max(c, 0.0), 1.0)) for c in (r, g, b))


def gambar() -> Image.Image:
    kanvas = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    # Latar: persegi membulat, gradasi biru gelap.
    latar = Image.fromarray(np.repeat(_gradasi((52, 66, 94), (22, 29, 45), N)[:, None, :], N, 1), "RGB").convert("RGBA")
    topeng = Image.new("L", (N, N), 0)
    ImageDraw.Draw(topeng).rounded_rectangle([24 * S, 24 * S, 488 * S, 488 * S], radius=108 * S, fill=255)
    kanvas.paste(latar, (0, 0), topeng)

    # Titik "proyektor IR", warnanya gradasi jarak (jauh di atas, dekat di bawah).
    titik = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    dt = ImageDraw.Draw(titik)
    acak = np.random.default_rng(7)
    for y in range(44, 470, 19):
        for x in range(44, 470, 19):
            jx, jy = acak.uniform(-5, 5, 2)
            r = 2.3 if acak.random() > 0.25 else 1.5
            warna = _turbo(0.15 + 0.75 * (y - 44) / 426)
            dt.ellipse([(x + jx - r) * S, (y + jy - r) * S, (x + jx + r) * S, (y + jy + r) * S], fill=warna + (95,))
    kanvas.alpha_composite(Image.composite(titik, Image.new("RGBA", (N, N)), topeng))

    # Tangga: 4 anak tangga, makin ke atas makin kecil (perspektif).
    d = ImageDraw.Draw(kanvas)
    cx, yb, w, h, t = 256.0, 404.0, 300.0, 66.0, 36.0
    muka = []
    for i in range(4):
        k = 0.8 ** i
        wi, hi, ti = w * k, h * k, t * k
        atas = yb - hi
        wa = wi * 0.965
        riser = (cx - wi / 2, yb, cx + wi / 2, yb, cx + wa / 2, atas, cx - wa / 2, atas)
        wb = w * 0.8 ** (i + 1)
        belakang = atas - ti
        tread = (cx - wa / 2, atas, cx + wa / 2, atas, cx + wb / 2, belakang, cx - wb / 2, belakang)
        muka.append((riser, tread))
        yb = belakang
    bayang = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    db = ImageDraw.Draw(bayang)
    for riser, tread in muka:
        db.polygon(_p(*[v + (8 if j % 2 else 0) for j, v in enumerate(riser)]), fill=(0, 0, 0, 90))
    kanvas.alpha_composite(bayang.filter(ImageFilter.GaussianBlur(6 * S)))
    for riser, tread in muka:
        d.polygon(_p(*tread), fill=(98, 176, 240), outline=(30, 40, 60), width=2 * S)
        d.polygon(_p(*riser), fill=(232, 102, 88), outline=(30, 40, 60), width=2 * S)
        # sorot tipis di bibir anak tangga
        d.line(_p(riser[6], riser[7], riser[4], riser[5]), fill=(255, 214, 200), width=3 * S)

    # Label: garis poligon putih dan titik sudut pada riser terbawah dan tread di atasnya.
    riser0, tread0 = muka[0]
    for poli, tepi in ((riser0, (232, 102, 88)), (tread0, (98, 176, 240))):
        sudut = _p(*poli)
        d.line(sudut + sudut[:1], fill=(255, 255, 255), width=7 * S, joint="curve")
    for poli, tepi in ((riser0, (214, 76, 64)), (tread0, (52, 132, 214))):
        for x, y in zip(poli[::2], poli[1::2]):
            r = 12.5
            d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=(255, 255, 255), outline=tepi, width=5 * S)
    return kanvas


if __name__ == "__main__":
    besar = gambar()
    for ukuran, tujuan in ((512, AKAR / "desktop" / "studio-rgbd.png"),
                           (256, AKAR / "kode" / "studio_rgbd" / "ikon_studio.png")):
        besar.resize((ukuran, ukuran), Image.LANCZOS).save(tujuan, optimize=True)
        print("ditulis", tujuan)
