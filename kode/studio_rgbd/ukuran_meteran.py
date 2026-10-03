"""Ukuran meteran per anak tangga fisik, dibandingkan dengan ukuran sistem dari pelacakan video.

Ukuran sistem dihitung ``rgbd_convnext.ukuran_sesi`` (Train-RGB-D-Model) dan
disimpan per sesi di ``<sesi>/derived/ukuran_sistem.json``: nomor anak tangga
diambil dari pelacak yang berjalan pada SETIAP frame rekaman, karena pada frame
ekspor saja anak tangga ke-1, 2, 3 sulit dibedakan.

Ukuran meteran milik tangga FISIK, jadi cukup diukur sekali:

- bila frame/sesi dikaitkan ke tangga fisik (``tangga.json``, diatur di tab
  Tinjau), ukurannya disimpan di ``tangga.json`` -> ``<ID>.ukuran_meteran`` dan
  dipakai semua rekaman tangga itu;
- bila belum, disimpan di ``<sesi>/ukuran_meteran.json`` untuk sesi itu saja.

``<sesi>/ukuran_meteran.json`` juga menyimpan ``geser`` per penomoran: bila
pelacak mulai menomori dari anak tangga fisik ke-2 (anak tangga pertama tidak
terlihat), nomor sistem #1 = fisik #2, jadi geser = 1.

    python -m studio_rgbd.ukuran_meteran [akar_data]     # CSV sistem lawan meteran + ringkasan galat
    python -m studio_rgbd.ukuran_meteran --impor ekspor_ukur_tangga.json [akar_data]

``--impor`` membaca berkas cadangan aplikasi iPad "Ukur Tangga" (repo khaichi11/ukur-tangga,
format ``ukur-tangga`` versi 1 atau 2): lokasi (v2) atau tangga (v1) yang "ID tangga di
Studio"-nya diisi (mis. T01) ditulis ke ``tangga.json`` -> ``<ID>.ukuran_meteran``; ID yang
belum ada dibuat. Tangga ke-1, 2, ... yang dipisah bordes menjadi penomoran 1, 2, ...
(nomor mulai dari 1 lagi, sama dengan pelacak).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

try:
    from . import catatan_rekaman as CR
except ImportError:
    import catatan_rekaman as CR

BERKAS_SESI = "ukuran_meteran.json"
KUNCI = "ukuran_meteran"


def _baca_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def _tulis_json(p: Path, data: dict) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)


def _bersihkan(anak: dict) -> dict:
    """{"1": {"tinggi_riser_cm": 17.0, ...}} -> kunci int, nilai float, baris kosong dibuang."""
    hasil = {}
    for k, v in (anak or {}).items():
        try:
            nomor = int(k)
        except (TypeError, ValueError):
            continue
        isi = {}
        for kunci in ("tinggi_riser_cm", "panjang_tread_cm"):
            try:
                if v.get(kunci) not in (None, ""):
                    isi[kunci] = round(float(v[kunci]), 1)
            except (TypeError, ValueError, AttributeError):
                pass
        if isi:
            hasil[nomor] = isi
    return hasil


def _anak_penomoran(isi: dict, ke: int) -> dict:
    """Anak tangga penomoran ke-``ke`` (tangga ke-n sesudah bordes); penomoran 1 juga di ``anak_tangga`` lama."""
    pen = isi.get("penomoran") or {}
    if str(ke) in pen:
        return pen[str(ke)]
    return isi.get("anak_tangga") if ke == 1 else {}


def baca(sesi: Path, akar_data: Path, tid: str | None, ke: int = 1) -> dict:
    """{'tempat': 'tangga'|'sesi', 'anak_tangga': {nomor fisik: {...}}, 'lebar_cm', 'geser': {ke: n}}.

    ``ke``: penomoran ke berapa (nomor anak tangga mulai dari 1 lagi sesudah setiap bordes).
    """
    sesi_d = _baca_json(Path(sesi) / BERKAS_SESI)
    if tid:
        isi = (_baca_json(Path(akar_data) / CR.BERKAS_TANGGA).get(tid) or {}).get(KUNCI) or {}
        tempat = "tangga"
    else:
        isi, tempat = sesi_d, "sesi"
    geser = {}
    for k, v in (sesi_d.get("geser") or {}).items():
        try:
            geser[int(k)] = int(v)
        except (TypeError, ValueError):
            pass
    return {"tempat": tempat, "anak_tangga": _bersihkan(_anak_penomoran(isi, ke)),
            "lebar_cm": isi.get("lebar_cm"), "geser": geser}


def _isi_baru(lama: dict, ke: int, anak_tangga: dict, lebar_cm, waktu: str) -> dict:
    anak = {str(k): v for k, v in sorted(_bersihkan(anak_tangga).items())}
    isi = {k: v for k, v in (lama or {}).items() if k in ("anak_tangga", "penomoran", "bordes", "sumber")}
    isi.setdefault("penomoran", {})[str(ke)] = anak
    if ke == 1:
        isi["anak_tangga"] = anak
    isi.update(lebar_cm=lebar_cm, diperbarui=waktu)
    return isi


def tulis(sesi: Path, akar_data: Path, tid: str | None, anak_tangga: dict, lebar_cm, geser: dict, ke: int = 1) -> str:
    """Simpan ukuran meteran penomoran ``ke`` (ke tangga fisik bila ada) dan geser penomoran sesi."""
    waktu = dt.datetime.now().isoformat(timespec="seconds")
    p_sesi = Path(sesi) / BERKAS_SESI
    sesi_d = _baca_json(p_sesi)
    sesi_d["geser"] = {str(k): int(v) for k, v in geser.items()}
    if tid:
        p = Path(akar_data) / CR.BERKAS_TANGGA
        semua = _baca_json(p)
        entri = semua.setdefault(tid, {})
        entri[KUNCI] = _isi_baru(entri.get(KUNCI), ke, anak_tangga, lebar_cm, waktu)
        _tulis_json(p, semua)
        tempat = f"tangga {tid}"
    else:
        sesi_d.update(_isi_baru(sesi_d, ke, anak_tangga, lebar_cm, waktu))
        tempat = f"sesi {Path(sesi).name}"
    sesi_d["diperbarui"] = waktu
    _tulis_json(p_sesi, sesi_d)
    return tempat


def tangga_segmen(sesi: Path, sistem: dict, catatan: dict) -> dict:
    """Tangga fisik tiap penomoran ``ke``: scene pada frame pertama ia teramati, selain itu tangga rekaman."""
    hasil = {}
    indeks = Path(sesi) / "derived" / "frame_index.csv"
    cap = []
    if indeks.exists():
        with indeks.open() as f:
            for baris in csv.DictReader(f):
                try:
                    cap.append((float(baris["timestamp_ms"]), int(baris["frame"])))
                except (KeyError, ValueError):
                    pass
    awal = {}
    for kunci, tanda in sistem.get("frame", {}).items():
        for ke, *_ in tanda:
            awal[ke] = min(awal.get(ke, float("inf")), float(kunci))
    for ke, t in awal.items():
        idx = min(cap, key=lambda c: abs(c[0] - t))[1] if cap else None
        hasil[ke] = CR.tangga_untuk(catatan, idx) if idx is not None else catatan.get("tangga")
    return hasil


def banding(akar_data: Path) -> list[dict]:
    """Baris per ukuran meteran yang punya pasangan sistem: sesi, tangga, nomor, besaran, sistem, meteran, selisih."""
    akar_data = Path(akar_data)
    baris = []
    for sistem_p in sorted(akar_data.glob("rekaman/*/*/derived/ukuran_sistem.json")):
        sesi = sistem_p.parent.parent
        try:
            sistem = json.loads(sistem_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        catatan = CR.baca(sesi)
        per_ke = tangga_segmen(sesi, sistem, catatan)
        for t in sistem.get("tangga", []):
            tid = per_ke.get(t["ke"]) or catatan.get("tangga")
            met = baca(sesi, akar_data, tid, t["ke"])
            geser = met["geser"].get(t["ke"], 0)
            for a in t["anak_tangga"]:
                fisik = a["nomor"] + geser
                m = met["anak_tangga"].get(fisik, {})
                for besaran in ("tinggi_riser", "panjang_tread"):
                    nilai_m = m.get(f"{besaran}_cm")
                    if nilai_m is None:
                        continue
                    s = a.get(besaran) or {}
                    baris.append({"sesi": sesi.name, "tangga_fisik": tid or "", "penomoran": t["ke"],
                                  "nomor_sistem": a["nomor"], "nomor_fisik": fisik, "besaran": besaran,
                                  "sistem_cm": s.get("cm"), "meteran_cm": nilai_m,
                                  "selisih_cm": round(s["cm"] - nilai_m, 1) if s.get("cm") is not None else None,
                                  "iqr_cm": s.get("iqr_cm"), "n_frame": s.get("n_frame"),
                                  "sumber_tinggi": a.get("sumber_tinggi") if besaran == "tinggi_riser" else ""})
    return baris


def impor_ukur_tangga(berkas: Path, akar_data: Path) -> list[str]:
    """Ukuran dari ekspor aplikasi Ukur Tangga ke tangga.json; mengembalikan ringkasan per tangga fisik.

    Versi 2: satu lokasi = satu tangga fisik (``id_studio`` lokasi); tangga ke-1, 2, ... di dalamnya
    (dipisah bordes) menjadi penomoran 1, 2, ... yang nomornya mulai dari 1 lagi, sama dengan pelacak.
    Versi 1: setiap tangga ber-``id_studio`` menjadi satu tangga fisik (penomoran 1).
    """
    data = json.loads(Path(berkas).read_text(encoding="utf-8"))
    if data.get("format") != "ukur-tangga":
        raise ValueError("bukan berkas ekspor Ukur Tangga")
    p = Path(akar_data) / CR.BERKAS_TANGGA
    semua = _baca_json(p)
    laporan = []

    def isi_anak(daftar, turun=False):
        n, hasil = len(daftar), {}
        for i, b in enumerate(daftar, start=1):
            v = {k: b.get(k) for k in ("tinggi_riser_cm", "panjang_tread_cm") if isinstance(b.get(k), (int, float))}
            if v:
                hasil[str(n + 1 - i if turun else i)] = v
        return hasil

    for lok in data.get("lokasi", []):
        if data.get("versi", 1) >= 2:
            tid = (lok.get("id_studio") or "").strip().upper()
            if not tid:
                continue
            pen = {str(t.get("ke", i + 1)): isi_anak(t.get("anak_tangga", [])) for i, t in enumerate(lok.get("tangga", []))}
            bordes = [t["bordes_sesudah"].get("panjang_cm") for t in lok.get("tangga", []) if t.get("bordes_sesudah")]
            entri = semua.setdefault(tid, {"nama": lok.get("nama", ""), "lokasi": lok.get("nama", "")})
            entri[KUNCI] = {"anak_tangga": pen.get("1", {}), "penomoran": pen, "bordes": bordes,
                            "diperbarui": data.get("diekspor"), "sumber": f"ukur-tangga: {lok.get('nama', '')}"}
            laporan.append(f"{tid}: {sum(len(v) for v in pen.values())} anak tangga dalam {len(pen)} tangga ({lok.get('nama', '')})")
            continue
        for t in lok.get("tangga", []):
            tid = (t.get("id_studio") or "").strip().upper()
            if not tid:
                continue
            anak = isi_anak([b for b in t.get("bagian", []) if b.get("jenis") == "anak_tangga"], t.get("arah") == "turun")
            entri = semua.setdefault(tid, {"nama": f"{lok.get('nama', '')} · {t.get('nama', '')}", "lokasi": lok.get("nama", "")})
            entri[KUNCI] = {"anak_tangga": anak, "penomoran": {"1": anak}, "lebar_cm": t.get("lebar_cm"),
                            "diperbarui": data.get("diekspor"), "sumber": f"ukur-tangga: {lok.get('nama', '')} / {t.get('nama', '')}"}
            laporan.append(f"{tid}: {len(anak)} anak tangga ({lok.get('nama', '')} / {t.get('nama', '')})")
    _tulis_json(p, semua)
    return laporan


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--impor"]:
        akar = Path(argv[2]) if len(argv) > 2 else Path(__file__).resolve().parents[2] / "dataset" / "studio_rgbd"
        for baris in impor_ukur_tangga(Path(argv[1]), akar) or ["tidak ada tangga ber-ID Studio di berkas ini"]:
            print(baris)
        return
    akar = Path(argv[0]) if argv else Path(__file__).resolve().parents[2] / "dataset" / "studio_rgbd"
    baris = banding(akar)
    if not baris:
        print("Belum ada anak tangga yang punya ukuran sistem dan ukuran meteran sekaligus.", file=sys.stderr)
        return
    tulis_csv = csv.DictWriter(sys.stdout, fieldnames=list(baris[0]))
    tulis_csv.writeheader()
    tulis_csv.writerows(baris)
    for besaran in ("tinggi_riser", "panjang_tread"):
        beda = sorted(abs(b["selisih_cm"]) for b in baris if b["besaran"] == besaran and b["selisih_cm"] is not None)
        if beda:
            print(f"# {besaran}: {len(beda)} ukuran, galat mutlak median {statistics.median(beda):.1f} cm, "
                  f"maks {beda[-1]:.1f} cm", file=sys.stderr)


if __name__ == "__main__":
    main()
