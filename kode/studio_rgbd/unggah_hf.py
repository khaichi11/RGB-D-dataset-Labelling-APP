"""Cadangkan dataset berlabel ke Hugging Face sebagai Parquet.

Satu baris per frame berlabel: RGB, depth, mask semantik (persis seperti yang
dilatih), set train/val/test dari split_dataset.json, intrinsik, dan poligon
asli. Kolom gambar memakai bentuk ``{"bytes", "path"}`` yang dikenali
penampil dataset Hugging Face; PNG asli disalin apa adanya (tanpa kompresi
ulang), jadi cadangan ini lossless.

Token dan tujuan dibaca dari ``.env`` di akar proyek (``paket_ubuntu_zenexo/.env``,
diabaikan Git) atau dari variabel lingkungan::

    HF_TOKEN=hf_xxx            # token tulis dari huggingface.co/settings/tokens
    HF_DATASET_REPO=khaichi11/Stairs-Skripsi
    HF_PRIVATE=true            # repo baru dibuat privat; repo lama tidak diubah

Setiap push menulis ulang seluruh berkas Parquet dan menghapus shard lama,
sehingga isi repo selalu sama dengan keadaan dataset lokal saat tombol ditekan.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    from .segmentasi_convnext_depth import akar_aplikasi, akar_proyek
except ImportError:
    from segmentasi_convnext_depth import akar_aplikasi, akar_proyek

NAMA_SPLIT_HF = {'train': 'train', 'val': 'validation', 'test': 'test'}   # selain itu: unassigned
UKURAN_SHARD = 250 * 1024 * 1024


def berkas_env() -> Path:
    return akar_proyek() / '.env'


def baca_env(path: Path | None = None) -> dict[str, str]:
    """Pengurai .env sederhana: KUNCI=nilai per baris, # komentar, kutip opsional."""
    p = path or berkas_env()
    hasil: dict[str, str] = {}
    if p.exists():
        for baris in p.read_text(encoding='utf-8').splitlines():
            baris = baris.strip()
            if not baris or baris.startswith('#') or '=' not in baris:
                continue
            k, v = baris.split('=', 1)
            hasil[k.strip()] = v.strip().strip('"').strip("'")
    return hasil


URL_TOKEN_BARU = 'https://huggingface.co/settings/tokens/new?tokenType=write'


def token_tersimpan() -> str:
    """Token dari penyimpanan kredensial Hugging Face (``hf auth login`` / :func:`simpan_token`)."""
    try:
        from huggingface_hub import get_token
        return get_token() or ''
    except Exception:                                   # noqa: BLE001
        return ''


def konfigurasi() -> dict:
    """Token, repo, dan privasi. Urutan token: variabel lingkungan, .env, lalu login HF tersimpan."""
    env = baca_env()
    ambil = lambda k, bawaan='': os.environ.get(k) or env.get(k) or bawaan
    return {'token': ambil('HF_TOKEN') or token_tersimpan(),
            'repo': ambil('HF_DATASET_REPO', 'khaichi11/Stairs-Skripsi'),
            'repo_model': ambil('HF_MODEL_REPO', 'khaichi11/Skripsi'),
            'privat': ambil('HF_PRIVATE', 'true').lower() in ('1', 'true', 'ya', 'yes')}


def simpan_token(token: str) -> str:
    """Periksa token ke Hugging Face lalu simpan di penyimpanan kredensial HF.

    Disimpan di ``~/.cache/huggingface/token`` (di luar folder proyek), bukan di
    repo, sehingga tidak mungkin ikut ter-commit. Kembalikan nama akun.
    """
    from huggingface_hub import HfApi, login
    token = token.strip()
    info = HfApi(token=token).whoami()
    izin = (info.get('auth') or {}).get('accessToken', {}).get('role')
    if izin and izin not in ('write', 'fineGrained'):
        raise RuntimeError(f'Token ini hanya "{izin}". Buat token jenis Write.')
    login(token=token, add_to_git_credential=False)
    return info.get('name', '?')


def _pustaka():
    akar = str(akar_aplikasi())
    if akar not in sys.path:
        sys.path.insert(0, akar)
    from rgbd_convnext.data import split_dataset
    from rgbd_convnext.data.dataset_d435 import frame_bersih
    return split_dataset, frame_bersih


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', arr)
    if not ok:
        raise RuntimeError('gagal mengodekan PNG')
    return buf.tobytes()


def _baris(d: Path, set_: str | None, diperiksa: bool) -> dict:
    rel = f'{d.parent.parent.parent.name}/{d.name}'
    riser = cv2.imread(str(d / 'mask_objek.png'), 0) > 0
    tread = cv2.imread(str(d / 'mask_acuan.png'), 0) > 0
    sem = np.zeros(riser.shape, np.uint8)
    sem[tread] = 2
    sem[riser] = 1                              # sama dengan FrameD435: riser menang bila tumpang tindih
    f_depth = d / 'depth_aligned_to_color.png'
    depth_png = f_depth.read_bytes() if f_depth.exists() else _png(np.load(d / 'depth_aligned_to_color.npy'))
    info = json.loads((d / 'frame.json').read_text()) if (d / 'frame.json').exists() else {}
    draft = json.loads((d / 'label_draft.json').read_text()) if (d / 'label_draft.json').exists() else {}
    return {
        'id_rekaman': d.parent.parent.parent.name, 'frame': d.name, 'set': set_ or 'belum',
        'diperiksa': diperiksa,
        'rgb': {'bytes': (d / 'color_raw.png').read_bytes(), 'path': f'{rel}/color_raw.png'},
        'depth': {'bytes': depth_png, 'path': f'{rel}/depth_aligned_to_color.png'},
        'semantic': {'bytes': _png(sem), 'path': f'{rel}/semantic.png'},
        'depth_scale': float(info.get('depth_scale', 0.001)),
        'intrinsik': json.dumps(info.get('intrinsics_rgb_native', {})),
        'poligon': json.dumps(draft.get('poligon', {})),
    }


def _skema():
    import pyarrow as pa
    gambar = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
    return pa.schema([('id_rekaman', pa.string()), ('frame', pa.string()), ('set', pa.string()),
                      ('diperiksa', pa.bool_()), ('rgb', gambar), ('depth', gambar), ('semantic', gambar),
                      ('depth_scale', pa.float64()), ('intrinsik', pa.string()), ('poligon', pa.string())])


def _kartu(repo: str, jumlah: dict[str, int], rekaman: int) -> str:
    split_ada = [s for s in ('train', 'validation', 'test', 'unassigned') if jumlah.get(s)]
    konfig = '\n'.join(f'  - split: {s}\n    path: data/{s}/*.parquet' for s in split_ada)
    fitur = '\n'.join(f'  - name: {n}\n    dtype: {t}' for n, t in (
        ('id_rekaman', 'string'), ('frame', 'string'), ('set', 'string'), ('diperiksa', 'bool'),
        ('rgb', 'image'), ('depth', 'image'), ('semantic', 'image'), ('depth_scale', 'float64'),
        ('intrinsik', 'string'), ('poligon', 'string')))
    tabel = '\n'.join(f'| {s} | {jumlah[s]} |' for s in split_ada)
    return f"""---
task_categories:
- image-segmentation
tags:
- rgb-d
- stairs
- realsense-d435
configs:
- config_name: default
  data_files:
{konfig}
dataset_info:
  features:
{fitur}
---

# {repo.split('/')[-1]}

Frame RGB-D tangga naik (Intel RealSense D435, 848 x 480) dengan mask semantik
tiga kelas: 0 latar, 1 riser, 2 tread. Diunggah dari Studio pada
{datetime.now().isoformat(timespec='minutes')}; {rekaman} rekaman.

| Split | Frame |
|---|---|
{tabel}

Kolom:
- `rgb`: PNG warna asli (BGR tersimpan sebagai PNG standar RGB).
- `depth`: PNG 16-bit selaras warna; meter = nilai x `depth_scale`.
- `semantic`: PNG 8-bit 0/1/2, dibentuk persis seperti saat pelatihan.
- `set`: set asli di split_dataset.json (`train`/`val`/`test`/`abaikan`/`belum`).
- `diperiksa`: label sudah dikoreksi manusia. Pelatihan dengan
  `--hanya-diperiksa` hanya memakai baris bernilai `true`.
- `intrinsik`, `poligon`: JSON intrinsik kamera dan poligon label asli.

Split `validation` dan `test` dibagi per rekaman bila memungkinkan; frame
berdekatan dalam satu video hampir identik.
"""


def bangun(akar: Path, berkas_split: Path, keluar: Path,
           lapor: Callable[[str], None] = print) -> dict[str, int]:
    """Tulis Parquet + README.md ke folder ``keluar``. Kembalikan jumlah frame per split HF."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    split_dataset, frame_bersih = _pustaka()
    data = split_dataset.baca(berkas_split)
    semua = frame_bersih(Path(akar), [p.name for p in sorted(Path(akar).iterdir()) if p.is_dir()])
    diperiksa = set(frame_bersih(Path(akar), [p.name for p in sorted(Path(akar).iterdir()) if p.is_dir()],
                                 hanya_diperiksa=True))
    kelompok: dict[str, list[Path]] = {}
    for d in semua:
        s = split_dataset.set_frame(data, d)
        kelompok.setdefault(NAMA_SPLIT_HF.get(s, 'unassigned'), []).append(d)
    skema = _skema()
    jumlah: dict[str, int] = {}
    total, selesai = len(semua), 0
    for nama, frames in kelompok.items():
        folder = keluar / 'data' / nama
        folder.mkdir(parents=True, exist_ok=True)
        buf, ukuran, bagian = [], 0, 0

        def tulis_shard():
            nonlocal buf, ukuran, bagian
            if buf:
                pq.write_table(pa.Table.from_pylist(buf, schema=skema), folder / f'bagian-{bagian:05d}.parquet')
                buf, ukuran, bagian = [], 0, bagian + 1

        for d in frames:
            b = _baris(d, split_dataset.set_frame(data, d), d in diperiksa)
            buf.append(b)
            ukuran += len(b['rgb']['bytes']) + len(b['depth']['bytes'])
            if ukuran >= UKURAN_SHARD:
                tulis_shard()
            selesai += 1
            if selesai % 50 == 0 or selesai == total:
                lapor(f'Menyusun Parquet: {selesai}/{total} frame')
        tulis_shard()
        jumlah[nama] = len(frames)
    rekaman = len({d.parent.parent.parent.name for d in semua})
    (keluar / 'README.md').write_text(_kartu(konfigurasi()['repo'], jumlah, rekaman), encoding='utf-8')
    return jumlah


def push(akar: Path, berkas_split: Path, lapor: Callable[[str], None] = print) -> str:
    """Susun Parquet lalu unggah ke repo dataset Hugging Face. Kembalikan URL repo."""
    from huggingface_hub import HfApi
    k = konfigurasi()
    if not k['token']:
        raise RuntimeError('Belum masuk Hugging Face. Tekan tombol Push untuk masuk.')
    tmp = Path(tempfile.mkdtemp(prefix='hf_dataset_'))
    try:
        jumlah = bangun(akar, berkas_split, tmp, lapor)
        api = HfApi(token=k['token'])
        api.create_repo(k['repo'], repo_type='dataset', private=k['privat'], exist_ok=True)
        lapor(f'Mengunggah ke {k["repo"]} ({sum(jumlah.values())} frame)…')
        api.upload_folder(folder_path=str(tmp), repo_id=k['repo'], repo_type='dataset',
                          delete_patterns=['data/**'],
                          commit_message=f'Perbarui dataset dari Studio: '
                                         + ', '.join(f'{s} {n}' for s, n in jumlah.items()))
        return f'https://huggingface.co/datasets/{k["repo"]}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
