"""Dataset Studio di Hugging Face: push (cadangkan) dan tarik (pulihkan untuk melabel ulang).

Satu baris Parquet per frame EKSPOR (berlabel maupun belum): RGB, depth selaras
16-bit, IR kiri, mask semantik (persis seperti yang dilatih), frame.json dan
label_draft.json lengkap, status sampah, dan set train/val/test. Repo juga
memuat ``split_dataset.json``, ``rekaman/<kategori>/<ID>/catatan.json``, dan
``rekaman/<kategori>/<ID>/sinkron_waktu.json`` (hasil Cek sinkron RGB-depth-IR,
agar IR yang tidak sejepretan tetap dikenali tanpa video mentah).
Isinya cukup untuk MELABEL ULANG di laptop lain tanpa video mentah: tombol
"Tarik dataset dari HF" membangun kembali folder ``exports/frames`` Studio.
Video mentah hanya diperlukan untuk mengekspor frame baru, jadi tidak diunggah.

Kolom gambar memakai bentuk ``{"bytes", "path"}`` yang dikenali penampil
dataset Hugging Face; PNG asli disalin apa adanya, jadi lossless.

Token dan tujuan dibaca dari ``.env`` di akar proyek (``paket_ubuntu_zenexo/.env``,
diabaikan Git) atau dari variabel lingkungan::

    HF_TOKEN=hf_xxx            # token tulis dari huggingface.co/settings/tokens
    HF_DATASET_REPO=khaichi11/Stairs-Skripsi
    HF_PRIVATE=true            # repo baru dibuat privat; repo lama tidak diubah

Push menulis ulang seluruh Parquet (shard lama dihapus). Tarik tidak pernah
menimpa label lokal yang lebih baru (``disimpan_iso``). Alur yang aman di
beberapa laptop: Tarik -> melabel -> Push.
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


def _json(f: Path) -> dict:
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except (OSError, ValueError):
        return {}


def _teks(f: Path) -> str:
    return f.read_text(encoding='utf-8') if f.exists() else ''


def _gambar(f: Path, rel: str) -> dict | None:
    return {'bytes': f.read_bytes(), 'path': rel} if f.exists() else None


def _baris(d: Path, kategori: str, set_: str | None) -> dict:
    rel = f'{d.parent.parent.parent.name}/{d.name}'
    draft = _json(d / 'label_draft.json')
    state = _json(d / 'frame_state.json')
    info = _json(d / 'frame.json')
    berlabel = (d / 'mask_objek.png').exists() and (d / 'mask_acuan.png').exists() and bool(draft)
    sem = None
    if berlabel:
        riser = cv2.imread(str(d / 'mask_objek.png'), 0) > 0
        tread = cv2.imread(str(d / 'mask_acuan.png'), 0) > 0
        peta = np.zeros(riser.shape, np.uint8)
        peta[tread] = 2
        peta[riser] = 1                     # sama dengan FrameD435: riser menang bila tumpang tindih
        sem = {'bytes': _png(peta), 'path': f'{rel}/semantic.png'}
    f_depth = d / 'depth_aligned_to_color.png'
    depth = (_gambar(f_depth, f'{rel}/depth_aligned_to_color.png') if f_depth.exists() else
             {'bytes': _png(np.load(d / 'depth_aligned_to_color.npy')), 'path': f'{rel}/depth_aligned_to_color.png'})
    return {
        'kategori': kategori, 'id_rekaman': d.parent.parent.parent.name, 'frame': d.name,
        'set': set_ or 'belum', 'berlabel': berlabel,
        'diperiksa': bool(draft.get('diperiksa_manual', not draft.get('otomatis', False))) if draft else False,
        'di_sampah': bool(state.get('di_sampah', False)),
        'disimpan_iso': str(draft.get('disimpan_iso') or ''),
        'rgb': _gambar(d / 'color_raw.png', f'{rel}/color_raw.png'),
        'depth': depth,
        'ir': _gambar(d / 'ir_left_raw.png', f'{rel}/ir_left_raw.png'),
        'semantic': sem,
        'depth_scale': float(info.get('depth_scale', 0.001)),
        # Teks berkas asli disalin apa adanya supaya hasil Tarik identik byte demi byte.
        'frame_json': _teks(d / 'frame.json'), 'label_draft': _teks(d / 'label_draft.json') if draft else '',
        'frame_state': _teks(d / 'frame_state.json') if state else '',
        'intrinsik': json.dumps(info.get('intrinsics_rgb_native', {})),
        'poligon': json.dumps(draft.get('poligon', {})),
    }


KOLOM = (('kategori', 'string'), ('id_rekaman', 'string'), ('frame', 'string'), ('set', 'string'),
         ('berlabel', 'bool'), ('diperiksa', 'bool'), ('di_sampah', 'bool'), ('disimpan_iso', 'string'),
         ('rgb', 'image'), ('depth', 'image'), ('ir', 'image'), ('semantic', 'image'), ('depth_scale', 'float64'),
         ('frame_json', 'string'), ('label_draft', 'string'), ('frame_state', 'string'),
         ('intrinsik', 'string'), ('poligon', 'string'))


def _skema():
    import pyarrow as pa
    gambar = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
    jenis = {'string': pa.string(), 'bool': pa.bool_(), 'float64': pa.float64(), 'image': gambar}
    return pa.schema([(n, jenis[t]) for n, t in KOLOM])


def _kartu(repo: str, jumlah: dict[str, int], rekaman: int, berlabel: int) -> str:
    split_ada = [s for s in ('train', 'validation', 'test', 'unassigned') if jumlah.get(s)]
    konfig = '\n'.join(f'  - split: {s}\n    path: data/{s}/*.parquet' for s in split_ada)
    fitur = '\n'.join(f'  - name: {n}\n    dtype: {t}' for n, t in KOLOM)
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

Frame RGB-D tangga (Intel RealSense D435, 848 x 480) dengan mask semantik tiga
kelas: 0 latar, 1 riser, 2 tread. Diunggah dari Studio pada
{datetime.now().isoformat(timespec='minutes')}; {rekaman} rekaman, {sum(jumlah.values())} frame
ekspor, {berlabel} berlabel.

| Split | Frame |
|---|---|
{tabel}

`unassigned` berisi frame yang belum dibagi, belum berlabel, atau di sampah
(lihat kolom `berlabel`, `diperiksa`, `di_sampah`). Pelatihan dengan
`--hanya-diperiksa` hanya memakai baris `berlabel` dan `diperiksa` bernilai true.

Kolom gambar: `rgb` (PNG asli), `depth` (PNG 16-bit selaras RGB; meter =
nilai x `depth_scale`), `ir` (IR kiri mentah, belum diselaraskan), `semantic`
(PNG 0/1/2, persis seperti saat pelatihan). `frame_json`, `label_draft`,
`frame_state` adalah berkas asli Studio sehingga dataset dapat dipulihkan dan
dilabel ulang tanpa video mentah (tombol "Tarik dataset dari HF" di Studio).
"""


def _semua_frame(akar_rekaman: Path):
    """(kategori, folder frame) untuk setiap frame ekspor di semua kategori."""
    for kat in sorted(p for p in Path(akar_rekaman).iterdir() if p.is_dir()):
        for sesi in sorted(p for p in kat.iterdir() if p.is_dir()):
            fr = sesi / 'exports' / 'frames'
            if fr.exists():
                for d in sorted(fr.iterdir()):
                    if d.is_dir() and (d / 'color_raw.png').exists() and (d / 'frame.json').exists():
                        yield kat.name, d


def bangun(akar: Path, berkas_split: Path, keluar: Path,
           lapor: Callable[[str], None] = print) -> dict[str, int]:
    """Tulis Parquet, catatan rekaman, split, dan README ke ``keluar``. ``akar``: folder kategori tangga_naik."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    split_dataset, _ = _pustaka()
    data = split_dataset.baca(berkas_split)
    akar_rekaman = Path(akar).parent
    semua = list(_semua_frame(akar_rekaman))
    skema = _skema()
    penulis: dict[str, list] = {}
    jumlah: dict[str, int] = {}
    berlabel = 0
    rekaman = set()

    def tulis(nama: str, buf: list) -> None:
        if not buf:
            return
        folder = keluar / 'data' / nama
        folder.mkdir(parents=True, exist_ok=True)
        i = len(list(folder.glob('*.parquet')))
        pq.write_table(pa.Table.from_pylist(buf, schema=skema), folder / f'bagian-{i:05d}.parquet')
        buf.clear()

    for n, (kat, d) in enumerate(semua, 1):
        s = split_dataset.set_frame(data, d)
        b = _baris(d, kat, s)
        nama = NAMA_SPLIT_HF.get(s, 'unassigned') if b['berlabel'] and not b['di_sampah'] else 'unassigned'
        buf = penulis.setdefault(nama, [])
        buf.append(b)
        jumlah[nama] = jumlah.get(nama, 0) + 1
        berlabel += b['berlabel']
        rekaman.add((kat, b['id_rekaman']))
        if sum(len(x['rgb']['bytes']) + len(x['depth']['bytes']) + len((x['ir'] or {}).get('bytes', b''))
               for x in buf) >= UKURAN_SHARD:
            tulis(nama, buf)
        if n % 50 == 0 or n == len(semua):
            lapor(f'Menyusun Parquet: {n}/{len(semua)} frame')
    for nama, buf in penulis.items():
        tulis(nama, buf)
    for kat, rid in rekaman:                                   # catatan rekaman (warna, lokasi, lux, scene) + sinkron
        for c in (akar_rekaman / kat / rid / 'catatan.json', akar_rekaman / kat / rid / 'exports' / 'sinkron_waktu.json'):
            if c.exists():
                tujuan = keluar / 'rekaman' / kat / rid / c.name
                tujuan.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(c, tujuan)
    if Path(berkas_split).exists():
        shutil.copy(berkas_split, keluar / 'split_dataset.json')
    if (akar_rekaman.parent / 'tangga.json').exists():                # daftar tangga fisik
        shutil.copy(akar_rekaman.parent / 'tangga.json', keluar / 'tangga.json')
    (keluar / 'README.md').write_text(_kartu(konfigurasi()['repo'], jumlah, len(rekaman), berlabel), encoding='utf-8')
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
                          delete_patterns=['data/**', 'rekaman/**', 'tangga.json'],
                          commit_message='Perbarui dataset dari Studio: '
                                         + ', '.join(f'{s} {n}' for s, n in jumlah.items()))
        return f'https://huggingface.co/datasets/{k["repo"]}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def tarik(akar: Path, berkas_split: Path, lapor: Callable[[str], None] = print) -> dict[str, int]:
    """Bangun kembali frame ekspor dari repo dataset HF agar bisa dilabel ulang tanpa video.

    Berkas gambar/depth/IR hanya ditulis bila belum ada. Label (label_draft.json
    dan mask) hanya ditulis bila frame lokal belum berlabel atau label di HF
    lebih baru (``disimpan_iso``); label lokal yang lebih baru tidak disentuh.
    Catatan rekaman lokal dipertahankan; split lokal digabung (entri lokal menang).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    k = konfigurasi()
    if not k['token']:
        raise RuntimeError('Belum masuk Hugging Face.')
    akar_rekaman = Path(akar).parent
    tmp = Path(tempfile.mkdtemp(prefix='hf_tarik_'))
    hitung = {'frame_baru': 0, 'label_diperbarui': 0, 'label_lokal_lebih_baru': 0, 'sudah_sama': 0}
    try:
        lapor(f'Mengunduh dataset {k["repo"]}…')
        snapshot_download(k['repo'], repo_type='dataset', local_dir=str(tmp), token=k['token'],
                          allow_patterns=['data/**', 'rekaman/**', 'split_dataset.json', 'tangga.json'])
        berkas = sorted(tmp.glob('data/**/*.parquet'))
        total = sum(pq.ParquetFile(f).metadata.num_rows for f in berkas)
        n = 0
        for f in berkas:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=16):
                for r in batch.to_pylist():
                    n += 1
                    d = akar_rekaman / r['kategori'] / r['id_rekaman'] / 'exports' / 'frames' / r['frame']
                    d.mkdir(parents=True, exist_ok=True)
                    if not (d / 'color_raw.png').exists():
                        (d / 'color_raw.png').write_bytes(r['rgb']['bytes'])
                        (d / 'depth_aligned_to_color.png').write_bytes(r['depth']['bytes'])
                        np.save(d / 'depth_aligned_to_color.npy',
                                cv2.imdecode(np.frombuffer(r['depth']['bytes'], np.uint8), cv2.IMREAD_UNCHANGED))
                        if r.get('ir'):
                            (d / 'ir_left_raw.png').write_bytes(r['ir']['bytes'])
                        (d / 'frame.json').write_text(r['frame_json'])
                        hitung['frame_baru'] += 1
                    if r.get('frame_state') and not (d / 'frame_state.json').exists():
                        (d / 'frame_state.json').write_text(r['frame_state'])
                    if r.get('label_draft'):
                        lokal = _json(d / 'label_draft.json')
                        waktu_lokal = str(lokal.get('disimpan_iso') or '')
                        if lokal and waktu_lokal > r['disimpan_iso']:
                            hitung['label_lokal_lebih_baru'] += 1
                        elif lokal and waktu_lokal == r['disimpan_iso']:
                            hitung['sudah_sama'] += 1
                        else:
                            (d / 'label_draft.json').write_text(r['label_draft'])
                            if r.get('semantic'):
                                sem = cv2.imdecode(np.frombuffer(r['semantic']['bytes'], np.uint8), cv2.IMREAD_UNCHANGED)
                                cv2.imwrite(str(d / 'mask_objek.png'), ((sem == 1) * 255).astype(np.uint8))
                                cv2.imwrite(str(d / 'mask_acuan.png'), ((sem == 2) * 255).astype(np.uint8))
                            hitung['label_diperbarui'] += 1
                    if n % 50 == 0 or n == total:
                        lapor(f'Memulihkan frame: {n}/{total}')
        for c in tmp.glob('rekaman/*/*/*.json'):
            tujuan = akar_rekaman / c.parent.parent.name / c.parent.name
            tujuan = {'catatan.json': tujuan / 'catatan.json',
                      'sinkron_waktu.json': tujuan / 'exports' / 'sinkron_waktu.json'}.get(c.name)
            if tujuan is not None and not tujuan.exists():
                tujuan.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(c, tujuan)
        if (tmp / 'tangga.json').exists():                         # ID tangga lokal menang, yang baru ditambahkan
            lokal_t = akar_rekaman.parent / 'tangga.json'
            gabung = json.loads((tmp / 'tangga.json').read_text()) | (json.loads(lokal_t.read_text()) if lokal_t.exists() else {})
            lokal_t.write_text(json.dumps(dict(sorted(gabung.items())), indent=1, ensure_ascii=False) + '\n')
        jauh = tmp / 'split_dataset.json'
        if jauh.exists():
            split_dataset, _ = _pustaka()
            lokal, remote = split_dataset.baca(berkas_split), split_dataset.baca(jauh)
            for kunci in ('rekaman', 'frame'):
                lokal[kunci] = remote[kunci] | lokal[kunci]
            split_dataset.tulis(berkas_split, lokal)
        return hitung
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
