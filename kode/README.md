# ZenExo — kode aplikasi

Repository GitHub ini hanya memuat aplikasi Studio untuk pengambilan data dan labeling RGB-D. Kode inferensi, eksperimen, dan alat lama tetap tersedia secara lokal tetapi tidak dipublikasikan di repository ini.

Jalankan aplikasi:

```bash
cd ~/paket_ubuntu_zenexo/kode
source .venv/bin/activate
python -m studio_rgbd.studio_dataset_rgbd --preset jangan
```

Untuk rekaman dengan exposure manual (mengurangi blur saat kamera bergerak):

```bash
python -m studio_rgbd.studio_dataset_rgbd --preset akurasi --exposure 6000
```

`--exposure 6000` = 6 ms; lebih rendah dari auto-exposure default (~33 ms) sehingga blur gerak berkurang tanpa mengubah FPS atau mengorbankan keselarasan depth.

Ekspor point cloud berwarna dari satu paket frame ekspor:

```bash
python -m studio_rgbd.ekspor_pointcloud \
  --frame ../dataset/studio_rgbd/rekaman/tangga_naik/SESI/exports/frames/frame_000450 \
  --out ../hasil/aktif/pointcloud/frame_000450.ply
```

Hasil PLY berformat binary, koordinat dalam meter, bisa dibuka di CloudCompare atau MeshLab.

Folder `.venv/` adalah lingkungan Python proyek dan sengaja tidak dipindahkan.

## Fine-tune dengan cross-validation

Gunakan dataset tangga lama ditambah satu video baru yang sudah dilabeli.
Pembagian grouped 5-fold menjaga varian satu gambar dan frame satu video
tetap berada pada fold yang sama:

```bash
python train_tangga_cv.py --folds 5 --epochs 30 --device cpu
```

Pada komputer dengan GPU NVIDIA, gunakan `--device 0` atau alias `--device gpu`.

Pantau progres di terminal lain:

```bash
python pantau_cv.py
```

Untuk hanya membuat dan memeriksa pembagian fold:

```bash
python train_tangga_cv.py --folds 5 --prepare-only
```

Bobot tiap fold tersimpan di `../bobot/kandidat/tangga_grouped_cv/`.
Kelas tetap `0: tapakan` dan `1: bidang_tegak`.

Untuk uji video RGB-D dengan mask temporal, validasi bidang 3-D, dan ID
permukaan persisten:

```bash
python inference/uji_tangga_bytetrack_depth.py \
  --bag ../dataset/studio_rgbd/rekaman/tangga_naik/TANGGA_NAIK_20260830_110530/source/raw.db3 \
  --model ../bobot/kandidat/tangga_grouped_cv/fold_0/weights/best.pt \
  --out ../hasil/aktif/stair_perception_fold0 \
  --device 0
```

ByteTrack dipakai sebagai cue gerak 2-D, tetapi bukan satu-satunya pemilik ID.
Sistem menggabungkannya dengan IoU mask, posisi gambar, depth, normal bidang,
dan deduplikasi proposal. `track_id` tetap milik track permukaan, sedangkan
`step_index` hanya urutan bawah-ke-atas yang dapat berubah ketika kamera bergerak.

Untuk pengujian jujur pada video `TANGGA_NAIK_20260830_110530`, pakai bobot
`fold_0`: video ini berada di validasi fold 0 dan tidak ikut melatih fold 0.
Jangan memakai fold 1–4 untuk mengklaim generalisasi pada video yang sama,
karena video tersebut masuk ke data train pada fold-fold itu.

Hasil utama:

- `annotated_clean.mp4`: kartu Poppins semi-transparan di pusat bidang;
- `annotated_minimal.mp4`: titik dan tulisan saja, tanpa fill mask tangga;
- `annotated_stable.mp4`: panel proposal mentah dan hasil stabil untuk audit;
- `tracks.csv`, `measurements.csv`, `observations.csv`, dan `frames.csv`;
- `summary.json` dan `run.json` untuk konfigurasi serta metrik.

Ambang dapat diubah di `inference/stair_perception.yaml`. Status `EST` adalah
estimasi awal pada sekitar 0,60–2,50 m, `LIVE` adalah hasil pada zona terbaik
0,75–1,20 m, dan `LOCK` mempertahankan hasil LIVE saat track/depth hilang
singkat. Bidang tegak muncul lebih awal sebagai `RISER AHEAD` tanpa mengarang
H/R. Model saat ini hanya mempunyai kelas tangga; model masa depan dengan kelas
seperti batu akan dirender dengan full segmentation untuk kelas non-tangga.
