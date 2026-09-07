# RGB-D Dataset Labelling App

Aplikasi desktop Tkinter untuk mengumpulkan dan melabeli dataset RGB-D dari
Intel RealSense D435 untuk objek batu, tangga naik, dan ramp naik.

## Fitur

- Merekam RGB, depth Z16, infrared, timestamp, intrinsics, dan extrinsics
  sebagai rekaman RealSense `.bag` mentah.
- Preview RGB dan depth hanya diaktifkan saat diperlukan agar proses labeling
  tetap ringan.
- Pemotongan non-destruktif: rekaman `.bag` asli tidak pernah diubah.
- Ekspor frame RGB-D berpasangan pada FPS yang dipilih pengguna.
- Pelabelan poligon YOLO instance segmentation dengan zoom, pan, dan overlay
  depth.
- Pencocokan jarak serta tinggi objek terhadap bidang acuan 3-D.
- Tempat sampah berbasis status, tanpa menghapus raw data.
- Ekspor struktur `images/<split>` dan `labels/<split>` untuk Ultralytics YOLO.

## Menjalankan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kode/requirements.txt
cd kode
python -m studio_rgbd.studio_dataset_rgbd --preset jangan
```

Lihat [panduan penggunaan](kode/studio_rgbd/PANDUAN_STUDIO_DATASET_RGBD.md) untuk alur
rekam, tinjau, potong, ekspor, label, dan ukur.

## Data dan privasi

Dataset, rekaman mentah `.bag`/`.db3`, point cloud, dan hasil ekspor rutin tetap
diabaikan oleh Git karena besar dan dapat memuat data lokasi atau objek nyata.

Pengecualian terbatas ada pada [artefak U-Net](artefak_unet/): satu checkpoint
terbaik hasil *fine-tuning*, contoh gambar, serta catatan eksperimen. Video RAW
tetap lokal karena dapat memperlihatkan lingkungan pengambilan data. Artefak
yang dibagikan membantu pemeriksaan decoder U-Net, bukan sebagai pengganti
dataset dan bukan bukti evaluasi akhir.
