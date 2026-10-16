# Varian ConvNeXt RGB-D dengan decoder U-Net

Folder ini adalah eksperimen terpisah. Baseline FPN di `kode/stair_fusion_atto`
tidak diubah. Kedua varian memakai data, encoder ConvNeXt, fusi RGB-D, kepala
garis, *loss function*, dan metrik yang sama. Perbedaan tunggalnya adalah
decoder: FPN pada baseline dibandingkan dengan koneksi-lewati U-Net di sini.
Kepala semantik bawaan memakai *stride* 2, sama seperti eksperimen pembanding
ConvNeXt Atto; gunakan `--semantic-stride 1` hanya sebagai ablasi terpisah.

## Struktur

- `model_unet.py`: model ConvNeXt V2 Atto RGB-D dengan fusi empat skala dan
  decoder U-Net ringan.
- `train_unet.py`: pelatihan serta penyimpanan checkpoint dan metrik.
- `finetune_d435_unet.py`: penyesuaian terpisah pada rekaman D435 berlabel.
- `smoke_test.py`: pemeriksaan bentuk masukan dan keluaran, tanpa pelatihan.
- `bandingkan_video_d435.py`: membuat video FPN dan U-Net berdampingan dari
  rekaman RAW D435 yang sama, tanpa penghalusan temporal.

## Uji cepat

Jalankan dari akar proyek:

```bash
python unet-new/smoke_test.py
```

## Pelatihan pembanding yang adil

Gunakan pembagian data, jumlah epoch, *seed*, ukuran masukan, dan parameter
pelatihan yang sama dengan baseline FPN. Contoh:

```bash
python unet-new/train_unet.py \
  --data dataset/rgbd_stair \
  --output bobot/kandidat/cnx_atto_unet \
  --varian cnx_atto_in1k \
  --epochs 24 --batch 4 --seed 2026
```

Model final tidak ditentukan dari Dice saja. Bandingkan `mean_dice`,
`boundary_f1`, kesalahan lantai sebagai tapakan, waktu inferensi, dan memori
pada Jetson Orin Nano. Jika U-Net lebih tajam tetapi melampaui batas latensi
atau memori, baseline FPN tetap pilihan yang lebih tepat.

`--no-pretrained` hanya untuk pengujian lokal atau ablasi. Eksperimen utama
memakai `cnx_atto_in1k` agar konsisten dengan model ConvNeXt V2 Atto berpralatih
ImageNet yang disebut dalam proposal.

## Penyesuaian pada data D435

Setelah pra-latih publik selesai, gunakan checkpoint U-Net terbaik untuk
*fine-tune* pada rekaman D435. Rekaman validasi ditahan utuh dan tidak boleh
masuk ke pelatihan. Contoh:

```bash
python unet-new/finetune_d435_unet.py \
  --init bobot/kandidat/cnx_atto_unet/best.pt \
  --output bobot/kandidat/cnx_atto_unet_d435 \
  --epochs 40
```

Rekaman batu atau lantai bukan kelas baru pada eksperimen ini. Simpan sebagai
data negatif, lalu ukur apakah model tetap memberi kelas latar dan tidak
menghasilkan tapakan palsu.

## Perbandingan visual yang dapat diulang

Gunakan checkpoint dan rekaman yang sama pada kedua sisi. Video hanya membantu
inspeksi visual; nilai akhir tetap dihitung dari label pada set uji terpisah.

```bash
python unet-new/bandingkan_video_d435.py \
  --bag dataset/studio_rgbd/rekaman/tangga_naik/TANGGA_NAIK_20260830_110152/source/raw.db3 \
  --fpn bobot/kandidat/final_d435/cnx_atto_in1k/best.pt \
  --unet bobot/kandidat/cnx_atto_unet/d435_110152_holdout/best.pt \
  --output output/banding_fpn_unet_110152.mp4
```

Artefak checkpoint dan contoh gambar perbandingan tersedia di
`../artefak_unet/`. Lihat catatannya sebelum menarik kesimpulan bahwa salah satu
decoder lebih baik. Video RAW tetap disimpan lokal karena dapat memuat lingkungan
pengambilan data.
