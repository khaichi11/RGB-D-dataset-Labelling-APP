# RGB-D Dataset Labelling App

Aplikasi desktop Tkinter untuk merekam, memilah, melabeli, dan membagi dataset
RGB-D dari Intel RealSense D435 (tangga naik, batu, ramp naik), dengan
auto-label ConvNeXt RGB-D.

## Pasang di laptop baru (satu perintah)

```bash
git clone https://github.com/khaichi11/RGB-D-dataset-Labelling-APP.git paket_ubuntu_zenexo
cd paket_ubuntu_zenexo
./pasang.sh
```

`pasang.sh` mengerjakan semuanya dan aman dijalankan ulang:

1. Memasang paket sistem `python3-venv`, `python3-tk`, dan `git` (meminta sudo sekali).
2. Mengambil repo **Train-RGB-D-Model** (privat; berisi pustaka model dan bobot
   rujukan) ke `paket_ubuntu_zenexo/Train-RGB-D-Model`. Bila diminta masuk GitHub,
   isi username dan *Personal Access Token*, bukan password. Bila `gh` sudah
   login, langkah ini otomatis.
3. Membuat virtualenv `kode/.venv` dan memasang `kode/requirements.txt`
   (torch sekitar 2–3 GB).
4. Memasang penjaga rahasia Git: hook yang menolak token, kunci API, email, dan
   berkas `.env`, serta email commit noreply GitHub.
5. Meminta login Hugging Face. Buat token jenis **Write** di
   <https://huggingface.co/settings/tokens/new?tokenType=write> lalu tempel.
   Token disimpan di `~/.cache/huggingface/`, di luar repo. Boleh dilewati
   (Enter kosong); login juga bisa dilakukan nanti dari tombol Push di aplikasi.
6. Mengunduh model pengusul label terbaru dari registri model Hugging Face.
   Tanpa login, Studio memakai bobot rujukan yang sudah ada di repo Train.
7. Menambahkan ikon **Studio Dataset RGB-D** ke menu aplikasi, lalu menguji
   impor semua pustaka dan GPU.

Setelah itu buka dari menu aplikasi, atau `desktop/jalankan-studio.sh`.

### Apakah jalan di semua laptop?

Jujurnya: tidak di semua. Yang sudah diuji dan didukung penuh adalah Linux x86_64.

| Laptop | Hasil |
|---|---|
| Ubuntu 22.04/24.04 x86_64, GPU NVIDIA | Semua fitur. Auto-label sekitar 20 ms/frame. |
| Ubuntu x86_64 tanpa GPU NVIDIA | Semua fitur, tetapi auto-label dan model berjalan di CPU (beberapa ratus ms/frame). |
| Windows 10/11 x86_64 | Aplikasinya jalan (Python, Tkinter, pyrealsense2, torch tersedia), tetapi `pasang.sh` tidak. Pasang manual: lihat di bawah. |
| macOS Apple Silicon | Melabel dan split bisa; merekam kamera **tidak**, karena pyrealsense2 tidak menyediakan wheel. |
| Jetson / ARM | Tidak lewat `pasang.sh`; pyrealsense2 harus dibangun dari sumber. |

Yang **tidak** ikut terpasang otomatis adalah **dataset**
(`dataset/studio_rgbd/`, ratusan GB rekaman mentah). Salin folder itu dari
laptop lama atau disk eksternal. Cadangan frame berlabel (Parquet) ada di
dataset Hugging Face `khaichi11/Stairs-Skripsi`; cadangan ini untuk pelatihan,
bukan pengganti rekaman mentah.

### Pasang manual (Windows atau tanpa skrip)

```bash
git clone https://github.com/khaichi11/RGB-D-dataset-Labelling-APP.git paket_ubuntu_zenexo
cd paket_ubuntu_zenexo
git clone https://github.com/khaichi11/Train-RGB-D-Model.git
python -m venv kode/.venv
kode/.venv/bin/python -m pip install -r kode/requirements.txt      # Windows: kode\.venv\Scripts\python
git config core.hooksPath .githooks
git -C Train-RGB-D-Model config core.hooksPath .githooks
kode/.venv/bin/hf auth login                                       # opsional
cd kode && .venv/bin/python -m studio_rgbd.studio_dataset_rgbd --preset jangan
```

Susunan folder wajib seperti ini, karena Studio mencari model di sebelahnya:

```
paket_ubuntu_zenexo/            <- repo ini
├── kode/studio_rgbd/           <- aplikasi
├── Train-RGB-D-Model/          <- repo Train (pustaka rgbd_convnext + bobot)
└── dataset/studio_rgbd/        <- data lokal, tidak masuk Git
```

## Alur kerja di aplikasi

| Tab | Isi |
|---|---|
| 1. Rekam | Rekam RGB, depth, dan IR D435 ke `.db3` mentah (tidak pernah diubah). |
| 2. Tinjau & Potong | Putar rekaman, potong non-destruktif, **stabilo warna**, **catatan/lokasi**, dan **scene** (rentang frame bernama) per rekaman. Kartu rekaman menampilkan **ukuran berkas** (mentah, turunan, ekspor, jumlah berkas) dan **⏱ Cek sinkron RGB–depth–IR**. |
| 3. Ekspor Frame | Ekspor frame RGB-D berpasangan dari rentang terpilih. |
| 4. Label & Ukur | Poligon riser (merah) dan tread (biru), auto-label ConvNeXt RGB-D, pemandu riser/tread, **Enter** = tandai sudah diperiksa manual lalu lanjut. Lapisan IR/depth dikoreksi rolling shutter per baris ke RGB; info frame menunjukkan apakah RGB, depth, dan IR satu jepretan; **📐 garis lipatan 3-D** dari depth membedakan ujung/pangkal anak tangga dari bayangan dan noda. |
| 5. Uji Realtime | Uji model pada kamera langsung. |
| 6. Uji Model pada Berkas | Uji model pada gambar, video, atau folder frame. |
| 7. Split Dataset | Bagi rekaman, scene, atau frame ke train/val/test dengan kalkulator rasio dan rekomendasi; pratinjau mask persis seperti yang dilatih; salin perintah latih/uji; **push dataset ke Hugging Face**. |

Panduan lengkap: [PANDUAN_STUDIO_DATASET_RGBD.md](kode/studio_rgbd/PANDUAN_STUDIO_DATASET_RGBD.md).

Setelah membagi dataset di tab 7, pelatihan tinggal satu perintah dari tombol
**📋 Perintah latih** (tempel di terminal). Model hasil latih otomatis masuk
registri model Hugging Face `khaichi11/Skripsi` bila laptop sudah login.

## Dataset di Hugging Face: melabel ulang di laptop mana pun

Tab **7. Split Dataset** punya dua tombol:

- **☁ Push dataset ke HF**: semua frame ekspor (berlabel maupun belum) diunggah
  sebagai Parquet ke `khaichi11/Stairs-Skripsi` (privat): RGB, depth, IR, label,
  status diperiksa, sampah, catatan rekaman, dan pembagian split.
- **⬇ Tarik dari HF**: frame dibangun kembali ke `dataset/studio_rgbd/rekaman/...`.
  Rekaman hasil tarik tampil "(tanpa video)" di tab Tinjau dan bisa langsung
  dilabel. Gambar yang sudah ada tidak ditimpa; label lokal yang lebih baru
  dipertahankan, label dari HF yang lebih baru dipakai.

Alur di beberapa laptop: **Tarik → melabel → Push**.

Video mentah (`raw.db3`, ±95 GB) **tidak** diunggah. Melabel ulang hanya butuh
frame ekspor; video hanya diperlukan untuk mengekspor frame baru. Simpan video
di disk eksternal sebagai arsip.

## Keamanan data

- Token hanya di `~/.cache/huggingface/` (hasil login) atau di
  `paket_ubuntu_zenexo/.env` (lihat `kode/studio_rgbd/.env.example`). Semua
  `.env` diabaikan Git di folder mana pun.
- Hook `.githooks/cek_rahasia.sh` menolak commit yang menambahkan token atau
  kunci API, alamat email, berkas `.env`, atau commit dengan email non-noreply.
  Lewati hanya bila yakin salah deteksi: `git commit --no-verify`.
- Dataset, rekaman mentah `.bag`/`.db3`, point cloud, dan hasil ekspor tidak
  pernah masuk Git karena besar dan dapat memuat lokasi atau orang.
- Repo dataset dan model di Hugging Face dibuat **privat**.

Pengecualian terbatas ada pada [artefak U-Net](artefak_unet/): satu checkpoint
terbaik hasil *fine-tuning*, contoh gambar, dan catatan eksperimen. Artefak ini
membantu pemeriksaan decoder U-Net; ia bukan pengganti dataset dan bukan bukti
evaluasi akhir.
