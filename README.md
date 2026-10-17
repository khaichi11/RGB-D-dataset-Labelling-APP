# RGB-D Dataset Labelling App

Aplikasi desktop Tkinter untuk merekam, memilah, melabeli, dan membagi dataset
RGB-D dari Intel RealSense D435 (tangga naik, batu, ramp naik), dengan
auto-label ConvNeXt RGB-D.

## Windows

`pasang.sh` untuk Linux. Di Windows clone repo ini ke `C:\skripsi_tangga` (dengan
`-c core.autocrlf=false`), lalu repo model ke `C:\skripsi_tangga\model` dan ikuti `model/WINDOWS.md`
(pemasangan, data, notebook skripsi).

## Pasang di laptop baru (satu perintah)

```bash
git clone https://github.com/khaichi11/RGB-D-dataset-Labelling-APP.git skripsi_tangga
cd skripsi_tangga
./pasang.sh
```

`pasang.sh` mengerjakan semuanya dan aman dijalankan ulang:

1. Memasang paket sistem `python3-venv`, `python3-tk`, dan `git` (meminta sudo sekali).
2. Mengambil repo **Train-RGB-D-Model** (privat; berisi pustaka model dan bobot
   rujukan) ke `skripsi_tangga/model`. Bila diminta masuk GitHub,
   isi username dan *Personal Access Token*, bukan password. Bila `gh` sudah
   login, langkah ini otomatis.
3. Membuat virtualenv `studio/.venv` dan memasang `studio/requirements.txt`
   (torch sekitar 2–3 GB).
4. Memasang penjaga rahasia Git: hook yang menolak token, kunci API, email, dan
   berkas `.env`, serta email commit noreply GitHub.
5. Meminta login Hugging Face. Buat token jenis **Write** di
   <https://huggingface.co/settings/tokens/new?tokenType=write> lalu tempel.
   Token disimpan di `~/.cache/huggingface/`, di luar repo. Boleh dilewati
   (Enter kosong); login juga bisa dilakukan nanti dari tombol Push di aplikasi.
6. Mengunduh model pengusul label yang dipakai Studio di laptop pengembang
   (penanda DIPAKAI STUDIO di registri model Hugging Face). Tanpa login, Studio
   memakai bobot rujukan lama yang ada di repo Train dan usulannya lebih buruk.
7. Memasang aturan udev RealSense (`perangkat/desktop/99-realsense-libusb.rules`, salinan
   aturan resmi Intel) agar D435 terbaca tanpa sudo; cabut lalu colok ulang kamera.
8. Menambahkan ikon **Studio Dataset RGB-D** ke menu aplikasi, lalu menguji
   impor semua pustaka dan GPU.

Setelah itu buka dari menu aplikasi, atau `perangkat/desktop/jalankan-studio.sh`.

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
git clone https://github.com/khaichi11/RGB-D-dataset-Labelling-APP.git skripsi_tangga
cd skripsi_tangga
git clone https://github.com/khaichi11/Train-RGB-D-Model.git model
python -m venv studio/.venv
studio/.venv/bin/python -m pip install -r studio/requirements.txt    # Windows: studio\.venv\Scripts\python
git config core.hooksPath .githooks
git -C model config core.hooksPath .githooks
studio/.venv/bin/hf auth login                                     # opsional
cd studio && .venv/bin/python -m studio_rgbd.studio_dataset_rgbd --preset jangan
```

Susunan folder (16 Oktober 2026). Studio mencari model di `model/` di sebelahnya:

```
skripsi_tangga/                 <- repo ini (RGB-D-dataset-Labelling-APP)
├── README.md, pasang.sh        <- panduan dan pemasang
├── studio/studio_rgbd/         <- aplikasi Studio (BRIEF.md: tab -> berkas -> letak masalah)
├── model/                      <- repo Train-RGB-D-Model: rgbd_convnext, pelatihan, bukti, naskah
│   └── aplikasi_tangga/rgbd_convnext/BRIEF.md   <- peta kode dan tabel gejala -> letak
├── ukur_tangga/                <- repo ukur-tangga: PWA iPad isian meteran
├── dataset/                    <- rekaman D435 dan data publik (lokal, tidak masuk Git)
├── bobot/                      <- checkpoint besar (pra_latih, kandidat, aktif)
├── naskah -> model/naskah      <- proposal, slide, gambar, judul & keputusan (repo privat Train)
├── referensi/                  <- paper, repo pembanding, catatan telaah
├── perangkat/                  <- setup Ubuntu/Jetson, peluncur desktop Studio
└── arsip/                      <- hasil, keluaran, laporan, eksperimen U-Net, brief GPT lama
```

Folder `~/paket_ubuntu_zenexo` (nama lama) kini hanya berisi tautan ke folder di atas agar
jalur lama di bukti dan data tetap berjalan; lihat README di sana.

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

Panduan lengkap: [PANDUAN_STUDIO_DATASET_RGBD.md](studio/studio_rgbd/PANDUAN_STUDIO_DATASET_RGBD.md).

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
  `skripsi_tangga/.env` (lihat `studio/studio_rgbd/.env.example`). Semua
  `.env` diabaikan Git di folder mana pun.
- Hook `.githooks/cek_rahasia.sh` menolak commit yang menambahkan token atau
  kunci API, alamat email, berkas `.env`, atau commit dengan email non-noreply.
  Lewati hanya bila yakin salah deteksi: `git commit --no-verify`.
- Dataset, rekaman mentah `.bag`/`.db3`, point cloud, dan hasil ekspor tidak
  pernah masuk Git karena besar dan dapat memuat lokasi atau orang.
- Repo dataset dan model di Hugging Face dibuat **privat**.

Pengecualian terbatas ada pada [artefak U-Net](arsip/artefak_unet/): satu checkpoint
terbaik hasil *fine-tuning*, contoh gambar, dan catatan eksperimen. Artefak ini
membantu pemeriksaan decoder U-Net; ia bukan pengganti dataset dan bukan bukti
evaluasi akhir.
