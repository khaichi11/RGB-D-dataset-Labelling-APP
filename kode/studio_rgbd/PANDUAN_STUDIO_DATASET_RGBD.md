# ZenExo Studio RGB-D

Jalankan dari folder `kode`:

```bash
source .venv/bin/activate
python -m studio_rgbd.studio_dataset_rgbd --keluar ../dataset/studio_rgbd --preset jangan
```

## Alur singkat

1. Tab **Rekam**: pilih `batu`, `tangga_naik`, atau `ramp_naik`, pilih split, lalu tekan **Mulai rekam**.
2. Rekam berbagai sudut dan jarak. Tekan **Selesai rekaman** saat adegan selesai.
3. Tab **Tinjau & Potong**: tekan **Putar** untuk melihat RAW langsung, tanpa menunggu preview lengkap. Tekan **Siapkan indeks & preview lengkap** hanya bila perlu lompat frame, memilih awal/akhir, ekspor rentang, atau mengukur 3-D presisi.
4. Tab **Ekspor Frame**: pilih FPS tidak lebih tinggi dari FPS kamera, kemudian ekspor rentang. Setiap frame dibaca dari RAW dan membawa RGB, depth Z16, depth selaras RGB, IR, timestamp, serta metadata kamera. Pemilihan frame menggunakan timestamp kamera, bukan asumsi 30 FPS.
5. Tab **Label & Ukur**: setelah preview lengkap tersedia, tombol **Ekspor frame video saat ini ke Label** mengirim satu paket frame yang sedang ditinjau langsung ke editor. Tarik titik mask untuk memindahkannya; lup muncul saat titik digeser. Mask merah mengikuti kategori (`batu`, `ramp`, atau sisi tinggi) dan biru adalah bidang acuan untuk pengukuran tinggi.
6. Tekan **Bangun folder dataset YOLO** setelah label disimpan. Hasil berada pada `dataset_yolo_seg/images/<split>` dan `dataset_yolo_seg/labels/<split>`.

## Data aman

- `source/raw.bag` adalah rekaman asli; jangan ubah atau hapus.
- Pemotongan hanya membuat `edit/rentang.json`; rekaman asli tidak pernah dipotong.
- Frame/video yang tidak layak dapat dipindahkan ke tempat sampah untuk dipulihkan kemudian. Bila memang tidak diperlukan lagi, tombol **Hapus frame ini permanen** hanya menghapus paket ekspor terpilih (RGB/depth/IR/mask/label); `source/raw.*` tidak pernah disentuh.
- Preview dan ekspor memakai FPS yang diukur dari timestamp rekaman, bukan asumsi 30 fps. Menekan **Ekspor** lagi MELANJUTKAN ekspor lama: frame yang sudah ada (termasuk yang di tempat sampah) dilewati, bukan diekspor ulang.
- Jika kalibrasi depth perlu diperbaiki, ekspor ulang frame dari `raw.bag`; tidak perlu mengambil data baru.
