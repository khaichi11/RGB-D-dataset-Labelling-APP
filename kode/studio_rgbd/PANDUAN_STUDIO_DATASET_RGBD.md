# ZenExo Studio RGB-D

Jalankan dari folder `kode`:

```bash
source .venv/bin/activate
python studio_dataset_rgbd.py --keluar ../dataset/studio_rgbd --preset jangan
```

## Alur singkat

1. Tab **Rekam**: pilih `batu`, `tangga_naik`, atau `ramp_naik`, pilih split, lalu tekan **Mulai rekam**.
2. Rekam berbagai sudut dan jarak. Tekan **Selesai rekaman** saat adegan selesai.
3. Tab **Tinjau & Potong**: pilih rekaman dan tekan **Buat preview** hanya bila ingin melihatnya. Atur awal/akhir dan simpan rentangnya.
4. Tab **Ekspor Frame**: pilih FPS tidak lebih tinggi dari FPS kamera, kemudian ekspor. Setiap frame membawa RGB, depth Z16, depth selaras RGB, IR, timestamp, dan metadata kamera.
5. Tab **Label & Ukur**: pilih frame, gambar poligon objek biru untuk YOLO, dan poligon acuan hijau bila ingin menghitung tinggi. Overlay depth dapat dibuat samar agar batas label lebih masuk akal.
6. Tekan **Bangun folder dataset YOLO** setelah label disimpan. Hasil berada pada `dataset_yolo_seg/images/<split>` dan `dataset_yolo_seg/labels/<split>`.

## Data aman

- `source/raw.bag` adalah rekaman asli; jangan ubah atau hapus.
- Pemotongan hanya membuat `edit/rentang.json`; rekaman asli tidak pernah dipotong.
- Frame/video yang tidak layak cukup dipindahkan ke tempat sampah. Aplikasi hanya memberi tanda status, tidak menghapus raw atau label.
- Jika kalibrasi depth perlu diperbaiki, ekspor ulang frame dari `raw.bag`; tidak perlu mengambil data baru.
