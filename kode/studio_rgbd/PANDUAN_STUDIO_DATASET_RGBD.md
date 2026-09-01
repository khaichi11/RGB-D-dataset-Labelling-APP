# ZenExo Studio RGB-D

Jalankan dari folder `kode`:

```bash
source .venv/bin/activate
python -m studio_rgbd.studio_dataset_rgbd --keluar ../dataset/studio_rgbd --preset jangan
```

Untuk rekaman dengan exposure manual (anti-blur saat kamera bergerak, misalnya di pinggang):

```bash
python -m studio_rgbd.studio_dataset_rgbd --keluar ../dataset/studio_rgbd --preset akurasi --exposure 6000
```

`--exposure` dalam mikro-detik. Nilai 0 (default) = auto-exposure. Disarankan 4000–8000 µs bila kamera dibawa berjalan. Nilai ini tidak memengaruhi depth — depth D435 menggunakan IR terpisah.

## Alur singkat

1. Tab **Rekam**: pilih `batu`, `tangga_naik`, atau `ramp_naik`, pilih split, lalu tekan **Mulai rekam**.
2. Rekam berbagai sudut dan jarak. Tekan **Selesai rekaman** saat adegan selesai.
3. Tab **Tinjau & Potong**: tekan **Putar** untuk melihat RAW langsung, tanpa menunggu preview lengkap. RAW adalah rekaman RealSense `raw.db3`/`raw.bag`, bukan MP4 biasa. Tekan **Siapkan indeks & preview lengkap** hanya bila perlu lompat frame, memilih awal/akhir, ekspor rentang, atau mengukur 3-D presisi.
4. Tab **Ekspor Frame**: pilih FPS tidak lebih tinggi dari FPS kamera, kemudian ekspor rentang. Setiap frame dibaca dari RAW dan membawa RGB, depth Z16, depth selaras RGB, IR, timestamp, serta metadata kamera. Pemilihan frame menggunakan timestamp kamera, bukan asumsi 30 FPS.
5. Tab **Label & Ukur**: setelah preview lengkap tersedia, tombol **Ekspor frame video saat ini ke Label** mengirim satu paket frame yang sedang ditinjau langsung ke editor. Tarik titik mask untuk memindahkannya; lup muncul saat titik digeser. Mask merah mengikuti kategori (`batu`, `ramp`, atau sisi tinggi) dan biru adalah bidang acuan untuk pengukuran tinggi.
6. Tekan **Bangun folder dataset YOLO** setelah label disimpan. Hasil berada pada `dataset_yolo_seg/images/<split>` dan `dataset_yolo_seg/labels/<split>`. Untuk tangga, hanya gambar yang memuat **tapakan dan bidang tegak** yang dimasukkan ke dataset latih.

## Data aman

- `source/raw.*` adalah rekaman asli; SDK RealSense modern menyimpan `raw.db3`, sedangkan sesi lama dapat berisi `raw.bag`. Jangan ubah atau hapus.
- Pemotongan hanya membuat `edit/rentang.json`; rekaman asli tidak pernah dipotong.
- Frame/video yang tidak layak dapat dipindahkan ke tempat sampah untuk dipulihkan kemudian. Bila memang tidak diperlukan lagi, tombol **Hapus frame ini permanen** hanya menghapus paket ekspor terpilih (RGB/depth/IR/mask/label); `source/raw.*` tidak pernah disentuh.
- Preview dan ekspor memakai FPS yang diukur dari timestamp rekaman, bukan asumsi 30 fps. Menekan **Ekspor** lagi MELANJUTKAN ekspor lama: frame yang sudah ada (termasuk yang di tempat sampah) dilewati, bukan diekspor ulang.
- Jika kalibrasi depth perlu diperbaiki, ekspor ulang frame dari `raw.db3`/`raw.bag`; tidak perlu mengambil data baru.
- `preview.mp4` hanya turunan untuk ditonton. Ia tidak dapat dipakai membuat mesh: gunakan RAW atau paket frame yang menyertakan depth Z16 dan intrinsics.

## Point cloud dan mesh

Studio menyimpan bahan mesh yang diperlukan: RGB, depth Z16, timestamp, intrinsics, dan extrinsics. Rekonstruksi mesh dilakukan dari RAW dengan skrip `studio_rgbd/rekonstruksi_mesh.py`, bukan dari `preview.mp4`.

```bash
python -m studio_rgbd.rekonstruksi_mesh \
  --raw ../dataset/studio_rgbd/rekaman/tangga_naik/NAMA_SESI/source/raw.db3 \
  --out ../hasil/mesh/NAMA_SESI
```

Outputnya adalah `pointcloud_fused.ply`, `mesh_poisson.ply`, dan `laporan_mesh.json`. Hasil mesh hanya layak bila kamera bergerak perlahan dengan area tangga saling tumpang tindih antar-frame; pantulan/permukaan tanpa depth akan dicatat sebagai frame yang ditolak.

## Catatan integrasi dan jejak masalah

Bagian ini menjelaskan keputusan teknis yang penting agar hasil inferensi dan
label dapat dilacak bila tampak tidak sesuai.

| Gejala / risiko | Aturan yang diterapkan | Lokasi pemeriksaan |
| --- | --- | --- |
| Mask tangga tampak jauh lebih buruk dibanding data training | `color_raw.png` dibaca OpenCV dalam urutan **BGR**. Kanvas dan SAM 2 memakai RGB, sehingga aplikasi secara eksplisit mengubah RGB kembali ke BGR sebelum memanggil YOLO. Jangan menghapus konversi ini. | `studio_dataset_rgbd.py` pada `usulkan_segmentasi()` dan `_rekomendasi_tangga_data()` |
| Batas mask berubah atau jumlah riser tidak konsisten | Bobot tangga dilatih memakai letterbox **512 px**. Inferensi YOLO selalu memakai `imgsz=512`; koordinat mask kemudian dikembalikan ke resolusi RGB asli sehingga tetap sejajar dengan depth. | `segmentasi_yolo_depth.py`: `IMGSZ_TANGGA = 512` |
| RGB dan depth tidak tepat pasangan atau ekspor dianggap selalu 30 FPS | Paket ekspor selalu berisi `color_raw.png`, `depth_raw_z16.npy`, `depth_aligned_to_color.npy`, IR, timestamp, dan metadata dari frame RAW yang sama. Pemilihan waktu memakai timestamp kamera, bukan asumsi FPS tetap. | `<frame>/frame.json` dan `<frame>/depth_aligned_to_color.npy` |
| Rekomendasi tangga kasar | Alurnya: **YOLO → SAM 2 → verifikasi depth**. YOLO memberi instance/tapak-riser, SAM 2 merapikan batas, dan depth hanya menolak serpihan/outlier—depth tidak boleh mengganti bentuk YOLO secara diam-diam. | `segmentasi_yolo_depth.py`, `segmentasi_sam2.py` |
| Batu atau ramp belum mendapat mask setara tangga | Saat ini batu/ramp memakai usulan depth karena belum ada bobot YOLO khusus. Gunakan label manual sebagai dataset untuk melatih bobot kelas batu/ramp sebelum menjadikannya rekomendasi otomatis. | `studio_dataset_rgbd.py`: `usulkan_segmentasi()` |
| Label hilang setelah berpindah frame | Editor menyimpan `label_draft.json` otomatis setelah perubahan berhenti. Label siap latih juga ditulis ke `label_yolo_seg.txt`; RAW tidak ikut diubah. | Folder paket frame ekspor |
| Editor berat saat banyak titik atau ketika zoom | Gambar RGB dasar dicache; hanya overlay mask yang digambar ulang. Drag/zoom dibatasi sekitar 30 FPS, dan zoom memakai render cepat dulu lalu kualitas tajam setelah roda mouse berhenti. | `KanvasLabel` di `studio_dataset_rgbd.py` |

### Checklist saat hasil rekomendasi tidak masuk akal

1. Pastikan `frame.json` menunjukkan kategori yang benar dan file RGB, depth, serta intrinsics tersedia dalam paket frame yang sama.
2. Pastikan bobot aktif ada di `bobot/kandidat/manual_tangga/weights/best.pt` atau fallback `bobot/aktif/tangga_yolo26s_seg_512_best.pt`.
3. Jangan mengubah BGR/RGB atau `imgsz=512` tanpa melatih dan mengevaluasi ulang model pada konfigurasi baru.
4. Buka overlay depth untuk memeriksa penyelarasan, lalu koreksi kandidat menggunakan opacity mask, magnet titik, dan lup. Simpan terjadi otomatis.
5. Untuk reproduksi, catat nama sesi, nama folder frame, commit aplikasi, bobot yang dipakai, dan apakah hasil berasal dari rekomendasi otomatis atau koreksi manual.
