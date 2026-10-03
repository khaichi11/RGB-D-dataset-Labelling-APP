# RGB-D Labelling Studio

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
7. Tab **Uji Sistem Tangga** (opsional, memeriksa sistem jadi): menjalankan mekanisme yang sama dengan perintah `tangga` di `Train-RGB-D-Model/aplikasi_tangga` -- model akhir ConvNeXt V2 Atto RGB-D 384 (FP16 + graf CUDA), pelacak geometri, dan ukuran anak tangga -- pada **Sesi aktif**, rekaman pilihan, atau kamera D435. Centang **Putar seperti kamera** agar rekaman diputar pada laju aslinya (frame yang tertinggal dibuang, seperti kamera). Panel menampilkan frame/detik per detik, latensi, waktu per tahap, frame terlewat, dan ukuran tiap anak tangga; centang **Simpan video + JSON** untuk menyimpannya ke `aplikasi_tangga/keluaran/`.
8. **Ukuran meteran sambil melabel** (kartu *Ukuran tangga (meteran)* di tab Label): tekan **Hitung nomor dari video** sekali per sesi (±20 detik). Pelacak sistem berjalan pada setiap frame rekaman, sehingga nomor anak tangga (#1 = paling bawah) konsisten seperti di video, lalu ditandai di atas frame yang sedang dilabel; **▶ Video bernomor** memutar urutannya. Isi tinggi riser dan panjang tread hasil meteran per anak tangga, di samping angka sistem, lalu **Simpan ukuran meteran**. Bila sesi/scene sudah dikaitkan ke tangga fisik (tab Tinjau), ukurannya disimpan untuk tangga itu dan berlaku di semua rekamannya. Bila pelacak mulai dari anak tangga fisik ke-2, ubah "Nomor sistem #1 = anak tangga fisik ke-". Ukuran dari aplikasi iPad *Ukur Tangga* dapat dimasukkan dengan `python -m studio_rgbd.ukuran_meteran --impor <berkas.json>`, dan `python -m studio_rgbd.ukuran_meteran` mencetak CSV selisih sistem - meteran per anak tangga.

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

Catatan masalah yang pernah terjadi beserta aturan yang dipakai sekarang.

| Gejala / risiko | Aturan yang diterapkan | Lokasi pemeriksaan |
| --- | --- | --- |
| Mask jauh lebih buruk daripada saat pelatihan | `color_raw.png` dibaca OpenCV sebagai **BGR**, sedangkan kanvas dan SAM 2 memakai RGB. Konversi baliknya wajib; pernah terlewat dan menyebabkan mask bocor. | `usulkan_segmentasi()`, `_rekomendasi_tangga_data()` |
| Batas mask berubah atau jumlah riser tidak konsisten | Bobot ConvNeXt V2 Atto RGB-D dilatih pada letterbox **384 px**; inferensi memakai ukuran yang sama lewat `rgbd_convnext.konfigurasi_utama`, lalu peta kelas dikembalikan ke resolusi RGB asli. | `segmentasi_convnext_depth.py` |
| RGB dan depth tidak tepat pasangan atau ekspor dianggap selalu 30 FPS | Paket ekspor selalu berisi `color_raw.png`, `depth_raw_z16.npy`, `depth_aligned_to_color.npy`, IR, timestamp, dan metadata dari frame RAW yang sama. Pemilihan waktu memakai timestamp kamera, bukan asumsi FPS tetap. | `<frame>/frame.json` dan `<frame>/depth_aligned_to_color.npy` |
| Rekomendasi tangga kasar | Poligon diambil langsung dari peta kelas **ConvNeXt RGB-D** (kedalaman sebagai masukan model), sudutnya diluruskan. SAM 2 tidak lagi dipakai: pada 52 frame terperiksa 211803 ia menurunkan Dice usulan dari 0,887 ke 0,841. | `segmentasi_convnext_depth.py`, `_rekomendasi_tangga_data()` |
| IR/depth tampak bergeser dari RGB padahal frame sama | Bukan salah pasang frame: **⏱ Cek sinkron** (tab Tinjau) membaca ulang rekaman mentah dan membuktikan RGB, depth, dan IR tiap frame ekspor satu jepretan (cap waktu sama dan isi berkas identik bit demi bit). Penyebabnya rolling shutter RGB: baris bawah tertinggal lebih jauh saat kamera bergerak. Lapisan IR dan depth dikoreksi **per baris** ke RGB. | `sinkron.py`, `ir_selaras.ukur_geser()` |
| Garis/pita tanpa IR atau warna bidang di sisi gambar | Dulu depth digeser dengan tepi 0 sebelum digambar. Kini tampilan dihitung dari depth apa adanya, lalu dikoreksi dengan tepi diisi piksel terdekat; lubang depth tidak lagi melubangi IR. Pengukuran tetap memakai tepi 0 (tanpa data). | `visual_depth.hitung_dari_folder()`, `ir_selaras.terapkan()` |
| Susah membedakan ujung anak tangga dari bayangan atau noda | Penajaman (unsharp/Laplace) menajamkan SEMUA perubahan terang-gelap, termasuk bayangan dan noda. Centang **📐 Garis lipatan 3-D**: garis dari bentuk depth, kuning = ujung/bibir tread, biru = pangkal riser. Bayangan dan noda tidak mengubah depth sehingga tidak pernah bergaris. Jenis garis hampir tidak pernah tertukar; letaknya median 2-3 px dari tepi label; tidak muncul bila depth kosong/jauh. | `visual_depth.garis_tepi()` |
| IR frame pertama rekaman dari saat lain | Frameset pertama kadang membawa IR basi (mis. 202 ms lebih awal). IR seperti itu tidak dipakai untuk tampilan dan tidak ditulis oleh ekspor baru; info frame memberi tanda ⚠. | `sinkron.ir_sinkron()`, `_ekspor_worker()` |
| Batu dan ramp belum punya mask setara tangga | Keduanya memakai usulan depth saja; belum ada bobot khusus. | `usulkan_segmentasi()` |
| Label hilang setelah berpindah frame | Editor menyimpan `label_draft.json` otomatis setelah perubahan berhenti. Label siap latih juga ditulis ke `label_yolo_seg.txt`; RAW tidak ikut diubah. | Folder paket frame ekspor |
| Editor berat saat banyak titik atau ketika zoom | Gambar RGB dasar dicache; hanya overlay mask yang digambar ulang. Drag/zoom dibatasi sekitar 30 FPS, dan zoom memakai render cepat dulu lalu kualitas tajam setelah roda mouse berhenti. | `KanvasLabel` di `studio_dataset_rgbd.py` |

### Checklist saat hasil rekomendasi tidak masuk akal

1. Pastikan `frame.json` menunjukkan kategori yang benar dan file RGB, depth, serta intrinsics tersedia dalam paket frame yang sama.
2. Pastikan bobot aktif ada pada salah satu jalur di `KANDIDAT_BOBOT` (`segmentasi_convnext_depth.py`). Usulan otomatis sengaja memakai bobot pra-latih publik saja, supaya label tidak melatih model yang membuat label itu sendiri.
3. Jangan mengubah urutan BGR/RGB atau ukuran 384 tanpa melatih ulang.
4. Buka overlay depth untuk memeriksa penyelarasan, lalu koreksi kandidat menggunakan opacity mask, magnet titik, dan lup. Simpan terjadi otomatis.
5. Untuk reproduksi, catat nama sesi, nama folder frame, commit aplikasi, bobot yang dipakai, dan apakah hasil berasal dari rekomendasi otomatis atau koreksi manual.
