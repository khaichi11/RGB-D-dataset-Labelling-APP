# BRIEF Studio RGB-D: peta tab dan letak masalah

Studio ada di repo ini (`studio/studio_rgbd`); model, pelacak, dan ukuran ada di repo
`model/aplikasi_tangga/rgbd_convnext` (repo Train-RGB-D-Model; peta kodenya:
`model/aplikasi_tangga/rgbd_convnext/BRIEF.md`). Riwayat masalah Studio yang
sudah dipecahkan ada di tabel "Catatan integrasi dan jejak masalah"
[PANDUAN_STUDIO_DATASET_RGBD.md](PANDUAN_STUDIO_DATASET_RGBD.md).

## Tab -> berkas

| tab | berkas utama | pendukung |
|---|---|---|
| 1. Rekam | `studio_dataset_rgbd.py` (`tab_rekam`) | `kamera_rgbd.py`, `resolusi_kamera.py`, `uji_depth_dinding.py` (tombol "Uji depth dinding") |
| 2. Tinjau & Ekspor | `studio_dataset_rgbd.py` (`tab_tinjau`; baris ekspor `ui_ekspor` di bawah rentang potong) | `catatan_rekaman.py`, `kelompok_tangga.py` (jendela 🪜 Banyak…, berisi `atribut_tangga.py`), `sinkron.py`, `ir_selaras.py`, `tempat_sampah.py`, `hitung_data.py`; tindakan jarang di menu `ui_bantu.tombol_menu` (⋯ Lainnya) |
| 3. Label & Ukur | `studio_dataset_rgbd.py` (`KanvasLabel`) | `segmentasi_convnext_depth.py` (usulan label), `visual_depth.py`, `ir_selaras.py`, `panel_ukuran.py`, `pemutar_ukur.py`, `ukuran_meteran.py`; sakelar tampilan di menu "⋯ Opsi", penggeser jarang di "Pengaturan lanjutan" |
| 4. Pilah & Split | `tab_split.py` (strip frame + pratinjau, pohon rekaman, kalkulator) | `kurasi.py` (Periksa ulang), `qa_label.py` (versi tanda), `jendela_versi.py` (Versi data) |
| 5. Uji Model | `uji_realtime.py` (sumber kamera) / `uji_berkas.py` (sumber berkas), dipilih `Studio.pilih_uji` | `daftar_model.py`, rgbd_convnext `paralel.py`/`sumber.py` |
| Hugging Face | `unggah_hf.py` (dataset) | model: `model/aplikasi_tangga/skrip/utama/unggah_model_hf.py` |

Pemberitahuan: `messagebox.showinfo/showwarning/showerror` jendela utama dialihkan ke pita di dalam jendela
(`ui_bantu.Kabar`, `pasang_kabar`, `Studio.kabar`; bisa membawa tombol "↶ Batalkan"). Popup ya/tidak hanya untuk
tindakan yang tidak bisa dibatalkan (hapus rekaman/ekspor, push HF, dsb.).

## Gejala -> letak

| gejala | lihat dulu |
|---|---|
| Uji Model: nomor/ukuran anak tangga beda dari video uji resmi (mis. #1 lalu #3) | `uji_realtime._spek`: rekaman dibaca dari frame 0 seperti `ukuran_sesi` dan `bukti/final_d435/video_uji/video_uji.py` (dulu `lewati` 20: pelacak mulai dari keadaan lain); model bawaan sama (`akhir_baru/deployment_penuh.pt`, FP16 + graf) |
| Uji Model: video tampak dipercepat / melompat, nomor anak tangga beda dari video uji | `uji_realtime.py` `_spek`/`_loop`: rekaman selalu dibaca tanpa membuang frame (`waktu_nyata` librealsense mati); centang "laju asli" menahan tiap frame menurut cap waktu rekaman (tertinggal = acuan digeser, tidak dikejar; sesudah jeda acuan diulang). Dulu mode "seperti kamera" membuang 22-36 % frame saat Studio tertinggal (27 Nov) |
| Uji Model: jeda, mundur/maju per frame, zoom, gambar terlalu kecil | `uji_realtime.py`: riwayat 900 frame terakhir (JPEG, `_riwayat`), `jeda`/`langkah`/`_lihat`/`_geser_riwayat`; saat jeda pekerja tertahan di `_loop` (pipeline paralel hanya tertahan antrean), ▶ di frame terbaru memproses satu frame (`_maju_satu`); `_gambar_ulang` mengepaskan frame ke panel (boleh diperbesar), roda tetikus zoom di kursor, seret geser, klik ganda pas layar; Spasi/←/→ sesudah gambar diklik |
| Tab Uji tidak menampilkan kamera / "device busy" | `uji_realtime.py` (pesan di kanvas), rgbd_convnext `sumber.alir_kamera` (coba ulang 5x); tutup program lain yang memakai D435 |
| Model yang dipakai salah / tidak muncul di daftar | `daftar_model.py`, `segmentasi_convnext_depth.pilih_bobot_usulan`; checkpoint `runs/` tidak masuk Git |
| Usulan label buruk | `segmentasi_convnext_depth.py`; checklist di PANDUAN |
| Depth berosilasi terhadap jarak (linearitas subpiksel) / A-factor rekaman | `resolusi_kamera.A_FACTOR_PRESET`, `terapkan_a_factor` (disetel sesudah preset di `kamera_rgbd._terapkan_a_factor`); nilai tercatat di `session.json` `pengaturan_kamera`; bawaan mati (A = 0 seperti data latih/uji), `STUDIO_A_FACTOR=1` menyalakan |
| Nomor/ukuran anak tangga di kartu Label | `panel_ukuran.py` -> rgbd_convnext `ukuran_sesi.py` ("Hitung nomor dari video") |
| Isian meteran | `ukuran_meteran.py`, `pemutar_ukur.py`; data dari aplikasi Ukur Tangga diimpor ke sini |
| Depth/jarak D435 diragukan (makin jauh makin meleset) | `uji_depth_dinding.py`: dinding datar vs meteran, hasil di `<akar data>/uji_depth_dinding/`; latar: `model/bukti/final_d435/galat_ukur_20261017/` |
| Uji depth dinding: tekan Ukur tidak terjadi apa-apa | `uji_depth_dinding.py`: isian "Jarak meteran (cm)" wajib diisi (kini bergaris oranye, langsung terfokus, terima "100 cm"); pratinjau warna + depth dengan kotak bagian yang diukur ada di jendela itu; frame dengan peringatan (depth sah < 50 %, titik tengah < 30 cm, atau miring > 10 derajat; miring 0 = lurus) tidak dihitung, gagal setelah 8 detik dengan pesan |
| Berapa derajat kamera menunduk saat merekam | `sudut_kamera.py` (bidang mendatar terbesar di bagian bawah depth; D435 tanpa IMU), dipanggil `Studio._perbarui_sudut` tiap 0,2 s: pita bawah preview RGB "Sudut tunduk N deg (OK/tundukkan/angkat; saran 25-35)"; ringkasan `sudut_kamera` di `session.json` saat rekaman selesai |
| Frame hilang dari daftar latih | tab 7 Tampilkan "dibuang" (manual = `frame_state.json` `di_sampah`, otomatis = `kurasi.json`), Del memulihkan; rekaman harus ✔ dan label terperiksa |
| Tab 7 / Buang terasa lambat | `tab_split.py`: pemindaian di thread latar (`muat_ulang`), strip hanya menggambar frame yang tampil, gambar kecil + frame tetangga dibaca `_pekerja`; Del hanya menulis satu `frame_state.json` dan memperbarui baris pohon rekaman itu (`_ubah`). Tab Label: `toggle_sampah_frame` menghapus baris tanpa memuat ulang daftar. Uji waktu: `uji_asap.py` ("waktu panah/Del") |
| Data apa yang melatih model mana; simpan/bandingkan versi dataset (v1, v2, ...) dan hasil CV-nya | `jendela_versi.py` (tab Split, "📦 Versi data…"); inti di model `rgbd_convnext/data/versi_data.py`, arsip `<akar data>/versi_data/`, hasil CV `model/bukti/final_d435/versi_data/hasil.json` |
| Membuang frame yang labelnya tidak cocok (saran model) | tab Pilah & Split: Tampilkan "model gagal"/"perlu dicek", Del per frame (tanpa buang massal: frame "model gagal" kebanyakan contoh sulit berlabel benar, pemeriksaan 23 Nov), K = label benar (`tandai_ok`, `qa_label.json` `ok`); saran dari `qa_label.json` kolom `saran` (`model/bukti/final_d435/cv_data6023_20261120/buat_perlu_dicek.py`) dan "perlu dilihat" `kurasi.json` (diabaikan bila label disunting sesudah kurasi) |
| Kurasi tidak membuang frame buram | sengaja sejak 20 Nov 2026: `kurasi.AturanKurasi.rasio_buram` = 0 (getar kamera pinggang = kondisi nyata); "perlu dilihat" kini juga dari CV 20 Nov (kedua model) dan depth hampir kosong |
| Tata letak terpotong | `uji_tata_letak.py`; galat tiap tab: `uji_asap.py` |
| Tulisan terlalu kecil/terpotong, kontrol sempit | `skala_ui.py` (angka piksel kode mengikuti `Xft.dpi`, dibaca sekali saat mulai; skala layar berubah -> tutup-buka Studio, `Studio._pantau_skala` memberi tahu) |
| Tukar merah/biru satu mask (T di atas mask, atau alat Tukar lalu klik); bilah kontrol di atas gambar | `KanvasLabel.mask_di`/`pindah_kelas`, `Studio.tukar_di_kursor`/`tukar_warna_mask`, `bilah_kelas`/`bilah_salin` di `ui_label` |
| Tandai rekaman sebagai data uji ("🧪 DATA TEST · <lokasi>" di daftar Rekaman; lokasi = tangga fisik rekaman, diisi dulu di kotak Tangga fisik dengan nama persis seperti sheet Ukur Tangga, mis. "Gedung F · Tangga 2") | `Studio.toggle_data_test`, `_teks_item_sesi`, `_warnai_item_sesi` (`studio_dataset_rgbd.py`); disimpan sebagai set `test` di `<akar data>/split_dataset.json` (sama dengan TEST di tab 7, `tab_split.py` membaca ulang berkas saat `muat_ulang`); latih `rgbd_convnext/latih/d435.py` membuang rekaman bertanda test dari daftar latih/validasi |
| Bayangan/salin label frame sebelum atau sesudahnya (B/P/N) | `salin_label.py` (homografi ORB), `Studio._salin_dari_tetangga`, `KanvasLabel.gambar_tampil`/`render`; salinan berstatus usulan sampai disunting/diperiksa |

## Rilis perubahan Studio

Cabang -> uji (`uji_asap.py`) -> `git merge --no-ff` ke `main` -> push -> tutup Studio
dengan wajar lalu buka lagi agar kode baru dimuat. Perubahan yang hanya dokumen tidak
perlu membuka ulang Studio.
