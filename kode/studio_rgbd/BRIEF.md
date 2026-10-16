# BRIEF Studio RGB-D: peta tab dan letak masalah

Studio ada di repo ini (`kode/studio_rgbd`); model, pelacak, dan ukuran ada di repo
`Train-RGB-D-Model/aplikasi_tangga/rgbd_convnext` (peta kodenya:
`Train-RGB-D-Model/aplikasi_tangga/rgbd_convnext/BRIEF.md`). Riwayat masalah Studio yang
sudah dipecahkan ada di tabel "Catatan integrasi dan jejak masalah"
[PANDUAN_STUDIO_DATASET_RGBD.md](PANDUAN_STUDIO_DATASET_RGBD.md).

## Tab -> berkas

| tab | berkas utama | pendukung |
|---|---|---|
| 1. Rekam | `studio_dataset_rgbd.py` (`tab_rekam`) | `kamera_rgbd.py`, `resolusi_kamera.py` |
| 2. Tinjau & Potong | `studio_dataset_rgbd.py` (`tab_tinjau`) | `catatan_rekaman.py`, `kelompok_tangga.py`, `atribut_tangga.py`, `sinkron.py`, `tempat_sampah.py`, `hitung_data.py` |
| 3. Ekspor Frame | `studio_dataset_rgbd.py` (`_ekspor_worker`) | `sinkron.py`, `ir_selaras.py` |
| 4. Label & Ukur | `studio_dataset_rgbd.py` (`KanvasLabel`) | `segmentasi_convnext_depth.py` (usulan label), `visual_depth.py`, `ir_selaras.py`, `qa_label.py`, `panel_ukuran.py`, `pemutar_ukur.py`, `ukuran_meteran.py` |
| 5. Uji Sistem Tangga | `uji_realtime.py` | `daftar_model.py` (pilihan model), rgbd_convnext `paralel.py`/`sumber.py` |
| 6. Uji Model pada Berkas | `uji_berkas.py` | `daftar_model.py` |
| 7. Split Dataset | `tab_split.py` | `kurasi.py`, `jendela_kurasi.py`, `atribut_tangga.py` |
| Hugging Face | `unggah_hf.py` (dataset) | model: `Train-RGB-D-Model/aplikasi_tangga/skrip/utama/unggah_model_hf.py` |

## Gejala -> letak

| gejala | lihat dulu |
|---|---|
| Tab Uji tidak menampilkan kamera / "device busy" | `uji_realtime.py` (pesan di kanvas), rgbd_convnext `sumber.alir_kamera` (coba ulang 5x); tutup program lain yang memakai D435 |
| Model yang dipakai salah / tidak muncul di daftar | `daftar_model.py`, `segmentasi_convnext_depth.pilih_bobot_usulan`; checkpoint `runs/` tidak masuk Git |
| Usulan label buruk | `segmentasi_convnext_depth.py`; checklist di PANDUAN |
| Nomor/ukuran anak tangga di kartu Label | `panel_ukuran.py` -> rgbd_convnext `ukuran_sesi.py` ("Hitung nomor dari video") |
| Isian meteran | `ukuran_meteran.py`, `pemutar_ukur.py`; data dari aplikasi Ukur Tangga diimpor ke sini |
| Frame hilang dari daftar latih | `tab_split.py`, `kurasi.py` (`kurasi.json`), rekaman harus ✔ dan label terperiksa |
| Tata letak terpotong | `uji_tata_letak.py`; galat tiap tab: `uji_asap.py` |

## Rilis perubahan Studio

Cabang -> uji (`uji_asap.py`) -> `git merge --no-ff` ke `main` -> push -> tutup Studio
dengan wajar lalu buka lagi agar kode baru dimuat. Perubahan yang hanya dokumen tidak
perlu membuka ulang Studio.
