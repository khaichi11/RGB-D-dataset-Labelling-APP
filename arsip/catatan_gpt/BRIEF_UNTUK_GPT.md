# Brief status penelitian — persepsi tangga RGB-D untuk kaki prostetis

Khairuramdhani, NIM 235150300111035, Teknik Komputer, Universitas Brawijaya.
Disusun 4 September 2026, diperbarui 5 September 2026 dengan hasil optimasi.
Akar proyek: `~/paket_ubuntu_zenexo/`. Dua berkas pendamping:
`BRIEF_OPTIMASI_UNTUK_GPT.md` (optimasi inferensi) dan
`BRIEF_TEKNIS_UNTUK_GPT.md` (praproses, augmentasi, arsitektur, rugi, jalur) dan
`BRIEF_GEOMETRI_UNTUK_GPT.md` (perhitungan tinggi, dan perbedaan tangga lawan batu).

## 1. Ringkasan satu paragraf

Sistem mensegmentasi bidang tegak (riser) dan tapakan (tread) tangga naik dari
kamera Intel RealSense D435, lalu mengukur geometrinya dalam sentimeter.
Perubahan rancangan intinya: kedalaman dipindahkan dari pasca-proses menjadi
kanal masukan model, memakai dua cabang encoder dengan fusi bergerbang.
**Arsitektur akhir belum ditetapkan** — masih tahap pemilihan. Sampai 4
September enam kriteria evaluasi menunjuk empat konfigurasi berbeda. Optimasi
inferensi pada 5 September menghapus satu pertentangan itu: ConvNeXt Atto
ImageNet kini juga yang tercepat, sehingga tinggal tiga konfigurasi yang
bersaing.

## 2. Yang sudah selesai diukur

Tujuh kandidat sudah dipra-latih pada dataset publik lalu di-fine-tune pada
rekaman D435, dengan pembagian rekaman, skala kedalaman, anggaran pelatihan, dan
metrik yang identik. Rekaman `105348` ditahan utuh sebagai set uji.

| Model | Mparam | ms | Dice | riser | tread | F1 garis |
| --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt V2 Atto (ImageNet) | 7,56 | 10,78 | **0,9217** | 0,9522 | 0,8913 | 0,8195 |
| ↳ **sama, setelah optimasi inferensi** | 7,56 | **5,42** | **0,9217** | 0,9522 | 0,8913 | 0,8195 |
| YOLO11n-seg | 2,84 | 7,72 | 0,9006 | 0,9437 | 0,8576 | **0,8517** |
| MobileNetV4 Small (acak) | 5,76 | 7,93 | 0,8877 | 0,9423 | 0,8330 | 0,7971 |
| MobileNetV4 Small (ImageNet) | 5,76 | 7,64 | 0,8835 | 0,9427 | 0,8242 | 0,7587 |
| YOLO26n-seg | 2,69 | **5,11** | 0,8588 | 0,9255 | 0,7921 | 0,7540 |
| ConvNeXt V2 Atto (acak) | 4,91 | 10,34 | 0,8427 | 0,9475 | 0,7379 | 0,7515 |
| ConvNeXt V2 Femto (acak) | 6,85 | 11,55 | 0,8075 | 0,9230 | 0,6920 | 0,6721 |

Kestabilan temporal **membalik peringkat di atas**:

| Model | frame tanpa riser | tanpa tread | runtun hilang maks |
| --- | --- | --- | --- |
| ConvNeXt V2 Atto (acak) | **2,0%** | 1,0% | 1 frame |
| MobileNetV4 Small (acak) | 4,0% | 0,0% | 1 frame |
| ConvNeXt V2 Femto (acak) | 4,0% | 0,0% | 1 frame |
| ConvNeXt V2 Atto (ImageNet) | 16,8% | 16,8% | 8 frame |
| MobileNetV4 Small (ImageNet) | 21,8% | 14,9% | 15 frame |
| YOLO11n-seg | 24,8% | 31,7% | **32 frame** |
| YOLO26n-seg | 28,7% | 35,6% | **32 frame** |

Pada 15 fps, runtun 32 frame berarti lebih dari dua detik tanpa deteksi.

Baris beranak panah pada tabel pertama adalah bobot yang **sama persis**, hanya
dijalankan dengan FP16, channels_last, dan torch.compile. Ketepatannya tidak
berubah sedikit pun: dari 13,6 juta piksel yang diperiksa, hanya 1.239 (0,0091%)
berbeda dari keluaran FP32. Kandidat lain belum dioptimasi dengan cara yang sama,
jadi kolom latensi mereka masih apa adanya.

## 3. Empat temuan yang perlu didiskusikan

**(a) Sumbangan pra-latih ImageNet bergantung arsitektur.** Dua pasangan dengan
jumlah parameter identik, hanya beda inisialisasi encoder:

- ConvNeXt V2 Atto: ImageNet **+0,079** Dice
- MobileNetV4 Small: ImageNet **−0,004** Dice (dalam rentang keragaman antar-jalan)

**(b) Kedalaman berperan berlawanan arah antar-model.** Kanal kedalaman dinolkan
saat inferensi, tanpa melatih ulang:

| Model | Dice ada→nol | Lantai salah ada→nol |
| --- | --- | --- |
| ConvNeXt Atto ImageNet | 0,9217 → 0,7471 | 4,40% → 25,44% (depth **menekan** lantai) |
| ConvNeXt Atto acak | 0,8427 → 0,6907 | 89,99% → 75,51% (depth **menyebabkan** lantai) |
| MobileNetV4 acak | 0,8877 → 0,5793 | 50,17% → 30,23% (depth **menyebabkan** lantai) |

Tafsiran: lantai dan tapakan nyaris kembar pada kedalaman saja — sama-sama
bidang datar mendatar di ketinggian rendah. Encoder tanpa pengetahuan semantik
RGB bersandar pada isyarat kedataran itu. Konsekuensinya, menaikkan bobot
kedalaman **bukan** solusi umum.

**(c) Titik terburuk bukan gelap, tapi remang.** RGB diredupkan bertahap dengan
derau sensor dipertahankan, kedalaman utuh:

| Model | 100% | 25% | 10% | 3% | gelap total |
| --- | --- | --- | --- | --- | --- |
| ConvNeXt Atto ImageNet | 0,922 | 0,890 | **0,877** | 0,900 | 0,902 |
| MobileNetV4 acak | 0,888 | 0,866 | **0,860** | 0,871 | 0,869 |
| ConvNeXt Atto acak | 0,843 | 0,817 | **0,801** | 0,793 | 0,790 |

Kurvanya turun sampai 10% lalu naik lagi. Informasi rusak sebagian lebih
merugikan daripada informasi yang jelas hilang. **Usul yang belum dikerjakan:**
augmentasi peredupan RGB saat latih, sebagai pasangan `depth_dropout`.

**(d) Depth dropout menukar tiga besaran sekaligus.** ConvNeXt Atto ImageNet,
identik kecuali nilai dropout:

| depth dropout | Dice | lantai salah | Dice gelap total |
| --- | --- | --- | --- |
| 0,20 | **0,9217** | 4,40% | **0,902** |
| 0,05 | 0,9189 | **2,09%** | 0,872 |

Dropout melatih model menghadapi hilangnya satu modalitas, dan ketahanan itu
ternyata berlaku dua arah — yang terlatih bekerja tanpa depth juga lebih tahan
bekerja tanpa RGB.

## 4. Pertentangan yang belum terselesaikan

Keadaan **sebelum** optimasi, 4 September:

| Kriteria | Pemenang | Angka |
| --- | --- | --- |
| Ketepatan piksel | ConvNeXt Atto ImageNet | Dice 0,9217 |
| Kestabilan temporal | ConvNeXt Atto acak | 2,0% frame hilang |
| Ketepatan garis | YOLO11n-seg | F1 0,8517 |
| Latensi dan ukuran | YOLO26n-seg | 2,69 Mparam, 5,11 ms |
| Kesalahan lantai | ConvNeXt Atto ImageNet dd 0,05 | 2,09% |
| Ketahanan gelap | ConvNeXt Atto ImageNet dd 0,20 | Dice 0,902 |

Keadaan **sesudah** optimasi inferensi, 5 September. Bobotnya tidak diubah sama
sekali; hanya cara menjalankannya (FP16 + channels_last + torch.compile):

| Kriteria | Pemenang | Angka | Berubah? |
| --- | --- | --- | --- |
| Ketepatan piksel | ConvNeXt Atto ImageNet | Dice 0,9217 | tetap |
| **Latensi** | **ConvNeXt Atto ImageNet teroptimasi** | **7,2 ms, 138 FPS** | **berpindah dari YOLO26n** |
| Ukuran model | YOLO26n-seg | 2,69 Mparam | tetap |
| Kestabilan temporal | ConvNeXt Atto acak | 2,0% frame hilang | tetap |
| Ketepatan garis | YOLO11n-seg | F1 0,8517 | tetap |
| Kesalahan lantai | ConvNeXt Atto ImageNet dd 0,05 | 2,09% | tetap |
| Ketahanan gelap | ConvNeXt Atto ImageNet dd 0,20 | Dice 0,902 | tetap |

Alasan YOLO26n kehilangan keunggulan latensi: kandidat lain tidak pernah
dioptimasi dengan cara yang sama, sehingga perbandingan lama membandingkan
ConvNeXt tanpa kompilasi lawan YOLO dengan pipeline ultralytics yang sudah
matang. Setelah disetarakan, ConvNeXt Atto ImageNet menjadi yang tercepat dari
kedelapan kandidat pada video 734 frame: 7,2 ms lawan 8,9 ms milik YOLO26n.

Yang tersisa: ConvNeXt Atto ImageNet unggul pada ketepatan, latensi, kesalahan
lantai, dan ketahanan gelap; ConvNeXt Atto acak unggul pada kestabilan temporal;
YOLO11n unggul pada ketepatan garis; YOLO26n unggul hanya pada jumlah parameter.

Penetapan tetap memerlukan dua pengukuran yang **belum ada**: kinerja
sesungguhnya di Jetson Orin Nano (perangkat belum tersedia, dan `torch.compile`
bukan jalur Jetson — di sana TensorRT), dan pengaruh kehilangan kelas terhadap
kesinambungan indeks anak tangga saat menaiki tangga.

## 5. Data

**Primer** — rekaman sendiri, 376 frame berlabel bersih dari 7 rekaman:

| Rekaman | Peran | Frame | Catatan |
| --- | --- | --- | --- |
| `TANGGA_NAIK_20260830_105840` | latih | 106 | lantai luas di depan tangga |
| `TANGGA_NAIK_20260830_110530` | latih | 103 | pendakian penuh |
| `TANGGA_NAIK_20260902_013859` | latih | 40 | dalam ruang, pukul 01.38 |
| `TANGGA_NAIK_20260830_105802` | latih | 27 | mendekat dari jauh |
| `TANGGA_NAIK_20260830_105633` | latih | 22 | sudut menyamping |
| `TANGGA_NAIK_20260830_110152` | validasi | 26 | pemilihan checkpoint |
| `TANGGA_NAIK_20260830_105348` | **uji** | 52 | ditahan utuh |

**Sekunder** — StairNetV2 publik, 2.276 latih + 556 validasi. **30,3% citra
malam**; dataset primer tidak punya variasi ini sama sekali, jadi ketahanan
gelap seluruhnya berasal dari tahap pra-latih.

Skala kedalaman kedua sumber disatukan: `meter = 0,00789 × uint8 + 0,4661`
(sisa maks 6,2 cm). Konsekuensi: dataset publik terpotong di ~2,48 m, sedangkan
D435 mencapai 6,05 m.

## 6. Jalur berkas

### Data
```
dataset/studio_rgbd/rekaman/tangga_naik/TANGGA_NAIK_<tgl>_<waktu>/
  source/raw.db3                        rekaman mentah RealSense
  exports/frames/frame_<nnnnnn>/
    color_raw.png                       RGB 848x480
    depth_aligned_to_color.npy          Z16, kalikan depth_scale dari frame.json
    depth_raw.npy, ir_left_raw.png, ir_right_raw.png
    frame.json                          intrinsics, depth_scale, timestamp
    mask_objek.png                      label RISER
    mask_acuan.png                      label TAPAKAN
    label_draft.json                    poligon
    frame_state.json                    di_sampah, diperiksa_manual
dataset/paper_stairnetv2/RGB-D stair dataset/RGB-D stair dataset/
dataset/yolo_d435_rgbd/  dataset/yolo_publik_rgbd/
```

### Kode
```
kode/stair_fusion_atto/model_kandidat.py          model 2 cabang + fusi bergerbang
kode/stair_fusion_atto/encoder_timm.py            backbone dapat ditukar
kode/stair_fusion_atto/normalisasi.py             satu definisi skala depth
kode/stair_fusion_atto/finetune_d435/train.py     pelatihan fine-tune
kode/stair_fusion_atto/finetune_d435/dataset.py   pembacaan frame + augmentasi
kode/finetune_semua.sh                            rantai lengkap 7 kandidat
kode/banding_arsitektur.py                        Dice, IoU, F1 garis, latensi
kode/banding_stabilitas.py                        kestabilan temporal
kode/ukur_lantai.py                               positif-palsu lantai
kode/banding_geometri.py                          geometri bidang RANSAC
kode/banding_geometri_garis.py                    geometri garis 3-D
kode/ukur_latensi.py                              porsi latensi per bagian
kode/video_berdampingan.py                        video perbandingan
kode/lembar_label.py                              lembar kontak periksa label
kode/gambar_dataset.py                            gambar contoh dataset
kode/susun_proposal.py                            menyusun proposal dari CSV
kode/ukur_optimasi.py                             ketepatan + FPS per tingkat optimasi
kode/studio_rgbd/studio_dataset_rgbd.py           RGB-D Labelling Studio
kode/studio_rgbd/segmentasi_convnext_depth.py     usulan label otomatis
```

### Bobot dan hasil
```
bobot/kandidat/banding5kecil/<nama>/pra/best.pt   pra-latih publik
bobot/kandidat/banding4/<nama>/pra/best.pt        pra-latih publik
bobot/kandidat/final_d435/<nama>/best.pt          hasil fine-tune
bobot/kandidat/final_d435/<nama>/ft/weights/best.pt   hasil fine-tune YOLO
bobot/kandidat/final_d435/banding_final.csv       Tabel perbandingan 7 arsitektur
bobot/kandidat/final_d435/stabilitas_final.csv    Tabel kestabilan temporal
bobot/kandidat/final_d435/banding_dd05.csv        Tabel depth dropout
bobot/kandidat/final_d435/banding_optimasi.csv    Tabel tingkat optimasi 512 & 384
bobot/kandidat/final_d435/stabilitas_384.csv      kestabilan 512 lawan 384
bobot/kandidat/final_d435/verifikasi384.log       kestabilan + lantai varian 384
bobot/kandidat/final_d435/cnx_atto_in1k_384/      varian 384 (DIBATALKAN)
bobot/kandidat/final_d435/rantai.log              catatan pelatihan
bobot/kandidat/final_d435/penuh_105348.mp4        video uji 734 frame
bobot/kandidat/final_d435/penuh_110530.mp4        video latih 1236 frame
output/proposal_v2/gambar/                        gambar proposal
```

### Perintah
```bash
bash kode/finetune_semua.sh                    # latih 7 kandidat + banding
python kode/gambar_dataset.py                  # gambar contoh dataset
python kode/susun_proposal.py                  # susun ulang proposal
python -m studio_rgbd.studio_dataset_rgbd --preset jangan

# ukur ketepatan dan kecepatan per tingkat optimasi
python kode/ukur_optimasi.py --rekaman 105348 \
  --bobot "cnx_atto_in1k=bobot/kandidat/final_d435/cnx_atto_in1k/best.pt"

# pakai model teroptimasi pada video (penanda :opt)
python kode/video_berdampingan.py --bag <raw.db3> \
  --model "OPT=bobot/kandidat/final_d435/cnx_atto_in1k/best.pt:opt"
```

Argumen `--latih/--validasi/--uji` menerima potongan waktu nama rekaman,
misalnya `105348`. Kelas: `0 latar`, `1 bidang tegak`, `2 tapakan`.

## 7. Yang saya ragukan sendiri

1. **Set uji satu hari dengan data latih.** `105348` direkam 30 Agustus, sama
   seperti lima rekaman latih, kemungkinan tangga yang sama. Generalisasi ke
   tangga yang benar-benar baru belum teruji.
2. **Label dikoreksi di atas usulan otomatis**, bukan dari nol. Bias sistematis
   usulan awal mungkin bertahan pada bagian yang tampak sudah benar.
3. **Jetson Orin Nano belum pernah diukur.** Semua latensi dari RTX 4060.
4. **Panjang tapakan tidak andal pada jarak.** IQR 21–37 cm bahkan pada acuan.
5. **Sesi pelabelan 4 September mungkin memakai kriteria berbeda** — semua model
   turun sekitar 0,12 pada frame-frame itu; belum ditelusuri.
6. **Skala kedalaman publik** bersandar pada andaian sebaran jarak sebanding,
   bukan kalibrasi kamera yang memang tidak diterbitkan penerbitnya.
7. **Dice dan kestabilan temporal dapat sama-sama menyesatkan.** Varian yang
   dilatih pada resolusi 384 menang pada keduanya (Dice 0,9363 lawan 0,9217;
   frame tanpa riser 3,0% lawan 16,8%) tetapi salah menandai lantai sebagai
   tapakan pada 51,87% piksel lawan 4,40%. Varian itu saya batalkan. Kejadian ini
   membuat saya ragu apakah kriteria evaluasi saya sudah lengkap untuk kandidat
   lain yang belum diuji pada ketiga metrik sekaligus.

## 8. Konteks kolaborasi dan atribusi

Penelitian ini menjadi modul persepsi bagi sebuah sistem eksoskeleton. Rancangan
penempatan sensor dan unit pemroses diperoleh dari **dr. Achmad Zaini, Fakultas
Kedokteran, Universitas Brawijaya**, melalui paparan daring — bukan rancangan
saya sendiri. Saya belum memastikan siapa yang benar-benar membuat simulasinya,
sehingga di proposal saya tulis apa yang saya ketahui saja: bahwa rancangan itu
**diperoleh dari** beliau, bukan **dirancang oleh** beliau. Dikutip sebagai
komunikasi pribadi, yang menurut kaidah tidak masuk daftar referensi karena
memang tidak dapat ditelusuri pembaca.

Parameter penempatan dari simulasi tersebut (dipakai di Tabel 3.4 proposal):

| Parameter | Nilai |
| --- | --- |
| Medan pandang kedalaman | 87° × 58° |
| Medan pandang RGB | 69° × 42° |
| Sudut tunduk kamera | 30° |
| Tinggi kamera | 0,94 m |
| Jangkauan tanah terlihat | 0,56 – 2,92 m |
| Lebar maksimum terlihat | 5,69 m |
| Luas area terlihat | 8,86 m² |
| Ketidakpastian kedalaman | ± 28 mm |

Dua angka ini mengikat pekerjaan saya:

- **Jangkauan 0,56–2,92 m** hampir seluruhnya tercakup dataset publik yang
  terpotong di ~2,48 m. Pita 2,48–2,92 m hanya terwakili data primer.
- **Ketidakpastian ± 28 mm** terhadap riser 16–20 cm berarti 14–18% bila hanya
  satu piksel dipakai. Karena itu pengukuran memfit bidang/garis pada banyak
  piksel, bukan satu titik.

Gambar terkait: `output/proposal_v2/gambar/exo_penempatan.png` dan
`output/proposal_v2/gambar/kamera_d435.jpg`.

## 9. Pertanyaan terbuka untuk didiskusikan

1. Bagaimana membobot kestabilan temporal terhadap ketepatan piksel untuk kaki
   prostetis? Apakah ada dasar dari literatur kendali prostetis?
2. Apakah augmentasi peredupan RGB masuk akal, dan bagaimana merancangnya agar
   tidak sekadar menghafal derau buatan?
3. Model dengan Dice tinggi tapi kestabilan buruk — apakah pelacakan temporal di
   hilir (`PelacakGaris`) dapat menutupi kekurangan itu, sehingga Dice tetap
   jadi kriteria utama?
4. Apakah wajar melaporkan bahwa arsitektur belum ditetapkan pada proposal, atau
   pembimbing biasanya mengharapkan satu pilihan sejak awal?
5. Bagaimana sebaiknya menuliskan atribusi rancangan eksoskeleton yang saya
   terima dari dr. Achmad Zaini, mengingat saya tidak tahu siapa pembuat aslinya
   dan tautannya tidak dapat dibagikan?

## 10. Pembaruan eksperimen decoder U-Net, 8 September 2026

Eksperimen tambahan membandingkan decoder FPN yang dipakai kandidat ConvNeXt V2
Atto dengan decoder U-Net ringan. Kedua model memakai encoder RGB dan depth,
fusi bergerbang, kepala garis, praproses, fungsi rugi, serta data *fine-tuning*
D435 yang sama: 298 frame dari rekaman 013859, 105633, 105802, 105840, dan
110530. Rekaman 110152 (26 frame) ditahan dari pelatihan dan dipakai untuk
memilih checkpoint.

Pada rekaman validasi tersebut, U-Net terbaik di epoch 15 memberi mean Dice
0,9002 dan Boundary F1 0,9326; FPN terbaik di epoch 22 memberi 0,8824 dan
0,8163. Angka ini belum cukup untuk memilih U-Net karena set 110152 dipakai
untuk pemilihan checkpoint, tahap pra-latih publik kedua model belum setara
sepenuhnya, dan inspeksi visual lokal memperlihatkan FPN kadang menghasilkan
bidang yang tampak lebih utuh. FPN tetap kandidat utama sementara; U-Net adalah
pembanding decoder yang harus diuji pada set akhir dan Jetson. Kode, checkpoint,
contoh gambar, serta catatan tersimpan di `artefak_unet/` dan `unet-new/`; video
RAW tetap lokal karena dapat memuat lingkungan pengambilan data.
