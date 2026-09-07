# RGB-D Labelling Studio

Aplikasi pengambilan dan pelabelan data RGB-D untuk penelitian persepsi tangga
pada kaki prostetis. Repositori ini memuat aplikasi Studio; kode inferensi dan
eksperimen tetap lokal.

## Menjalankan

```bash
cd ~/paket_ubuntu_zenexo/kode
source .venv/bin/activate
python -m studio_rgbd.studio_dataset_rgbd --preset jangan
```

Exposure manual untuk mengurangi blur saat kamera dibawa berjalan:

```bash
python -m studio_rgbd.studio_dataset_rgbd --preset akurasi --exposure 6000
```

Satuan `--exposure` adalah mikrodetik; 0 berarti auto. Depth tidak terpengaruh
karena D435 memakai jalur IR terpisah.

## Point cloud

```bash
python -m studio_rgbd.ekspor_pointcloud \
  --frame ../dataset/studio_rgbd/rekaman/tangga_naik/SESI/exports/frames/frame_000450 \
  --out ../hasil/aktif/pointcloud/frame_000450.ply
```

PLY biner, koordinat meter, terbaca di CloudCompare dan MeshLab.

## Pelatihan model kandidat

Pra-latih pada dataset publik lalu fine-tune pada rekaman D435:

```bash
bash finetune_semua.sh
```

Skrip ini melatih tujuh kandidat (ConvNeXt V2 Atto/Femto, MobileNetV4 Small,
YOLO11n-seg, YOLO26n-seg) pada pembagian rekaman yang sama, lalu menjalankan
perbandingan dan uji kestabilan. Hasil tersimpan di
`../bobot/kandidat/final_d435/`.

Fine-tune satu kandidat saja:

```bash
python -m stair_fusion_atto.finetune_d435.train \
  --init ../bobot/kandidat/banding5kecil/cnx_atto_in1k/pra/best.pt \
  --output ../bobot/kandidat/final_d435/cnx_atto_in1k \
  --latih 013859 105633 105802 105840 110530 --validasi 110152 --bagi \
  --epochs 60 --patience 15 --bg-weight 2.0 --tread-weight 1.6 --depth-dropout 0.2
```

## Pengukuran

| Skrip | Fungsi |
| --- | --- |
| `banding_arsitektur.py` | Dice, IoU, F1 garis, latensi per kandidat |
| `banding_stabilitas.py` | frame yang kehilangan kelas dan getar antar-frame |
| `ukur_lantai.py` | positif-palsu lantai pada fase mendekat |
| `banding_geometri.py` | tinggi riser dan panjang tapakan dari bidang RANSAC |
| `banding_geometri_garis.py` | geometri dari garis 3-D dengan pelacak |
| `ukur_latensi.py` | porsi latensi tiap bagian model |
| `video_berdampingan.py` | video perbandingan semua kandidat dalam satu frame |
| `lembar_label.py` | lembar kontak untuk memeriksa label |
| `gambar_dataset.py` | gambar contoh dataset primer dan sekunder |
| `susun_proposal.py` | menyusun dokumen proposal dari CSV hasil |

Kelas: `0 latar`, `1 bidang tegak`, `2 tapakan`.
