#!/usr/bin/env bash
# Pasang Studio RGB-D di laptop baru dengan satu perintah:
#
#   git clone https://github.com/khaichi11/RGB-D-dataset-Labelling-APP.git paket_ubuntu_zenexo
#   cd paket_ubuntu_zenexo && ./pasang.sh
#
# Diuji untuk Linux x86_64 (Ubuntu 22.04/24.04). Aman dijalankan ulang: langkah
# yang sudah selesai dilewati. Token Hugging Face tidak pernah ditulis ke repo.
set -euo pipefail
AKAR="$(cd "$(dirname "$0")" && pwd)"
cd "$AKAR"
langkah() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
PY_VENV="$AKAR/kode/.venv/bin/python"

langkah "1/7 Paket sistem (python3-venv, python3-tk, git)"
if command -v apt-get >/dev/null 2>&1; then
  perlu=()
  for p in python3-venv python3-tk git; do dpkg -s "$p" >/dev/null 2>&1 || perlu+=("$p"); done
  if ((${#perlu[@]})); then sudo apt-get update -q && sudo apt-get install -y "${perlu[@]}"; else echo "  sudah lengkap"; fi
else
  echo "  bukan Ubuntu/Debian: pastikan Python 3.10-3.12 dengan Tkinter dan git terpasang"
fi
PY="${PYTHON:-python3}"
versi=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$versi" in 3.10|3.11|3.12) echo "  Python $versi" ;; *) echo "  PERINGATAN: Python $versi belum diuji (disarankan 3.10-3.12)";; esac

langkah "2/7 Repo Train-RGB-D-Model (model dan pustaka rgbd_convnext; repo privat)"
if [ -d Train-RGB-D-Model/.git ]; then
  echo "  sudah ada"
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo clone khaichi11/Train-RGB-D-Model
else
  echo "  Masuk GitHub bila diminta (username + Personal Access Token, bukan password)."
  git clone https://github.com/khaichi11/Train-RGB-D-Model.git
fi

langkah "3/7 Virtualenv kode/.venv dan pustaka Python (beberapa menit; torch ~2-3 GB)"
[ -x "$PY_VENV" ] || "$PY" -m venv kode/.venv
"$PY_VENV" -m pip install -q -U pip wheel
"$PY_VENV" -m pip install -r kode/requirements.txt

langkah "4/7 Penjaga rahasia Git (hook + email noreply)"
for r in . Train-RGB-D-Model; do
  git -C "$r" config core.hooksPath .githooks
  case "$(git -C "$r" config user.email || true)" in
    *users.noreply.github.com) ;;
    *) git -C "$r" config user.email "201188805+khaichi11@users.noreply.github.com" ;;
  esac
done
echo "  commit dari laptop ini tidak memuat email pribadi; token/email/.env ditolak hook"

langkah "5/7 Masuk Hugging Face (model terbaru + tombol push; Enter kosong = lewati)"
if "$PY_VENV" -c 'import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)'; then
  echo "  sudah masuk"
elif [ -t 0 ]; then
  echo "  Buka https://huggingface.co/settings/tokens/new?tokenType=write , buat token jenis Write,"
  echo "  lalu tempel di bawah. Token disimpan di ~/.cache/huggingface (di luar repo)."
  "$AKAR/kode/.venv/bin/hf" auth login || echo "  dilewati; bisa masuk nanti dari tombol Push di tab 7"
fi

langkah "6/7 Model pengusul label terbaru dari registri Hugging Face"
"$PY_VENV" - <<'PYEOF' || echo "  dilewati: Studio memakai model rujukan yang ada di repo Train"
import json, shutil
from pathlib import Path
from huggingface_hub import get_token, hf_hub_download
akar = Path.cwd()
if not get_token():
    raise SystemExit("  belum masuk Hugging Face")
try:
    reg = json.loads(Path(hf_hub_download("khaichi11/Skripsi", "registri.json")).read_text())
except Exception as e:                        # repo belum berisi registri, atau tanpa internet
    raise SystemExit(f"  registri model belum tersedia ({type(e).__name__})")
for jalur, m in reg.items():
    if "DIPAKAI STUDIO" in m.get("status", []):
        tujuan = akar / jalur
        if tujuan.exists():
            print(f"  sudah ada: {jalur}")
        else:
            tujuan.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download("khaichi11/Skripsi", f"model/{jalur}"), tujuan)
            print(f"  diunduh: {jalur}")
PYEOF

langkah "7/7 Ikon menu aplikasi dan uji cepat"
bash desktop/pasang-ikon.sh || true
"$PY_VENV" -c 'import cv2, numpy, PIL, torch, timm, pyrealsense2, huggingface_hub, pyarrow, tkinter
print(f"  torch {torch.__version__} | GPU CUDA: {torch.cuda.is_available()}")'
langkah "Selesai. Buka dari menu aplikasi 'Studio Dataset RGB-D', atau: desktop/jalankan-studio.sh"
