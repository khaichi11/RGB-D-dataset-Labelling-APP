#!/usr/bin/env bash
# Peluncur Studio Dataset RGB-D.
#
# Dipakai oleh berkas .desktop. Ia menjalankan Studio dari virtualenv proyek,
# bukan dari python sistem: paket pyrealsense2/torch hanya terpasang di venv.
# Kalau gagal, pesannya ditampilkan lewat dialog supaya tetap terlihat walaupun
# diluncurkan dari ikon (tanpa terminal).
set -u
AKAR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$AKAR/kode/.venv/bin/python"
LOG="$AKAR/desktop/studio.log"

lapor() {
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --width=460 --title="Studio Dataset RGB-D" --text="$1" 2>/dev/null
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send -u critical "Studio Dataset RGB-D" "$1"
  fi
  echo "$1" >&2
}

if [ ! -x "$VENV" ]; then
  lapor "Virtualenv tidak ditemukan di:\n$VENV\n\nJalankan dulu: bash env/setup_ubuntu.sh"
  exit 1
fi

cd "$AKAR/kode" || exit 1
if ! "$VENV" -m studio_rgbd.studio_dataset_rgbd "$@" >"$LOG" 2>&1; then
  lapor "Studio berhenti dengan galat. Log:\n$LOG\n\n$(tail -n 12 "$LOG")"
  exit 1
fi
