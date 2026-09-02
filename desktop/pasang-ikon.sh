#!/usr/bin/env bash
# Pasang ikon Studio ke menu aplikasi Ubuntu (per pengguna, tanpa sudo).
set -eu
AKAR="$(cd "$(dirname "$0")/.." && pwd)"
TUJUAN="$HOME/.local/share/applications/studio-rgbd.desktop"
mkdir -p "$(dirname "$TUJUAN")"
sed "s|/home/khai/paket_ubuntu_zenexo|$AKAR|g" "$AKAR/desktop/studio-rgbd.desktop" > "$TUJUAN"
chmod +x "$TUJUAN" "$AKAR/desktop/jalankan-studio.sh"
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
echo "Terpasang: $TUJUAN"
echo "Cari 'Studio Dataset RGB-D' di menu aplikasi. Untuk menaruh di dock, klik kanan ikonnya."
