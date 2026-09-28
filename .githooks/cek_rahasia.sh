#!/usr/bin/env bash
# Tolak commit yang MENAMBAHKAN token/kunci API, kunci privat, email, atau berkas
# .env. Yang diperiksa hanya baris baru pada perubahan yang di-stage. Lolos
# paksa (bila yakin salah deteksi): git commit --no-verify
set -u
tolak=0
lapor() { printf '\n  [rahasia] %s\n' "$*" >&2; tolak=1; }

env_baru=$(git diff --cached --name-only --diff-filter=A | grep -E '(^|/)\.env($|\.)' | grep -v '\.env\.example$' || true)
[ -n "$env_baru" ] && lapor "berkas .env ikut di-stage: $env_baru"

TOKEN='hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|(^|[^A-Za-z0-9-])sk-(ant-|proj-)?[A-Za-z0-9_]{32,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-[A-Za-z0-9-]{10,}'
EMAIL='[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'
# Email yang memang boleh: alamat noreply dan contoh.
EMAIL_BOLEH='users\.noreply\.github\.com|noreply@|@example\.(com|org)'

tambah=$(git diff --cached -U0 --no-color --diff-filter=ACMR -- . ':(exclude)*.ipynb' | grep -E '^\+[^+]' || true)
t=$(printf '%s\n' "$tambah" | grep -oE "$TOKEN" | head -3 || true)
[ -n "$t" ] && lapor "pola token/kunci terdeteksi: $(printf '%s' "$t" | cut -c1-12 | tr '\n' ' ')…"
e=$(printf '%s\n' "$tambah" | grep -oE "$EMAIL" | grep -vE "$EMAIL_BOLEH" | sort -u | head -3 || true)
[ -n "$e" ] && lapor "alamat email terdeteksi: $(printf '%s' "$e" | tr '\n' ' ')"

email_git=$(git config user.email || true)
case "$email_git" in
  *users.noreply.github.com) ;;
  *) lapor "user.email Git ($email_git) bukan alamat noreply; email ini tercatat publik di setiap commit." \
           "Perbaiki: git config user.email 201188805+khaichi11@users.noreply.github.com" ;;
esac
[ "$tolak" -ne 0 ] && printf '\n  Commit dibatalkan. Hapus data di atas, atau git commit --no-verify bila salah deteksi.\n\n' >&2
exit "$tolak"
