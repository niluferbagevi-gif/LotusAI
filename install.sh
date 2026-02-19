#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  LotusAI - Otomatik Kurulum Scripti
#  Versiyon : 2.5.6
#  Platform  : Ubuntu (22.04 / 24.04) - Root kullanıcı
#  GitHub    : https://github.com/niluferbagevi-gif/LotusAI
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Renkler ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Değişkenler ──────────────────────────────────────────────────────────────
REPO_URL="https://github.com/niluferbagevi-gif/LotusAI.git"
INSTALL_DIR="/opt/LotusAI"
CONDA_DIR="/opt/miniconda3"
CONDA_ENV="lotus-ai"
MINICONDA_SH="/tmp/miniconda.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
NODE_VERSION="20"

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────
log()     { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[HATA]${NC}  $*"; exit 1; }

banner() {
  echo -e "${CYAN}"
  echo "══════════════════════════════════════════════════════════════"
  echo "   🌿  LotusAI Kurulum Scripti v2.5.6"
  echo "══════════════════════════════════════════════════════════════"
  echo -e "${NC}"
}

# ─── Kontroller ───────────────────────────────────────────────────────────────
check_root() {
  if [[ $EUID -ne 0 ]]; then
    error "Bu script root yetkisiyle çalıştırılmalıdır.\nKullanım: sudo bash install.sh"
  fi
  success "Root yetkisi doğrulandı."
}

check_ubuntu() {
  if ! grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
    warn "Ubuntu dışı bir sistem tespit edildi. Devam ediliyor..."
  else
    UBUNTU_VER=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
    success "Ubuntu $UBUNTU_VER algılandı."
  fi
}

check_internet() {
  log "İnternet bağlantısı kontrol ediliyor..."
  if ! curl -s --connect-timeout 5 https://google.com > /dev/null; then
    error "İnternet bağlantısı yok. Lütfen bağlantınızı kontrol edin."
  fi
  success "İnternet bağlantısı mevcut."
}

# ─── 1. Sistem Güncelleme ─────────────────────────────────────────────────────
install_system_packages() {
  log "Sistem paketleri güncelleniyor..."
  apt-get update -qq

  log "Temel sistem kütüphaneleri kuruluyor..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libboost-all-dev \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    libreadline-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    portaudio19-dev \
    libportaudio2 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgtk-3-dev \
    v4l-utils \
    espeak-ng \
    libespeak-ng1 \
    python3-tk \
    xvfb \
    2>/dev/null

  success "Sistem paketleri kuruldu."
}

# ─── 2. Node.js & Claude Code ─────────────────────────────────────────────────
install_nodejs() {
  if command -v node &>/dev/null; then
    success "Node.js zaten kurulu: $(node --version)"
    return
  fi
  log "Node.js $NODE_VERSION kuruluyor..."
  curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - 2>/dev/null
  apt-get install -y nodejs 2>/dev/null
  success "Node.js kuruldu: $(node --version)"
}

install_claude_code() {
  if command -v claude &>/dev/null; then
    success "Claude Code zaten kurulu: $(claude --version 2>/dev/null || echo 'kurulu')"
    return
  fi
  log "Claude Code (claude-code) kuruluyor..."
  npm install -g @anthropic-ai/claude-code 2>/dev/null || \
    warn "Claude Code kurulumu başarısız oldu, atlanıyor."
  if command -v claude &>/dev/null; then
    success "Claude Code kuruldu."
  else
    warn "Claude Code PATH'e eklenemedi. Sonradan manuel kurabilirsiniz: npm install -g @anthropic-ai/claude-code"
  fi
}

# ─── 3. Miniconda ─────────────────────────────────────────────────────────────
install_miniconda() {
  if [[ -f "$CONDA_DIR/bin/conda" ]]; then
    success "Miniconda zaten kurulu: $CONDA_DIR"
    return
  fi
  log "Miniconda indiriliyor..."
  wget -q "$MINICONDA_URL" -O "$MINICONDA_SH"
  log "Miniconda kuruluyor: $CONDA_DIR"
  bash "$MINICONDA_SH" -b -p "$CONDA_DIR"
  rm -f "$MINICONDA_SH"

  # Tüm kullanıcılar için PATH
  cat > /etc/profile.d/miniconda.sh << 'EOF'
export PATH="/opt/miniconda3/bin:$PATH"
EOF
  chmod +x /etc/profile.d/miniconda.sh
  export PATH="$CONDA_DIR/bin:$PATH"

  "$CONDA_DIR/bin/conda" init bash 2>/dev/null || true
  "$CONDA_DIR/bin/conda" config --set auto_activate_base false 2>/dev/null || true
  success "Miniconda kuruldu: $CONDA_DIR"
}

# ─── 4. Ollama ────────────────────────────────────────────────────────────────
install_ollama() {
  if command -v ollama &>/dev/null; then
    success "Ollama zaten kurulu."
    return
  fi
  log "Ollama kuruluyor..."
  curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null
  if command -v ollama &>/dev/null; then
    success "Ollama kuruldu."
    # Servis olarak başlat
    if systemctl is-available ollama &>/dev/null 2>&1; then
      systemctl enable ollama 2>/dev/null || true
      systemctl start ollama 2>/dev/null || true
    fi
  else
    warn "Ollama kurulumu tamamlanamadı, atlanıyor."
  fi
}

# ─── 5. Projeyi Klonla ────────────────────────────────────────────────────────
clone_project() {
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Proje zaten mevcut, güncelleniyor: $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull origin main 2>/dev/null || \
    git -C "$INSTALL_DIR" pull origin master 2>/dev/null || \
      warn "Git pull başarısız, mevcut sürüm kullanılacak."
    success "Proje güncellendi: $INSTALL_DIR"
    return
  fi
  log "LotusAI GitHub'dan indiriliyor..."
  git clone "$REPO_URL" "$INSTALL_DIR"
  success "Proje klonlandı: $INSTALL_DIR"
}

# ─── 6. Conda Ortamı ──────────────────────────────────────────────────────────
setup_conda_env() {
  export PATH="$CONDA_DIR/bin:$PATH"

  if conda env list 2>/dev/null | grep -q "^${CONDA_ENV}"; then
    log "Conda ortamı zaten mevcut, güncelleniyor: $CONDA_ENV"
    conda env update -n "$CONDA_ENV" -f "$INSTALL_DIR/environment.yml" --prune 2>/dev/null && \
      success "Conda ortamı güncellendi." || warn "Conda güncelleme kısmi başarısız oldu."
    return
  fi

  log "Conda ortamı oluşturuluyor: $CONDA_ENV (Bu işlem uzun sürebilir...)"
  conda env create -f "$INSTALL_DIR/environment.yml" 2>/dev/null && \
    success "Conda ortamı oluşturuldu: $CONDA_ENV" || {
      warn "environment.yml'den kurulum kısmi hatalı tamamlandı."
      warn "Hataları görmek için: conda env create -f $INSTALL_DIR/environment.yml"
    }
}

# ─── 7. .env Dosyası ──────────────────────────────────────────────────────────
setup_env_file() {
  if [[ -f "$INSTALL_DIR/.env" ]]; then
    warn ".env dosyası zaten mevcut, atlanıyor: $INSTALL_DIR/.env"
    return
  fi
  log ".env şablon dosyası oluşturuluyor..."
  cat > "$INSTALL_DIR/.env" << 'EOF'
# ═══════════════════════════════════════════════════════════════
# LotusAI Ortam Değişkenleri
# Bu dosyayı kendi API anahtarlarınızla doldurun.
# ═══════════════════════════════════════════════════════════════

# Google Gemini API
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# GPU Kullanımı (True/False)
USE_GPU=True

# Meta / WhatsApp Business API
META_ACCESS_TOKEN=YOUR_META_ACCESS_TOKEN_HERE
WHATSAPP_PHONE_ID=YOUR_WHATSAPP_PHONE_ID_HERE

# Instagram
INSTAGRAM_ACCESS_TOKEN=YOUR_INSTAGRAM_TOKEN_HERE

# Uygulama Ayarları
FLASK_SECRET_KEY=change_this_to_a_random_secret_key
DEBUG=False
HOST=0.0.0.0
PORT=5000

# Ses Ayarları
VOICE_ENABLED=True
USE_XTTS=False

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
EOF
  success ".env dosyası oluşturuldu: $INSTALL_DIR/.env"
  warn "Lütfen $INSTALL_DIR/.env dosyasını API anahtarlarınızla düzenleyin."
}

# ─── 8. Başlatma Scripti ──────────────────────────────────────────────────────
create_launcher() {
  cat > /usr/local/bin/lotusai << EOF
#!/bin/bash
# LotusAI Başlatıcı
export PATH="$CONDA_DIR/bin:\$PATH"
cd "$INSTALL_DIR"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
python main.py "\$@"
EOF
  chmod +x /usr/local/bin/lotusai
  success "Başlatıcı oluşturuldu: /usr/local/bin/lotusai"
}

# ─── Özet ─────────────────────────────────────────────────────────────────────
print_summary() {
  echo ""
  echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
  echo -e "${GREEN}${BOLD}   LotusAI kurulumu tamamlandı!${NC}"
  echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
  echo ""
  echo -e "  ${BOLD}Proje dizini   :${NC} $INSTALL_DIR"
  echo -e "  ${BOLD}Conda ortamı   :${NC} $CONDA_ENV"
  echo -e "  ${BOLD}Conda dizini   :${NC} $CONDA_DIR"
  echo ""
  echo -e "  ${YELLOW}Sonraki adımlar:${NC}"
  echo -e "  1) API anahtarlarınızı girin:"
  echo -e "     ${CYAN}nano $INSTALL_DIR/.env${NC}"
  echo ""
  echo -e "  2) Conda ortamını aktive edin:"
  echo -e "     ${CYAN}source /opt/miniconda3/bin/activate && conda activate lotus-ai${NC}"
  echo ""
  echo -e "  3) Sistemi başlatın (kısa yol):"
  echo -e "     ${CYAN}lotusai${NC}"
  echo ""
  echo -e "  4) Veya manuel olarak:"
  echo -e "     ${CYAN}cd $INSTALL_DIR && conda activate lotus-ai && python main.py${NC}"
  echo ""
  echo -e "  5) Web arayüzü: ${CYAN}http://localhost:5000${NC}"
  echo ""
  if command -v ollama &>/dev/null; then
    echo -e "  ${BOLD}Ollama komutları:${NC}"
    echo -e "  - Model indir : ${CYAN}ollama pull llama3${NC}"
    echo -e "  - Model listesi: ${CYAN}ollama list${NC}"
    echo ""
  fi
  echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
}

# ─── ANA AKIŞ ─────────────────────────────────────────────────────────────────
main() {
  banner
  check_root
  check_ubuntu
  check_internet

  echo ""
  log "Kurulum başlıyor..."
  echo ""

  # Adım 1: Sistem paketleri
  echo -e "${BOLD}[1/7] Sistem paketleri kuruluyor...${NC}"
  install_system_packages

  # Adım 2: Node.js
  echo -e "${BOLD}[2/7] Node.js kuruluyor...${NC}"
  install_nodejs

  # Adım 3: Claude Code
  echo -e "${BOLD}[3/7] Claude Code kuruluyor...${NC}"
  install_claude_code

  # Adım 4: Miniconda
  echo -e "${BOLD}[4/7] Miniconda kuruluyor...${NC}"
  install_miniconda

  # Adım 5: Ollama
  echo -e "${BOLD}[5/7] Ollama kuruluyor...${NC}"
  install_ollama

  # Adım 6: Proje klonla
  echo -e "${BOLD}[6/7] LotusAI projesi indiriliyor...${NC}"
  clone_project

  # Adım 7: Conda ortamı + .env + başlatıcı
  echo -e "${BOLD}[7/7] Python ortamı ve yapılandırma ayarlanıyor...${NC}"
  setup_conda_env
  setup_env_file
  create_launcher

  print_summary
}

main "$@"
