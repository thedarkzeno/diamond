#!/bin/bash
# Setup DIAMOND environment using uv in WSL
set -e

PROJECT_DIR="/mnt/c/Users/Adalberto/Documents/code/diamond"
PYTHON_VERSION="3.10"
CUDA_VERSION="cu121"  # Change to cu118 if your CUDA is 11.8

cd "$PROJECT_DIR"

# --- Fix WSL DNS (common cause of package download failures) ---
fix_wsl_dns() {
    echo "==> Fixing WSL DNS (requires sudo)..."
    if ! grep -q "generateResolvConf = false" /etc/wsl.conf 2>/dev/null; then
        echo -e "[network]\ngenerateResolvConf = false" | sudo tee -a /etc/wsl.conf > /dev/null
    fi
    echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" | sudo tee /etc/resolv.conf > /dev/null
    echo "    DNS set to 8.8.8.8 and 1.1.1.1"
}

# --- Fix Cloudflare WARP DNS hijack for files.pythonhosted.org ---
fix_pythonhosted_dns() {
    # WARP routes files.pythonhosted.org to a CGNAT IP (100.x.x.x) that breaks TLS.
    # Pin to the real Fastly IP via /etc/hosts to bypass it.
    local REAL_IP="167.82.48.223"
    local HOST="files.pythonhosted.org"
    if ! grep -q "$HOST" /etc/hosts; then
        echo "$REAL_IP $HOST" | sudo tee -a /etc/hosts > /dev/null
        echo "    Pinned $HOST -> $REAL_IP in /etc/hosts"
    fi
}

echo "==> Checking network connectivity..."
if ! curl -sf --max-time 5 https://pypi.org > /dev/null 2>&1; then
    echo "    pypi.org unreachable — attempting DNS fix..."
    fix_wsl_dns
    if ! curl -sf --max-time 10 https://pypi.org > /dev/null 2>&1; then
        echo "ERROR: Still cannot reach pypi.org after DNS fix."
        echo "       Check your Windows firewall or VPN settings."
        exit 1
    fi
    echo "    Network OK after DNS fix."
else
    echo "    Network OK."
fi

echo "==> Checking files.pythonhosted.org TLS..."
if ! curl -sf --max-time 10 https://files.pythonhosted.org > /dev/null 2>&1; then
    echo "    TLS failing — pinning real IP in /etc/hosts (requires sudo)..."
    fix_pythonhosted_dns
    if ! curl -sf --max-time 10 https://files.pythonhosted.org > /dev/null 2>&1; then
        echo "ERROR: Still cannot reach files.pythonhosted.org."
        echo "       Try pausing your VPN/WARP and running setup again."
        exit 1
    fi
    echo "    TLS OK after DNS pin."
else
    echo "    TLS OK."
fi

echo "==> Checking uv installation..."
if ! command -v uv &> /dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi
echo "uv version: $(uv --version)"

echo "==> Creating virtual environment with Python $PYTHON_VERSION..."
uv venv .venv --python "$PYTHON_VERSION" --clear
source .venv/bin/activate

# uv uses UV_HTTP_TIMEOUT (seconds) for download timeouts
export UV_HTTP_TIMEOUT=120

echo "==> Installing PyTorch $CUDA_VERSION wheels..."
uv pip install \
    torch==2.4.1 \
    torchvision \
    --index-url "https://download.pytorch.org/whl/$CUDA_VERSION"

echo "==> Installing project dependencies..."
uv pip install -r requirements.txt \
    --extra-index-url "https://download.pytorch.org/whl/$CUDA_VERSION" \
    --index-strategy unsafe-best-match

echo "==> Downloading Atari ROMs..."
uv pip install autorom
AutoROM --accept-license

echo ""
echo "==> Setup complete!"
echo "    To activate the environment: source $PROJECT_DIR/.venv/bin/activate"
echo "    To start training:           bash $PROJECT_DIR/train.sh"
echo "    To verify GPU:               python -c 'import torch; print(torch.cuda.is_available())'"
