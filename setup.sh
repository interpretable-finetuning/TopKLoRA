#!/bin/bash
# Runpod Environment Setup Script
# For LLM development and interpretability work

set -euo pipefail  # Exit on error, undefined variables, and pipeline failures

# Use sudo only when not already root (RunPod containers often run as root without sudo installed)
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "=== Updating package lists ==="
$SUDO apt-get update

echo "=== Installing core tools ==="
$SUDO apt-get install -y \
    tmux \
    vim \
    htop \
    nvtop \
    git-lfs \
    jq \
    ripgrep \
    ncdu \
    tree \
    fd-find \
    bat \
    curl \
    wget \
    build-essential

# Initialize git-lfs
git lfs install

echo "=== Setting vim as git editor ==="
git config --global core.editor vim

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source uv for current session
export PATH="$HOME/.local/bin:$PATH"

echo "=== Setting up tmux config with mouse support ==="
cat > ~/.tmux.conf << 'EOF'
# Enable mouse support
set -g mouse on

# Better scrollback
set -g history-limit 50000

# Start windows and panes at 1, not 0
set -g base-index 1
setw -g pane-base-index 1

# Easier split commands
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# Reload config with r
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# Better colors
set -g default-terminal "screen-256color"

# Reduce escape time (better for vim)
set -sg escape-time 10

# Enable focus events
set-option -g focus-events on
EOF

echo "=== Creating profile file ==="
# This profile can live in persistent storage and be symlinked
cat > ~/.runpod_profile.sh << 'EOF'
# Runpod Profile - sourced from ~/.bashrc
# Move this file to persistent storage and symlink it

# Load environment variables from /proc/1/environ (RunPod injects vars there)
if [ -r /proc/1/environ ]; then
    while IFS= read -r -d '' line; do
        if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            varname="${line%%=*}"
            # Skip PATH - managed below
            if [ "$varname" != "PATH" ]; then
                export "$line"
            fi
        fi
    done < /proc/1/environ
    unset line varname
fi

# Add uv to PATH (only once)
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# Default editor (git, crontab, etc.)
export EDITOR=vim
export VISUAL=vim

# Git aliases
alias gits='git status'
alias gitb='git branch'
alias gitl='git log --graph --oneline'
alias gd='git diff'

# System aliases
alias ll='ls -lah'
alias nv='nvtop'
alias top='htop'
alias gpus='nvidia-smi'
alias usage='ncdu'

# fd-find is installed as fdfind on Debian/Ubuntu
alias fd='fdfind'

# bat is installed as batcat on Debian/Ubuntu
alias cat='batcat --paging=never'
alias bat='batcat'

# Quick navigation
alias ..='cd ..'
alias ...='cd ../..'

# Greeting (interactive shells only)
runpod_greeting() {
    echo "Runpod environment ready"
    echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"
    echo "uv: $(uv --version 2>/dev/null || echo 'not found')"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
}

if [[ $- == *i* ]]; then
    runpod_greeting
fi
EOF

echo "=== Setting up bash profile ==="
cat >> ~/.bashrc << 'EOF'

# Runpod environment
if [ -f ~/.runpod_profile.sh ]; then
    . ~/.runpod_profile.sh
fi
EOF

echo "=== Setting up vim config ==="
cat > ~/.vimrc << 'EOF'
" Basic settings
set number              " Line numbers
set relativenumber      " Relative line numbers
set mouse=a             " Enable mouse
set expandtab           " Spaces instead of tabs
set tabstop=4           " Tab width
set shiftwidth=4        " Indent width
set autoindent          " Auto indent
set smartindent         " Smart indent
set hlsearch            " Highlight search
set incsearch           " Incremental search
set ignorecase          " Case insensitive search
set smartcase           " Case sensitive if uppercase
set clipboard=unnamedplus " Use system clipboard
syntax on               " Syntax highlighting
set background=dark     " Dark background
set cursorline          " Highlight current line
set scrolloff=8         " Keep 8 lines above/below cursor
set signcolumn=yes      " Always show sign column
set updatetime=300      " Faster completion
set encoding=utf-8      " UTF-8 encoding
EOF

# Install claude
curl -fsSL https://claude.ai/install.sh | bash

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Installed:"
echo "  - tmux (with mouse support)"
echo "  - vim (configured, set as git editor)"
echo "  - htop, nvtop (system/GPU monitoring)"
echo "  - uv (Python package manager)"
echo "  - git-lfs (large file support)"
echo "  - jq (JSON parser)"
echo "  - ripgrep (rg), fd-find (fd), tree"
echo "  - ncdu (disk usage)"
echo "  - bat (better cat)"
echo "  - build-essential"
echo "  - claude (AI CLI)"
echo ""
echo "Profile files created:"
echo "  - ~/.runpod_profile.sh (main profile - move to persistent storage)"
echo "  - ~/.bashrc (updated to source the profile)"
echo ""
echo "Git aliases: gits, gitb, gitl"
echo ""
echo "Start a new shell or run: source ~/.bashrc"