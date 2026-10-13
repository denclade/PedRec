# PedRec training environment. The PyTorch wheels ship their own CUDA 13 runtime, so a plain Ubuntu base is enough;
# the GPU driver / libcuda is injected at container start by the NVIDIA Container Toolkit (CDI, see devcontainer.json).
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates p7zip-full ffmpeg \
        libgl1 libglib2.0-0t64 libegl1 libxkbcommon0 libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# mise (task runner, pinned Python 3.14 + uv from mise.toml), installed system-wide from the official apt-less binary.
ARG MISE_VERSION=v2026.10.8
RUN curl -fsSL "https://github.com/jdx/mise/releases/download/${MISE_VERSION}/mise-${MISE_VERSION}-linux-x64" -o /usr/local/bin/mise \
    && chmod +x /usr/local/bin/mise

USER vscode
# Create the volume mount points (and their parents) as vscode, otherwise Docker creates ~/.local and ~/.cache as root
# and mise cannot write ~/.local/state. Named volumes inherit this ownership on first use.
RUN mkdir -p ~/.local/share/mise ~/.local/state ~/.cache/uv
ENV MISE_DATA_DIR=/home/vscode/.local/share/mise \
    MISE_TRUSTED_CONFIG_PATHS=/workspaces \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
RUN echo 'eval "$(mise activate bash)"' >> ~/.bashrc
