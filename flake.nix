# SPDX-FileCopyrightText: 2026 Dennis Burgermeister
# SPDX-License-Identifier: GPL-3.0-or-later

{
  description = "PedRec MicroVM Dev Sandbox";

  inputs = {
    dev-sandbox.url = "github:denclade/nix-dev-sandbox";
  };

  outputs = { self, dev-sandbox }:
    let
      name = "pedrec";
      workspacePath = "/workspaces/${name}";
    in {
      # CPU only (`mise run setup:cpu` is enough); training and GPU tests run on the host via `mise run`.
      packages.x86_64-linux.default = dev-sandbox.lib.mkDevSandbox {
        inherit name;
        hostname = "pedrec-vm";
        hostAddress = "127.0.10.17";
        sshPort = 2200;
        vcpu = 8;
        mem = 12288;
        diskSize = 65536;
        # desktop: Xvfb :99 / ssh -X for the Qt demo, Mesa llvmpipe for PyOpenGL / pyqtgraph.
        profiles = [ "desktop" ];
        # Python 3.14 + uv come from mise.toml (python-build-standalone via nix-ld), not from Nix.
        extraPackages = pkgs: [ pkgs.ffmpeg pkgs.p7zip ];
        extraEnv = {
          MISE_TRUSTED_CONFIG_PATHS = workspacePath;
          PYTHONUNBUFFERED = "1";
        };
        # The venv holds a few GB of PyTorch / CUDA wheels: keep it off 9p and away from the host's .venv.
        persistDirs = [ ".venv" ];
        # Conti01 lives outside the project on the host, read-only at the place the code expects it.
        hostShares = [{
          tag = "conti01";
          envVar = "PEDREC_CONTI01_DIR";
          default = "/mnt/DataNVME/datasets/Conti01";
          mountPoint = "${workspacePath}/data/datasets/Conti01";
        }];
        extraModules = [
          ({ pkgs, ... }: {
            # Shared libs the pip wheels expect from the system (cf. the former .devcontainer/Dockerfile):
            # PyQt6 xcb platform plugin, PyOpenGL, opencv / matplotlib.
            programs.nix-ld.libraries = with pkgs; [
              libGL libGLU libglvnd libxkbcommon fontconfig freetype dbus glib
              libx11 libxext libxrender libxi libxcb
              libxcb-cursor libxcb-image libxcb-keysyms libxcb-render-util libxcb-util libxcb-wm
            ];
            environment.interactiveShellInit = ''
              eval "$(mise activate bash)"
            '';
          })
        ];
      };
    };
}
