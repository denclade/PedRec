"""Dataset downloader: resumable download, mirror fallback, archive extraction, H36M frame extraction."""
import functools
import http.server
import os
import threading
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from pedrec.tools.datasets import download_datasets as dd


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_download_mirror_fallback_and_resume(server, tmp_path, monkeypatch):
    root, url = server
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    data = os.urandom(3 << 20)
    (root / "file.bin").write_bytes(data)
    target = tmp_path / "out" / "file.bin"
    # a partial previous download is continued (the simple test server ignores ranges -> restart)
    os.makedirs(target.parent)
    (tmp_path / "out" / "file.bin.part").write_bytes(data[:1000])
    dd.download([f"{url}/missing.bin", f"{url}/file.bin"], str(target))
    assert target.read_bytes() == data and not os.path.exists(str(target) + ".part")
    with pytest.raises(RuntimeError):
        dd.download([f"{url}/missing.bin"], str(tmp_path / "x.bin"))


@pytest.mark.parametrize("top_level", [True, False])
def test_download_and_extract(server, tmp_path, monkeypatch, top_level):
    root, url = server
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with zipfile.ZipFile(root / "ROMb.zip", "w") as z:
        prefix = "ROMb/" if top_level else ""
        z.writestr(f"{prefix}rt_rom_01b.pkl", b"df")
        z.writestr(f"{prefix}scene/img_00001.jpg", b"img")
    target = tmp_path / "datasets" / "ROMb"
    dd.download_and_extract(f"{url}/ROMb.zip", str(target), "rt_rom_01b.pkl")
    assert (target / "rt_rom_01b.pkl").read_bytes() == b"df" and (target / "scene" / "img_00001.jpg").is_file()
    assert not (tmp_path / "datasets" / "ROMb.zip").exists()
    dd.download_and_extract(f"{url}/missing.zip", str(target), "rt_rom_01b.pkl")  # skipped, marker exists


def test_extract_7z(tmp_path):
    py7zr = pytest.importorskip("py7zr")
    src = tmp_path / "src" / "RT3DValidate"
    os.makedirs(src)
    (src / "rt_validate_3d.pkl").write_bytes(b"df")
    with py7zr.SevenZipFile(tmp_path / "a.7z", "w") as z:
        z.writeall(str(src), "RT3DValidate")
    dd.extract(str(tmp_path / "a.7z"), str(tmp_path / "out"), keep_archive=True)
    assert (tmp_path / "out" / "rt_validate_3d.pkl").read_bytes() == b"df"


def test_h36m_images(tmp_path):
    base = tmp_path / "Human3.6m" / "val"
    os.makedirs(base / "S9" / "Videos")
    writer = cv2.VideoWriter(str(base / "S9" / "Videos" / "Walking 1.54138969.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 50, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    pd.DataFrame({"img_dir": pd.Categorical(["S9/Images/Walking 1.54138969"] * 3),
                  "img_id": np.array([1, 5, 10], np.uint32)}).to_pickle(base / "h36m_val_pedrec.pkl")
    image_dir = base / "S9" / "Images" / "Walking 1.54138969"
    dd.extract_h36m_images(str(tmp_path), ["val"], workers=0)  # default: every 64th row, as the validation
    assert sorted(os.listdir(image_dir)) == ["img_00001.jpg"]
    dd.extract_h36m_images(str(tmp_path), ["val"], workers=0, steps={"val": 1})
    assert sorted(os.listdir(image_dir)) == ["img_00001.jpg", "img_00005.jpg", "img_00010.jpg"]


def test_h36m_images_missing_videos_summary(tmp_path, capsys):
    base = tmp_path / "Human3.6m" / "train"
    os.makedirs(base)
    pd.DataFrame({"img_dir": pd.Categorical([f"S5/Images/{n}.54138969" for n in ("Eating", "Eating 1", "Photo")]),
                  "img_id": np.array([1, 1, 1], np.uint32)}).to_pickle(base / "h36m_train_pedrec.pkl")
    dd.extract_h36m_images(str(tmp_path), ["train"], workers=0, steps={"train": 1})
    out = capsys.readouterr().out
    assert "train/S5: 3 of 3 videos missing" in out and len(out.strip().splitlines()) == 1  # not one per video


def _zip_with_macos_junk(path, folder: str, files: dict):
    """Archive with a top level folder + __MACOSX/._* metadata (as created by the macOS Finder)."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{folder}/", b"")
        z.writestr(f"__MACOSX/._{folder}", b"junk")
        for name, data in files.items():
            z.writestr(f"{folder}/{name}", data)
            z.writestr(f"__MACOSX/{folder}/._{name}", b"junk")


def test_extract_removes_macos_junk(tmp_path):
    _zip_with_macos_junk(tmp_path / "a.zip", "ROMb", {"rt_rom_01b.pkl": b"df"})
    dd.extract(str(tmp_path / "a.zip"), str(tmp_path / "out"))
    assert os.listdir(tmp_path / "out") == ["rt_rom_01b.pkl"]
