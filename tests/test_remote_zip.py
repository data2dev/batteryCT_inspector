"""RemoteZip tests against an in-memory archive (no network)."""

import io
import zipfile
import zlib

import pytest

from batteryct.remote_zip import RemoteZip

URL = "https://example.invalid/archive.zip"


def make_zip(files: dict[str, bytes], force_zip64: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, (name, data) in enumerate(files.items()):
            ctype = zipfile.ZIP_DEFLATED if i % 2 == 0 else zipfile.ZIP_STORED
            info = zipfile.ZipInfo(name)
            info.compress_type = ctype
            with zf.open(info, "w", force_zip64=force_zip64) as f:
                f.write(data)
    return buf.getvalue()


class FakeServer:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.calls = 0

    def __call__(self, url: str, start: int, end: int) -> bytes:
        assert url == URL
        assert 0 <= start <= end < len(self.blob)
        self.calls += 1
        return self.blob[start : end + 1]


FILES = {
    "top/a.bin": b"alpha" * 1000,
    "top/b.bin": bytes(range(256)) * 40,
    "top/dir/c.bin": b"",
    "top/d.bin": b"x",
}


@pytest.mark.parametrize("zip64", [False, True])
def test_read_all_members(zip64):
    blob = make_zip(FILES, force_zip64=zip64)
    server = FakeServer(blob)
    rz = RemoteZip(URL, size=len(blob), fetch=server)
    assert sorted(rz.names()) == sorted(FILES)
    for name, data in FILES.items():
        assert rz.read(name) == data


def test_one_request_per_member():
    blob = make_zip(FILES)
    server = FakeServer(blob)
    rz = RemoteZip(URL, size=len(blob), fetch=server)
    before = server.calls
    rz.read("top/b.bin")
    assert server.calls - before == 1


def test_corruption_detected():
    blob = bytearray(make_zip({"top/a.bin": b"hello world" * 100}))
    rz = RemoteZip(URL, size=len(blob), fetch=FakeServer(bytes(blob)))
    m = rz.members["top/a.bin"]
    # flip a byte in the compressed payload
    pos = m.header_offset + 30 + len(m.name) + 5
    blob[pos] ^= 0xFF
    rz.fetch = FakeServer(bytes(blob))
    with pytest.raises((ValueError, zlib.error)):
        rz.read("top/a.bin")


def test_custom_fetch_requires_size():
    with pytest.raises(TypeError):
        RemoteZip(URL, fetch=FakeServer(make_zip(FILES)))
