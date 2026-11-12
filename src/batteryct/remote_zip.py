"""Read individual members of a remote ZIP archive with HTTP range requests.

The Glimpse archives are 2-122 GB each, larger than the free compute disks we
use. Instead of downloading a whole archive, we read its central directory
(a few MB at the end of the file) and then fetch only the members we need,
one range request per member. ZIP64 archives are supported.
"""

from __future__ import annotations

import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass

# fetch(url, start, end_inclusive) -> bytes
RangeFetcher = Callable[[str, int, int], bytes]

_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
_LOCAL_SIG = b"PK\x03\x04"
USER_AGENT = "batteryct-inspector (+https://github.com/data2dev/batteryct_inspector)"


class RemoteAccessError(RuntimeError):
    """The remote server refused access (e.g. HTTP 403)."""


def http_range_fetch(
    url: str, start: int, end: int, retries: int = 4, timeout: float = 120.0
) -> bytes:
    """Fetch bytes ``start..end`` (inclusive) of ``url``, retrying transient errors.

    Redirects are followed, so short-lived pre-signed storage URLs are re-resolved
    on every call.
    """
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 206:
                    raise RemoteAccessError(f"{url}: server ignored Range (HTTP {resp.status})")
                data: bytes = resp.read()
            if len(data) != end - start + 1:
                raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise RemoteAccessError(
                    f"{url}: HTTP {e.code}. Some Figshare files refuse certain networks; "
                    "try from a GitHub-hosted runner (see docs/data.md)."
                ) from e
            if attempt == retries:
                raise
        except OSError:
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def remote_size(url: str, fetch: RangeFetcher = http_range_fetch) -> int:
    """Total size of the remote file, read from a one-byte range response."""
    if fetch is http_range_fetch:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                rng = resp.headers.get("Content-Range", "")
        except urllib.error.HTTPError as e:
            raise RemoteAccessError(f"{url}: HTTP {e.code}") from e
        if "/" not in rng:
            raise RemoteAccessError(f"{url}: no Content-Range header; Range not supported")
        return int(rng.rsplit("/", 1)[1])
    raise TypeError("pass size explicitly when using a custom fetcher")


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over a range fetcher (used only for the central directory)."""

    def __init__(self, url: str, size: int, fetch: RangeFetcher) -> None:
        self.url, self.size, self.fetch, self.pos = url, size, fetch, 0
        self.requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer: memoryview | bytearray) -> int:  # type: ignore[override]
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.fetch(self.url, self.pos, self.pos + n - 1)
        self.requests += 1
        buffer[:n] = data
        self.pos += n
        return n


@dataclass(frozen=True)
class Member:
    """One file inside the archive, as listed in the central directory."""

    name: str
    header_offset: int
    compress_size: int
    file_size: int
    compress_type: int
    crc: int


class RemoteZip:
    """Index of a remote ZIP archive plus per-member range fetches."""

    def __init__(
        self, url: str, size: int | None = None, fetch: RangeFetcher = http_range_fetch
    ) -> None:
        self.url = url
        self.fetch = fetch
        self.size = size if size is not None else remote_size(url, fetch)
        raw = _RangeFile(url, self.size, fetch)
        with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20)) as zf:
            self.members: dict[str, Member] = {
                i.filename: Member(
                    i.filename,
                    i.header_offset,
                    i.compress_size,
                    i.file_size,
                    i.compress_type,
                    i.CRC,
                )
                for i in zf.infolist()
                if not i.is_dir()
            }
        self.index_requests = raw.requests

    def names(self) -> list[str]:
        return list(self.members)

    def read(self, name: str) -> bytes:
        """Fetch, decompress and CRC-check one member with a single range request.

        The local header's extra field can differ from the central directory's, so
        we over-read a small margin and parse the real local header.
        """
        m = self.members[name]
        margin = 30 + len(m.name.encode()) + 1024
        end = min(m.header_offset + margin + m.compress_size, self.size) - 1
        blob = self.fetch(self.url, m.header_offset, end)
        fields = _LOCAL_HEADER.unpack_from(blob)
        if fields[0] != _LOCAL_SIG:
            raise ValueError(f"{name}: bad local header signature")
        name_len, extra_len = fields[9], fields[10]
        start = _LOCAL_HEADER.size + name_len + extra_len
        if start + m.compress_size > len(blob):
            blob += self.fetch(
                self.url,
                m.header_offset + len(blob),
                m.header_offset + start + m.compress_size - 1,
            )
        comp = blob[start : start + m.compress_size]
        if m.compress_type == zipfile.ZIP_STORED:
            data = comp
        elif m.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(comp, -15)
        else:
            raise ValueError(f"{name}: unsupported compression type {m.compress_type}")
        if len(data) != m.file_size or zlib.crc32(data) != m.crc:
            raise ValueError(f"{name}: size/CRC mismatch, refusing corrupted member")
        return data
