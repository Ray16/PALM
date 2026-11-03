"""Seekable HTTP reader that refuses full-file downloads and counts traffic."""
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, max_bytes=64 * 1024**2):
        self.url, self.size, self.pos = url, size, 0
        self.bytes_read, self.requests_made = 0, 0
        self.max_bytes = max_bytes
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        if self.pos < 0:
            raise ValueError('negative seek')
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b''
        if self.bytes_read + n > self.max_bytes:
            raise RuntimeError(f'HTTP byte budget exceeded: {self.max_bytes}')
        start, end = self.pos, self.pos + n - 1
        # Keep the HF redirect request small; verify the final server supports ranges.
        r = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}',
                            'Accept-Encoding': 'identity'}, timeout=90, stream=True)
        try:
            if r.status_code != 206:
                raise RuntimeError(f'Refusing non-range response: HTTP {r.status_code}')
            expected = f'bytes {start}-{end}/{self.size}'
            if r.headers.get('Content-Range') != expected:
                raise RuntimeError(f'Unexpected Content-Range: {r.headers.get("Content-Range")}')
            payload = r.raw.read(n + 1)
            if len(payload) != n:
                raise RuntimeError(f'Expected {n} bytes, received {len(payload)}')
        finally:
            r.close()
        self.pos += n
        self.bytes_read += n
        self.requests_made += 1
        return payload

    def close(self):
        self.session.close()
        super().close()
