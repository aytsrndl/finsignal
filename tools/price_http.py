"""Windows-safe certificate paths for the price provider's curl backend."""
from contextlib import contextmanager
import locale
import os
from pathlib import Path
import sys
import tempfile


@contextmanager
def price_session():
    """Keep the selected trust bundle intact when its path cannot be ANSI encoded.

    curl_cffi encodes Windows certificate paths using the system code page.
    A virtual environment under a Unicode directory can therefore fail before
    any HTTPS request. A private temporary copy changes only the path, never
    the trust roots or verification settings. Non-Windows callers use yfinance's
    default session.
    """
    if sys.platform != 'win32':
        yield None
        return

    from curl_cffi.curl import DEFAULT_CACERT
    from curl_cffi.requests import Session

    with Session(impersonate='chrome') as session:
        source = Path(session.verify if isinstance(session.verify, str) else DEFAULT_CACERT)
        encoding = locale.getpreferredencoding(False)
        try:
            str(source).encode(encoding)
        except UnicodeEncodeError:
            # Prefer the user's temporary directory; LocalAppData is a fallback
            # if TEMP itself contains characters unavailable in this code page.
            candidates = [tempfile.gettempdir(), os.environ.get('LOCALAPPDATA')]
            root = None
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    str(candidate).encode(encoding)
                except UnicodeEncodeError:
                    continue
                if Path(candidate).is_dir() and os.access(candidate, os.W_OK):
                    root = candidate
                    break
            if root is None:
                raise ValueError('Set TEMP to a writable ASCII-only directory before fetching prices.')
            with tempfile.TemporaryDirectory(prefix='finsignal-ca-', dir=root) as directory:
                destination = Path(directory) / 'cacert.pem'
                destination.write_bytes(source.read_bytes())
                session.verify = str(destination)
                yield session
        else:
            yield session
