from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.price_http import price_session


class WindowsCertificatePathTests(unittest.TestCase):
    def test_unicode_bundle_is_copied_without_changing_contents(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'Документы' / 'cacert.pem'
            source.parent.mkdir()
            contents = b'Test certificate bundle; no network call in this test.'
            source.write_bytes(contents)
            session = MagicMock(verify=str(source))
            session.__enter__.return_value = session
            with patch('tools.price_http.sys.platform', 'win32'), patch('tools.price_http.locale.getpreferredencoding', return_value='cp1252'), patch('tools.price_http.tempfile.gettempdir', return_value=root), patch('curl_cffi.requests.Session', return_value=session):
                with price_session() as result:
                    self.assertIs(result, session)
                    copied = Path(result.verify)
                    self.assertNotEqual(copied, source)
                    str(copied).encode('cp1252')
                    self.assertEqual(copied.read_bytes(), contents)
                    self.assertEqual(source.read_bytes(), contents)
                self.assertFalse(copied.exists())

    def test_encodable_bundle_is_not_replaced(self):
        session = MagicMock(verify='C:/certificates/cacert.pem')
        session.__enter__.return_value = session
        with patch('tools.price_http.sys.platform', 'win32'), patch('tools.price_http.locale.getpreferredencoding', return_value='cp1252'), patch('curl_cffi.requests.Session', return_value=session), patch('tools.price_http.tempfile.TemporaryDirectory') as temporary:
            with price_session() as result:
                self.assertEqual(result.verify, 'C:/certificates/cacert.pem')
            temporary.assert_not_called()

    def test_default_bundle_is_preserved_when_copying(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'Документы.pem'
            source.write_bytes(b'default CA bundle')
            session = MagicMock(verify=True)
            session.__enter__.return_value = session
            with patch('tools.price_http.sys.platform', 'win32'), patch('tools.price_http.locale.getpreferredencoding', return_value='cp1252'), patch('tools.price_http.tempfile.gettempdir', return_value=root), patch('curl_cffi.curl.DEFAULT_CACERT', str(source)), patch('curl_cffi.requests.Session', return_value=session):
                with price_session() as result:
                    self.assertEqual(Path(result.verify).read_bytes(), source.read_bytes())

    def test_non_windows_retains_default_provider_session(self):
        with patch('tools.price_http.sys.platform', 'linux'):
            with price_session() as result:
                self.assertIsNone(result)
