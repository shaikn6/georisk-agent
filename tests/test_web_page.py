"""Guards for the single-file demo page: no browser needed."""

import re
import shutil
import subprocess
from importlib import resources

import pytest
from fastapi.testclient import TestClient

from georisk import api

HTML = resources.files("georisk").joinpath("web/index.html").read_text()


def test_every_element_id_the_script_uses_exists_in_the_markup():
    used = set(re.findall(r'getElementById\("([^"]+)"\)', HTML))
    defined = set(re.findall(r'\bid="([^"]+)"', HTML)) | set(re.findall(r'\.id="([^"]+)"', HTML))  # markup ids + ids set from script
    assert used and used <= defined, f"script references missing ids: {sorted(used - defined)}"


def test_page_uses_textcontent_not_innerhtml_for_server_data():
    assert "innerHTML" not in HTML  # all dynamic text goes through textContent, so the API can never inject markup


def test_csp_allows_only_the_hosts_the_page_needs():
    csp = TestClient(api.app).get("/").headers["content-security-policy"]
    for host in ("fonts.googleapis.com", "fonts.gstatic.com", "services.arcgisonline.com", "nominatim.openstreetmap.org", "cdnjs.cloudflare.com"):
        assert host in csp
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp and "*" not in csp.replace("*.", "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_csv_cells_are_spreadsheet_safe_but_numbers_stay_numbers():
    line = next(ln for ln in HTML.splitlines() if ln.startswith("function safeCell"))
    src = re.sub(r"/\*.*?\*/", "", line)  # drop the inline comments, keep the code
    js = src + '''
const cases = [[-83.024149, '-83.024149'], [0.5, '0.5'], ['=1+1', "'=1+1"], ['+cmd', "'+cmd"], ['@SUM(A1)', "'@SUM(A1)"],
               ['a,b', '"a,b"'], ['say "hi"', '"say ""hi"""'], [null, ''], ['plain', 'plain']];
for (const [i, want] of cases) { const got = safeCell(i); if (got !== want) { console.error(JSON.stringify([i, got, want])); process.exit(1); } }
'''
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=20, check=False)
    assert r.returncode == 0, r.stderr
