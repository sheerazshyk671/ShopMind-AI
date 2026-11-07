"""Download the two CSV exports of Online Retail II from a GitHub mirror."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

FILES = {
    'b198faf46c95ee1e1e02fe045941abb3b4508e37': 'c0b620b14bba72407be681523fffe958ebd588c34ef21f5143ea9a26dac9be3b',
    '714fc5de5b08fec5c9a0478e470dc838a60af940': '2eadbfd77e66707cb2ffcb458f73db4f39948a8ecabba7c2adb5ce1a270bddf0',
}
out = Path('data/raw')
out.mkdir(parents=True, exist_ok=True)
for sha, expected in FILES.items():
    url = f'https://api.github.com/repos/nicolexyu/Team_Project_Online_Retail/git/blobs/{sha}'
    req = urllib.request.Request(url, headers={'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'ShopMind-AI'})
    with urllib.request.urlopen(req) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f'Checksum mismatch for {sha}')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        # Extract only a flat CSV filename, never arbitrary archive paths.
        for member in archive.namelist():
            if member.endswith('.csv') and Path(member).name == member:
                (out / member).write_bytes(archive.read(member))
                print('Saved', out / member)
