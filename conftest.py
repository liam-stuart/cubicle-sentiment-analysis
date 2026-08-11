import sys
import os
from pathlib import Path
import pytest

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def remove_tar():
    yield
    os.remove("src/GRU_32_32.pth.tar")
