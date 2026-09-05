import sys
import os
from typing import Generator
from pathlib import Path
import pytest

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def remove_tar() -> Generator[None, None, None]:
    yield
    os.remove("models/GRU_32_32.pth.tar")
    os.rmdir("models/")
