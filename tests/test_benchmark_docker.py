import subprocess

import pytest

from fixpilot.benchmark import load_bugs, validate_bug

pytestmark = pytest.mark.docker

try:
    _up = subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0
except FileNotFoundError:
    _up = False
if not _up:
    pytest.skip("Docker is not running", allow_module_level=True)


@pytest.mark.parametrize("bug", load_bugs(), ids=lambda b: b.id)
def test_bug_breaks_exactly_its_listed_tests(bug):
    v = validate_bug(bug)
    assert v.ok, v.problems
