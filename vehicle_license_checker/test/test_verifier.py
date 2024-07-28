import os
from vehicle_license_checker.verifier_modeling import Verifier
import pytest

CODE_PATH = os.path.dirname(os.path.abspath(__file__))

#@pytest.fixture
def real_verifier() -> Verifier:
    return Verifier(os.path.join(CODE_PATH, "../../model/ctc_best.h5"))

def test_verifier(real_verifier: Verifier):
    res = real_verifier.get_result(os.path.join(CODE_PATH, "img/image1.jpg"))
    print(res)


test_verifier(real_verifier())