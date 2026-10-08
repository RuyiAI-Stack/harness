"""AST-only test association fixture, never executed as a numerical test."""
from relu_and_mul import relu_and_mul


def test_relu_and_mul_forward():
    raise RuntimeError("discovery fixture is not a numerical acceptance test")
