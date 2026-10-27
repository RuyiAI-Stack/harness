"""Small transport-level guards; source authorization belongs to the caller."""
import math

MAX_CANDIDATES = 200
MAX_PAGE_SIZE = 100
MAX_QUERY_TOKENS = 2048


def required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value


def validate_vector(vector, dimensions, *, strict=True):
    if (not isinstance(vector, (list, tuple)) or len(vector) != dimensions
            or any((type(v) not in {int, float} if strict else not isinstance(v, (int, float)))
                   or not math.isfinite(v) for v in vector) or not any(vector)):
        raise ValueError("invalid embedding dimension, value or zero norm")
    return vector


def page_budget(limit, offset):
    if (type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE
            or type(offset) is not int or not 0 <= offset < MAX_CANDIDATES
            or limit + offset > MAX_CANDIDATES):
        raise ValueError("vector query exceeds candidate budget")
