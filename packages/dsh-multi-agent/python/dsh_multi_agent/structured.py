"""Strict JSON normalization using a trusted, local JSON Schema contract."""
import json
import math

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource


def _no_external_resource(uri):
    raise NoSuchResource(ref=uri)


def json_result(schema, *, max_bytes=65536):
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, registry=Registry(retrieve=_no_external_resource))

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Non-finite JSON number")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite JSON number")
        return result

    def parse(text):
        if not isinstance(text, str) or len(text.encode()) > max_bytes:
            raise ValueError("Task output exceeds its JSON budget")
        try:
            value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant,
                               parse_float=finite_float)
            validator.validate(value)
        except (ValidationError, RecursionError) as error:
            raise ValueError("Output failed the task schema") from error
        return value

    return parse
