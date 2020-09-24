"""Configuration utilities."""
from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import Type
from typing import TypeVar

C = TypeVar("C")


def dataclass_to_dict(dataklass: C) -> Dict[str, Any]:
    """Converts a dataclass to a nested dictionary.

    :param dataklass: The dataclass instance
    :return: dictionary
    """
    data = {}
    if is_dataclass(dataklass):
        for field in fields(dataklass):
            value = getattr(dataklass, field.name)
            if is_dataclass(value):
                value = dataclass_to_dict(value)
            data[field.name] = value
    return data


def dataclass_from_dict(dataklasstype: Type[C], data: Dict[str, Any]) -> C:
    """Converts a nested dictionary to a dataclass.

    :param dataklasstype: the dataclass
    :param data: nested data as dictionary
    :return: dataclass instance
    """
    if is_dataclass(dataklasstype):
        for field in fields(dataklasstype):
            if is_dataclass(field.type):
                value = dataclass_from_dict(field.type, data[field.name])
            else:
                value = data[field.name]
            data[field.name] = value
        return dataklasstype(**data)
