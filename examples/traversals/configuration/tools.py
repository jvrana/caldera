from dataclasses import fields
from dataclasses import is_dataclass
from typing import Any
from typing import Dict
from typing import Type
from typing import TypeVar

from omegaconf import DictConfig
from omegaconf import OmegaConf


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


def validate_all_accessible(cfg: DictConfig):
    for k in cfg:
        v = cfg[k]
        if isinstance(v, DictConfig):
            validate_all_accessible(v)


def validate_structured(klass: Type, cfg: DictConfig):
    OmegaConf.structured(klass).update(cfg)


class ConfigObj:
    """Generic configuraiton object that performs validation.

    Can convert from dataclass to DictConfig.
    """

    @classmethod
    def validate_cfg(cls, cfg: DictConfig):
        validate_structured(cls, cfg)
        validate_all_accessible(cfg)

    def validate(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, ConfigObj):
                value.validate()

    @classmethod
    def from_dict_config(cls, cfg: DictConfig):
        cls.validate_cfg(cfg)
        return dataclass_from_dict(cls, OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return dataclass_from_dict(cls, data)

    def to_dict(self):
        return dataclass_to_dict(self)
