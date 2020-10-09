"""tensor_composition.py.

TensorComposition is intended to be a special dataclass that has the following features:

1. fields comprised of tensor instance
1. optionally validated tensor fields
1. common methods to applying tensor operations to groups of tensors
1. special datamaps that maps entries of tensors to each other
"""
import functools
import inspect

import torch


class TensorFieldBase:
    pass


class TensorField(TensorFieldBase):
    def __init__(self, validate: bool = True):
        self.annotation = None
        self.validate = validate


class TensorMap(TensorField):
    def __init__(self, validate: bool = True, dim: int = 1):
        super().__init__(validate)
        self.dim = dim


def tensorfield(c):
    if hasattr(c, "__annotations__"):
        for name, annot in c.__annotations__.items():
            v = getattr(c, name)
            if issubclass(v.__class__, TensorField):
                v.annotation = annot
    return c


class TensorCompositionMeta(type):
    _fields = tuple()
    _maps = tuple()

    def __new__(cls, clsname, superclasses, attributedict):
        do_validate = True
        for k, v in attributedict.items():
            if cls.is_tensor_map(v):
                cls._maps += ((k, v),)
            elif cls.is_tensor_field(v):
                cls._fields += ((k, v),)

        if cls._maps or cls._fields:
            has_new_fields = True
        else:
            has_new_fields = False
            do_validate = False

        if has_new_fields:
            if "__init__" not in attributedict:
                raise ValueError(
                    "__init__ must be defined if any {} or {} are defined".format(
                        TensorField, TensorMap
                    )
                )

        if do_validate:

            @functools.wraps(attributedict["__init__"])
            def __init__(self, *args, **kwargs):
                attributedict["__init__"](self, *args, **kwargs)
                self.__class__.validate()

        newcls = super().__new__(cls, clsname, superclasses, attributedict)
        newcls = tensorfield(newcls)
        if has_new_fields:
            argspec = inspect.getfullargspec(attributedict["__init__"])
            fields = cls._fields + cls._maps
            for field_name, field in fields:
                if field_name not in argspec.args:
                    raise ValueError(
                        "field '{}: {}' is missing from __init__ signature".format(
                            field_name, field.annotation
                        )
                    )
                elif (
                    field_name in argspec.annotations
                    and argspec.annotations[field_name] != field.annotation
                ):
                    raise ValueError(
                        "__init__ arg '{}:{}' annotation does not match field annotation {}".format(
                            field_name,
                            argspec.annotations[field_name],
                            field.annotation,
                        )
                    )

        if do_validate:

            @functools.wraps(attributedict["__init__"])
            def __init__(self, *args, **kwargs):
                attributedict["__init__"](self, *args, **kwargs)
                self.__class__.validate(self)

            newcls.__init__ = __init__
        return newcls

    def validate(cls, self):
        print("validating")
        for field_name, field in cls.fields + cls.maps:
            value = getattr(self, field_name)
            if cls.is_tensor_field_base(value):
                raise AttributeError(
                    "instance field '{}: {}' is not defined for {} instance".format(
                        field_name, field.annotation.__name__, cls.__name__
                    )
                )
            elif not isinstance(value, torch.Tensor):
                raise AttributeError(
                    "instance attribute '{}' was expected to be a torch.Tensor, but found a {}".format(
                        field_name, value.__class__
                    )
                )
            elif field.validate and value.dtype != field.annotation.dtype:
                raise AttributeError(
                    "instance attribute '{}' was expected to have dtype={}, but found {}".format(
                        field_name, value.dtype, field.annotation.dtype
                    )
                )

    @staticmethod
    def is_tensor_field_base(other):
        return issubclass(other.__class__, TensorFieldBase)

    @staticmethod
    def is_tensor_field(other):
        return isinstance(other, TensorField)

    @staticmethod
    def is_tensor_map(other):
        return isinstance(other, TensorMap)

    @property
    def fields(cls):
        return cls._fields

    @property
    def maps(cls):
        return cls._maps


class TensorComposition(metaclass=TensorCompositionMeta):
    def get_fields(self):
        return self.__class__.fields

    def get_maps(self):
        return self.__class__.maps


class MyComposition(TensorComposition):
    node: torch.FloatTensor = TensorField()
    edge: torch.FloatTensor = TensorField()
    edges: torch.LongTensor = TensorMap()

    def __init__(
        self, node: torch.FloatTensor, edge: torch.FloatTensor, edges: torch.LongTensor
    ):
        self.node = node
        self.edge = edge
        self.edges = edges


tc = MyComposition(torch.randn(5, 4), torch.randn(5, 4), torch.LongTensor([0, 1]))
# tc.get_maps()
tc.node
