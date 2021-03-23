"""Module of serialization/deserialization."""
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, get_origin, get_args, _SpecialForm, Union, _GenericAlias
from dataclasses import Field, is_dataclass, asdict
import keyword
from io import StringIO
import types

import yaml

T = TypeVar("T")


__all__ = [
    "read",
    "serialize"
]


def _preprocess(plainStr: str, **varsToReplace):
    for key, value in varsToReplace.items():
        plainStr = plainStr.replace("{{{0}}}".format(key), str(value))
    return plainStr


def _replaceKeyword(parsedYaml: dict) -> dict:
    if not isinstance(parsedYaml, dict):
        return parsedYaml
    newDict = dict()
    for key, value in parsedYaml.items():
        if keyword.iskeyword(key):
            key += "_"
        newDict[key] = _replaceKeyword(value)
    return newDict


def _assert(name, instance, types):
    if not isinstance(instance, types):
        raise TypeError(f"{name} in yaml not match the type definition in {name} ({types}), got {type(instance)}.")


def _serialize(instance: Any) -> dict:
    if isinstance(instance, (str, int, bool, float)):
        return instance
    if isinstance(instance, list, set, tuple):
        return instance.__class__(_serialize(x) for x in instance)
    if isinstance(instance, dict):
        return {k: _serialize(v) for k, v in instance.items()}
    return {k: _serialize(v) for k, v in instance.__dict__}


def _deserialize(parsedYaml: Union[dict, str, int, bool, float, list, set, tuple], classDef: Type[T], logger: Logger) -> T:
    if classDef in (str, int, bool, float):
        return classDef(parsedYaml)
    if classDef in (list, set, tuple):
        return classDef(x for x in parsedYaml)
    if classDef is dict:
        return classDef((k, v) for k, v in parsedYaml.items())
    annotations: Dict[str, Field] = classDef.__dataclass_fields__
    updateDict = dict()
    for attr, fieldDef in annotations.items():
        if attr not in parsedYaml and (fieldDef.default is None):
            raise AttributeError(f"{attr} not found in yaml and no init method provide.")
        if fieldDef.type in (str, int, bool, float):
            _assert(attr, parsedYaml[attr], fieldDef.type)
            updateDict[attr] = parsedYaml[attr]
        elif fieldDef.type in (list, set, tuple):
            _assert(attr, parsedYaml[attr], (list, set, tuple))
            updateDict[attr] = fieldDef.type(_deserialize(x, type(x), logger) for x in parsedYaml[attr])
        elif fieldDef.type is dict:
            _assert(attr, parsedYaml[attr], dict)
            updateDict[attr] = {k: _deserialize(x, type(x), logger) for k, x in parsedYaml[attr].items()}
        else:
            if isinstance(fieldDef.type, _GenericAlias):
                origin = get_origin(fieldDef.type)
                if origin not in (list, set, tuple, dict):
                    raise TypeError(f"{attr} in {classDef} is not any of (str, int, bool, float, list, set, tuple, dict), or generic list/dict/tuple, got {fieldDef.type}.")
                args = get_args(fieldDef.type)
                for arg in args:
                    if issubclass(arg, _SpecialForm):
                        raise NotImplementedError(f"Not support for {attr} which is annotated as a special form {arg}.")
                if origin in (list, set, tuple):
                    _assert(attr, parsedYaml[attr], (list, set, tuple))
                    updateDict[attr] = origin(_deserialize(x, args[0], logger) for x in parsedYaml[attr])
                elif origin is dict:
                    if args[0] is not str:
                        raise TypeError(f"Dict must have str as key, {attr} in {classDef} has {args[0]} as key.")
                    updateDict[attr] = {k: _deserialize(x, args[1], logger) for k, x in parsedYaml[attr].items()}
            elif is_dataclass(fieldDef.type):
                updateDict[attr] = _deserialize(parsedYaml[attr], fieldDef.type, logger)
            else:
                raise TypeError(f"Unrecognized type {fieldDef.type} of {attr} in {fieldDef}.")
    return classDef(**updateDict)


def read(configPath: str, varsToReplace: Dict[str, Any], classDef: Type[T], logger: Logger = None) -> T:
    """Read from config.yaml

    Args:
        configPath (str): The path of yaml

    Returns:
        T: The resulting config.
    """
    plainYaml = Path(configPath).read_text()
    if isinstance(varsToReplace, dict):
        plainYaml = _preprocess(plainYaml, **varsToReplace)
    result = _deserialize(_replaceKeyword(yaml.full_load(plainYaml)), classDef, logger or logging)

    # result.summary = types.MethodType(serialize, result)
    return result


def serialize(instance: Any, logger: Logger = None) -> dict:
    """Serialize any object to dict-like

    Args:
        instance (Any): Any object.
        logger (Logger, optional): Logger for logging. Defaults to None.

    Returns:
        dict: The serialized dict.
    """
    if is_dataclass(instance):
        return asdict(instance)
    (logger or logging).debug("Instance is not a dataclass, use custom serialize method.")
    return _serialize(instance)


def summary(instance, logger: Logger = None) -> str:
    """Serialize any object to yaml format summary

    Args:
        instance (Any): Any object.

    Returns:
        str: The serialized string.
    """
    with StringIO() as stream:
        yaml.safe_dump(serialize(instance), stream, default_flow_style=False)
        return stream.getvalue()


class Config:
    @classmethod
    def read(cls, configPath: str, varsToReplace: Dict[str, Any], logger: Logger = None):
        return read(configPath, varsToReplace, type(cls), logger)

    def serialize(self, logger: Logger = None):
        """Serialize self to dict-like

        Args:
            logger (Logger, optional): Logger for logging. Defaults to None.

        Returns:
            dict: The serialized dict.
        """
        return serialize(self, logger)

    def summary(self, logger: Logger = None):
        """Serialize self to yaml format summary

        Args:
            logger (Logger, optional): Logger for logging. Defaults to None.

        Returns:
            str: The serialized string.
        """
        return self.serialize(logger)
