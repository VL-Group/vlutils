"""Module of Config"""
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, get_origin, get_args, _SpecialForm, Union, _GenericAlias
from dataclasses import Field, is_dataclass

import yaml

T = TypeVar("T")

__all__ = [
    "read"
]

def _preprocess(plainStr: str, **varsToReplace):
    for key, value in varsToReplace.items():
        plainStr = plainStr.replace("{{{0}}}".format(key), str(value))
    return plainStr

def _assert(name, instance, types):
    if not isinstance(instance, types):
        raise TypeError(f"{name} in yaml not match the type definition in {name} ({types}), got {type(instance)}.")

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
    plainYaml = _preprocess(plainYaml, **varsToReplace)
    return _deserialize(yaml.full_load(plainYaml), classDef, logger or logging)
