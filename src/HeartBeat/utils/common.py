import os
from box.exceptions import BoxValueError
import yaml
from HeartBeat.logging import logger
from ensure import ensure_annotations # for reduced bugs
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """read ymal files and returns

    args: value error if ymal file is empty
    e if empty file

    return:
        configBox: configbox type    
    """
    try:
        with open(path_to_yaml) as ymal_file:
            content = yaml.safe_load(ymal_file)
            logger.info(f"ymal file {path_to_yaml} loaded sucessfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("ymal is emepty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """create list of directories
    
    args:
        path to directories (list) : list of path to directories
        ignore_log (bool, optional): ignore if multiple directories is to be created. default to false
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in kb

    args:
        path (Path): path of the file
        
    Return:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"