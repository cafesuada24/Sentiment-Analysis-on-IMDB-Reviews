from typing import Union
import sys

import argparse
from pathlib import Path

MODEL_FILENAME = r'model.keras'
TOKENIZER_FILENAME = r'tokenizer.json'


def check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.train and args.review:
        parser.error('train mode cannot be combined with --review flag')
    if args.train and args.model:
        parser.error('train mode cannot be combined with --model flag')
    if args.train and not args.file:
        parser.error('you must provide the dataset to train model')
    if args.file and args.review:
        parser.error("file and review flag can not be used together")
    if not args.train and not (args.file or args.review):
        parser.error("predict mode require a CSV input or a review string")

def validate_path(path: Union[str, Path], file_type: str = None, err_msg: str = None, dir: bool = False) -> bool:
    if isinstance(path, str):
        path = Path(path)
    if (not dir and (not path.is_file() or path.suffix != file_type)) or (dir and not path.is_dir()):
        print(err_msg, file=sys.stderr)
        return False
    return True