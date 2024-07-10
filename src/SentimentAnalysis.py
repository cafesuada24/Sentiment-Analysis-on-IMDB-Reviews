import os
import uuid
from pathlib import Path
import argparse

from core.predict import predict
from core.train import train
from lib import check_args, validate_path

current_dir = os.getcwd()
DEFAULT_MODEL_DIRECTORY = Path(current_dir + r'\model')
DEFAULT_PREDICTED_OUPUT_DIRECTORY = Path(current_dir + r'\predicted')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='SentimentAnalysis.py',
        description='The model with ability to predict the sentiment of user based on their review',
    )
    parser.add_argument('-t', '--train', action='store_true', help="switch to train mode")
    parser.add_argument('-r', '--review', metavar='STRING', type=str, nargs='?', help="evaluate the sentiment of given review")
    parser.add_argument('-f', '--file', metavar='CSV_FILE', type=str, nargs='?', help="evaluate the sentiment from dataset containing 'review' column")
    parser.add_argument('-o', '--output', metavar='OUTPUT', type=str, nargs='?', help="provide output")
    parser.add_argument('-m', '--model', metavar='DIR', type=Path, nargs='?', help='the model directory', const=DEFAULT_MODEL_DIRECTORY)

    return parser

def main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    check_args(args, parser)

    if args.train:
        dataset = Path(args.file) 
        output = Path(args.output) if args.output else DEFAULT_MODEL_DIRECTORY
        if not output.exists():
            os.mkdir(output)
        if not (
            validate_path(dataset, '.csv', 'Dataset file does not exists or is not a CSV file') and
            validate_path(output, None, 'The output dir does not exist or is not a directory', True)
        ):
            return

        train(dataset, output)
    else:
        input = args.review[0] if args.review else Path(args.file)
        model = Path(args.model) if args.model else DEFAULT_MODEL_DIRECTORY
        output = Path(args.output) if args.output else DEFAULT_PREDICTED_OUPUT_DIRECTORY.joinpath(str(uuid.uuid4()) + '.csv')
        if not (
            (isinstance(input, str) or validate_path(input, '.csv', 'Data to predict does not exists or is not a CSV file')) and
            validate_path(model, None, 'Model directory using to predict does not exists or is not a directory', True)
        ):
            return
        predict(model, input, output)
    print('DONE')

if __name__ == '__main__':
    parser = get_parser()
    main(parser)