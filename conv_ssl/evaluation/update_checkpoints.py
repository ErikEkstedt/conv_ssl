from argparse import ArgumentParser
from os.path import dirname
from os import makedirs
from conv_ssl.evaluation.utils import get_checkpoint, load_paper_versions


parser = ArgumentParser()
parser.add_argument("--id", type=str)
parser.add_argument("--savepath", type=str)

args = parser.parse_args()
ch = get_checkpoint(run_path=args.id)

makedirs(dirname(args.savepath), exist_ok=True)
new_ch = load_paper_versions(ch, savepath=args.savepath)
