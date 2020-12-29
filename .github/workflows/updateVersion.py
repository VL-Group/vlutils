import argparse

parser = argparse.ArgumentParser(description='Script for compose version.')
parser.add_argument('--branch', help="Current git branch.", type=str)
parser.add_argument('--hash', help="Current commit SHA", type=str)

args = parser.parse_args()

print(args.branch)
print(args.hash)
