import argparse
import datetime

parser = argparse.ArgumentParser(description='Script for compose version.')
parser.add_argument('--branch', help="Current git branch.", type=str)
parser.add_argument('--hash', help="Current commit SHA", type=str)

args = parser.parse_args()


branch: str = args.branch.split("/")[-1]
sha: str = args.hash[:6]

with open("cfmUtils/VERSION", "r") as fp:
    verions = fp.read()

# check correct format
major, minor, micro = verions.split(".")

major = int(major)
minor = int(minor)
micro = int(micro)

if branch == "main":
    version = ".".join([str(major), str(minor), str(micro)]) + "dev{0}".format(datetime.datetime.today().strftime(r"%y%m%d"))
elif branch.startswith("r"):
    version = branch[1:]
    with open("cfmUtils/VERSION", "w") as fp:
        fp.write(version)
    print(f"Bump version to {version}")
with open("cfmUtils/BUILD", "w") as fp:
    fp.write(version)

print(f"Change build to {version}")
