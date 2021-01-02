with open(".github/workflows/VERSION", "r") as fp:
    verions = fp.read()

# check correct format
major, minor, micro = verions.strip().split(".")

major = int(major)
minor = int(minor)
micro = int(micro)

version = ".".join(str(x) for x in [major, minor, micro + 1])

with open(".github/workflows/VERSION", "w") as fp:
    fp.write(version)
print(f"Increment version to {version}")
