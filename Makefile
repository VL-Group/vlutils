.PHONY: clean update build

clean:
	rm -rf cfmUtils/BUILD

update:
	@echo "Check build version by branch"
	python ".github/workflows/updateVersion.py" $(ARGS)

build: update
	@echo "Install tools"
	python -m pip install setuptools wheel --user
	@echo "Packaging tarball and whl"
	python setup.py sdist bdist_wheel

all: update build clean
