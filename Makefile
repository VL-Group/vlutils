.PHONY: clean build

clean:
	rm -rf vlutils/BUILD

build:
	@echo "Install tools"
	python -m pip install setuptools wheel --user
	@echo "Packaging tarball and whl"
	python setup.py sdist bdist_wheel

all: build clean
