
from setuptools import setup, find_packages

setup(
  name = "cfmUtils",         # How you named your package folder (MyLib)
  packages = ["cfmUtils"],   # Chose the same as "name"
  version = "0.1",      # Start with a small number and increase it with every change you make
  license="apache-2.0",        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = "A simple shared tools.",   # Give a short description about your library
  author = "Xiaosu Zhu",                   # Type in your name
  author_email = "xiaosu.zhu@outlook.com",      # Type in your E-Mail
  url = "https://github.com/cfm-uestc/cfmUtils",   # Provide either the link to your github or to your website
  download_url = "https://github.com/cfm-uestc/cfmUtils/releases",    # I explain this later on
  keywords = ["Utilities"],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          "torch>=1.7",
          "tqdm",
          "nvidia-ml-py3"
      ],
  classifiers=[
    "Development Status :: 3 - Alpha",      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    "Intended Audience :: Developers",      # Define that your audience are developers
    "Topic :: Software Development :: Build Tools",
    "License :: OSI Approved :: Apache Software License",   # Again, pick a license
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
  ],
  python_requires='>=3.4',
)
