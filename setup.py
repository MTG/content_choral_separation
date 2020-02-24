import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pritish-phd-research",
    version="0.0.1",
    author="Pritish Chandna",
    author_email="pritish.chandna@upf.edu",
    description="A python library for Pritish's PhD research, including functions for source separation, voice change, synthesis and analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
