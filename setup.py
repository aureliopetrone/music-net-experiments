from setuptools import setup, find_packages

setup(
    name="music-net-experiments",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pretty_midi",
        "pygame"
    ]
) 