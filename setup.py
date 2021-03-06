from setuptools import setup, find_packages

setup(
    name="voicefilter",
    version="0.0.1",
    author="Moritz Boos",
    author_email="moritz.boos@gmail.com",
    packages=find_packages(),
    install_requires=["numpy", "librosa", "matplotlib", "tqdm", "Pillow", "tensorboardX", "pyyaml", "mir_eval"]
)
