from setuptools import setup

setup(
    name='mrs_utils',
    version='0.1',
    packages=['mrs_utils','tests'],
    url='https://github.com/Osburg/mrs_utils',
    license='',
    author='Aaron Osburg',
    author_email='osburg@stud.uni-heidelberg.de',
    description='',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'torch',
        'torchvision',
        'overrides',
        ],
    extras_require={
        "tests": ["pytest"],
    }
)