from setuptools import setup

setup(
    name='utils',
    version='0.1',
    packages=['utils','tests'],
    url='gttps://github.com/Osburg/utils',
    license='',
    author='Aaron Osburg',
    author_email='osburg@stud.uni-heidelberg.de',
    description='',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        ],
    extras_require={
        "tests": ["pytest"],
    }
)