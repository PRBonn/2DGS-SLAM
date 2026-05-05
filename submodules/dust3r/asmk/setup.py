import sys
from setuptools import setup, Extension
from setuptools.command.install import install

class InstallWrapper(install):

    def run(self):
        try:
            import faiss
        except ImportError:
            sys.stderr.write("\nERROR: faiss package not installed (install either faiss-cpu or " \
                             "faiss-gpu before installing this package.).\n\n")
            sys.exit(1)

        install.run(self)


setup(
    name="asmk",
    version="0.1",
    description="ASMK Python implementation for ECCV'20 paper \"Learning and aggregating deep " \
                "local descriptors for instance-level recognition\"",
    author="Tomas Jenicek, Giorgos Tolias",
    packages=[
        "asmk",
    ],
    ext_modules=[Extension("asmk.hamming", ["cython/hamming.c"])],
    install_requires=[
        "numpy",
        "pyaml",
    ],
    cmdclass={
        "install": InstallWrapper,
    },
    zip_safe=True)
