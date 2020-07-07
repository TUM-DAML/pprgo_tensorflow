from setuptools import setup

# Note that CUDA 10.0 includes a bug that affects PPRGo on large graphs, so we urgently advise using e.g. 10.1
install_requires = [
        "numpy",
        "scipy>=1.3",
        "numba>=0.49",
        "tensorflow<2.0",
        "scikit-learn",
        "tqdm",
        "sacred",
        "seml"
]

setup(
        name='pprgo',
        version='1.0',
        description='PPRGo model, from "Scaling Graph Neural Networks with Approximate PageRank"',
        author='Aleksandar Bojchevski, Johannes Klicpera, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann',
        author_email='a.bojchevski@in.tum.de, klicpera@in.tum.de',
        packages=['pprgo'],
        install_requires=install_requires,
        zip_safe=False
)
