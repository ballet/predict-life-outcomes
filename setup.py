from setuptools import setup, find_packages

requirements = [
    'ballet[all]==0.19.3',
    'pyarrow',
    'fsspec[s3]',
    'mlblocks',
    'scikit-learn',
]

extras = {
    'analysis': [
        'ffmetadata-py',
        'funcy',
        'pandas',
        'tqdm',
    ],
    'evaluation': [
        'python-dotenv',
        'timer_cm',
    ],
    'search': [
        'dill',
        'baytune',
        'stacklog',
    ]
}

setup(
    name='fragile_families',
    version='0.1.0-dev',
    packages=find_packages(where='src', include=('fragile_families', 'fragile_families.*')),
    package_dir={'': 'src'},
    install_requires=requirements,
    extras_require=extras,

    # metadata
    author='Micah Smith',
    author_email='micahs@mit.edu',
    description='A data science project built on the Ballet framework',
    url='https://github.com/HDI-Project/ballet-fragile-families',
)
