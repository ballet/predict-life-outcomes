from setuptools import setup, find_packages

requirements = [
    'ballet==0.19.4',
]

setup(
    name='fragile_families',
    version='0.1.0-dev',
    packages=find_packages(where='src', include=('fragile_families', 'fragile_families.*')),
    package_dir={'': 'src'},
    install_requires=requirements,

    # metadata
    author='Micah Smith',
    author_email='micahs@mit.edu',
    description='A data science project built on the Ballet framework',
    url='https://github.com/HDI-Project/ballet-fragile-families',
)
