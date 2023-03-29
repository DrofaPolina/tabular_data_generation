
from setuptools import find_packages, setup
setup(
    name='tabular_data_generation',
    packages=find_packages(include=['tabular_data_generation']),
    version='0.1.0',
    description='Table Data Generation library',
    author='Me',
    license = 'MIT',
    install_requires=["numpy>=1.20.0,<2;python_version<'3.10'",
                        "numpy>=1.23.3,<2;python_version>='3.10'",
                        "pandas>=1.1.3,<2;python_version<'3.10'",
                        "pandas>=1.5.0,<2;python_version>='3.10'",
                        'ctgan>=0.7.1,<0.8'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)