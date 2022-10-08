from setuptools import setup, find_packages

setup(
    name='ffxiv-stats',
    version='0.1',
    author='Acerola Paracletus',
    author_email='ffxivacerola@gmail.com',
    url='https://github.com/ffxiv-acerola/ffxiv_stats',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy >= 1.20.2',
        'matplotlib >= 3.4.2', 
        'pandas >= 1.2.4',
        'scipy >= 1.6.3'
    ]
)