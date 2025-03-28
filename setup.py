from setuptools import setup, find_packages

setup(
    name='space-trajectory-optimizer',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.6',
        'plotly>=5.7.0',
        'scipy>=1.7.3',
        'matplotlib>=3.5.2'
    ],
    author='SpaceHack Team',
    description='Advanced spacecraft trajectory optimization tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/space-trajectory-optimizer',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
)
