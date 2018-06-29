from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='q2',
    version='0.1.3',
    description='A reinforcement learning framework and command line tool',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Environment :: Console',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Artificial Life',
    ],
    keywords='deep reinforcement learning framework tool',
    url='https://github.com/tdb-alcorn/q2',
    author='Tom Alcorn',
    author_email='tdb.alcorn@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'PyYAML',
        'gym',
        'gym-retro',
    ],
    entry_points={
        'console_scripts': ['q2=q2.tool:main'],
    },
    include_package_data=True,
    zip_safe=False,
)
