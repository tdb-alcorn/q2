from setuptools import setup

setup(
    name='q2',
    version='0.1',
    description='A reinforcement learning framework and command line tool',
    url='https://github.com/tdb-alcorn/q2',
    author='Tom Alcorn',
    author_email='tdb.alcorn@gmail.com',
    license='MIT',
    packages=['q2'],
    install_requires=[
        'numpy',
        'tensorflow',
        'PyYAML',
        'gym',
        'gym-retro',
    ],
    entry_points={
        'console_scripts': ['q2=q2:main']
    },
    zip_safe=False,
)
