from setuptools import setup, find_packages

setup(
    name='q2',
    version='0.1.2',
    description='A reinforcement learning framework and command line tool',
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
