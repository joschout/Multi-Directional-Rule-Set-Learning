import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name='pyIDS',
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='MIT',
    author='Jiri Filip',
    author_email='',
    long_description=long_description,
    install_requires=['numpy']
)
