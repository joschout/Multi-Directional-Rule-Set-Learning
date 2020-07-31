import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mdrsl',
    version='0.1.0',
    # package_dir={'': 'src'},
    packages=setuptools.find_packages(),
    url='https://github.com/joschout/Multi-Directional_Rule_Set_Learning',
    license='Apache 2.0',
    author='Jonas Schouterden',
    author_email='',
    description='Multi-Directional Rule Set Learning',
    long_description=long_description,
    install_requires=['numpy']
)
