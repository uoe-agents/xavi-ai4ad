import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='xavi-ai4ad',
                 version='1.0',
                 description='Implementation of XAVI from the paper:'
                             ' A Human-Centric Method for Generating Causal Explanations in Natural Language'
                             ' for Autonomous Vehicle Motion Planning',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author='Balint Gyevnar',
                 author_email='balint.gyevnar@ed.ac.uk',
                 url='https://github.com/uoe-agents/xavi-ai4ad',
                 packages=setuptools.find_packages(),
                 install_requires=requirements
                 )
