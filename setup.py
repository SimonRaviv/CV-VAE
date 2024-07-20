from setuptools import setup, find_packages

setup(
    name='cv_vae',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'diffusers',
        'xformers',
        'numpy',
        'decord',
        'einops',
        'fire',
        'prettytable'
    ]
)
