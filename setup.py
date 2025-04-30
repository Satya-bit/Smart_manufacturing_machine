from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Smart_manufacturing_machines",
    version="0.1",
    author="Satya",
    packages=find_packages(), #This will finmd the __init__.py file in every folder
    install_requires=requirements,
    
)