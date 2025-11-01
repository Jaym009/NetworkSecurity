from setuptools import setup, find_packages
import os
from typing import List

def get_requirements(filename: str) -> List[str]:
    """Read requirements from a file and return them as a list."""
    requirement_list:List[str] = []
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")

    return requirement_list


setup(
    name='NetworkSecurity', 
    version='0.1.0',
    author='Jay Mervana',
    author_email='jaymervana421@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)