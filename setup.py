from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str = "requirements.txt") -> List[str]:
    """
    This function returns the list of requirements
    by reading from the requirements.txt file.
    It removes '-e .' if present.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements




setup(
    name="mlproject",
    version="0.0.1",
    description="End-to-end ML project 1",
    author="Bala",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
 
)
