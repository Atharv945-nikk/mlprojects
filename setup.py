from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        requirements = f.readlines()

    requirements = [
        r.strip()
        for r in requirements
        if r.strip() and r.strip() != HYPHEN_E_DOT
    ]

    return requirements


setup(
    name="ml-project",
    version="0.0.1",
    author="Atharv Patil",
    author_email="atharvpatil49v@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
