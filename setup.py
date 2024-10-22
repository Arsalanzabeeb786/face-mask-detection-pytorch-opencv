from setuptools import find_packages, setup
from typing import List

# Constant representing the editable install requirement '-e .' (commonly used for local projects)
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements file and returns a list of requirements,
    excluding any instances of '-e .'.

    Args:
        file_path (str): The path to the requirements.txt file.

    Returns:
        List[str]: A list of package names required by the project.
    """
    try:
        # Open the file and read all non-empty lines after stripping newline characters
        with open(file_path, 'r') as file_obj:
            requirements = [line.strip() for line in file_obj if line.strip()]
    
    # Handle the case where the file doesn't exist (e.g., if the requirements file is missing)
    except FileNotFoundError:
        return []

    # Remove the '-e .' entry if it's present, as it's used for local project installation
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements

# Setup function that configures the package details for distribution
setup(
    name='face-mask-detection-pytorch-opencv',  # Name of the project
    version='0.0.1',  # Initial version of the project
    author='Arsalan Zabeeb',  # Author's name
    author_email='arsalanzabeeb786@gmail.com',  # Author's email
    packages=find_packages(),  # Automatically find all packages and sub-packages
    install_requires=get_requirements('requirements.txt'),  # List of dependencies from requirements.txt
)
