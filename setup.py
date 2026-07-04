from setuptools import setup, find_packages
from typing import List

HYPHEN_e_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    '''
    This will return the packages required for this setup
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        # remove \n from the list
        for req in requirements:
            req_clean = req.replace('\n', '')
            if HYPHEN_e_DOT in req_clean:
                req_clean = req_clean.replace(HYPHEN_e_DOT, '')

            if req_clean:
                requirements.append(req_clean)
    
    return requirements


setup(
    Name = "mlops-tutorial",
    version = "0.0.1",
    author = "Ravi",
    author_email = "ravibhagatatgdsc@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),
)

