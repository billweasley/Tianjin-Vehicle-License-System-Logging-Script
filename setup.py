from setuptools import find_packages, setup
setup(
    name='vehicle_license_checker',
    version='0.2',
    author='Haoxuan Wang',
    author_email='billweasley20092@gmail.com',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pydot",
        "tqdm",
        "beautifulsoup4",
        "tensorflow==2.2.0",
        "requests",
        "Image",
        "opencv-python==4.6.*",
        "pyyaml",
        "selenium"
    ],
)