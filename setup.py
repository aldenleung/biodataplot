from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
	readme = readme_file.read()

requirements = ["genomictools>=0.0.8", "biodata>=0.1.6", "simplevc>=0.0.4", "commonhelper>=0.0.5", "mphelper>=0.0.3", "numpy>=1.26.4", "scipy>=1.13.1", "matplotlib>=3.9.2"]

setup(
	name="biodataplot",
	version="0.0.5",
	author="Alden Leung",
	author_email="alden.leung@gmail.com",
	description="A python package with useful biological data plotting methods",
	long_description=readme,
	long_description_content_type="text/markdown",
	url="https://github.com/aldenleung/biodataplot/",
	packages=find_packages(),
	install_requires=requirements,
	classifiers=[
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
	]
)
