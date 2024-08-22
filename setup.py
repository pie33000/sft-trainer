from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="fine_tuning_llm",
    version="0.1.0",
    author="Pierrick Rugery",
    author_email="rugery.pierrick.us@gmail.com",
    description="Fine Tuning LLMs with SFT or DPO.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pie33000/sft-trainer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=required,
)
