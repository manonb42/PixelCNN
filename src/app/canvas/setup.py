from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = "TODO"

setuptools.setup(
    name="streamlit-bit-canvas",
    version="0.0.1",
    author="Manon Baha",
    author_email="baha.manon@gmail.com",
    description="Draw big pixels on a canvas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.7",
    install_requires=[
        # By definition, a Custom Component depends on Streamlit.
        # If your component has other Python dependencies, list
        # them here.
        "streamlit >= 0.63",
    ],
    extras_require={
        "devel": [
            "wheel",
            "pytest==7.4.0",
            "playwright==1.48.0",
            "requests==2.31.0",
            "pytest-playwright-snapshot==1.0",
            "pytest-rerunfailures==12.0",
        ]
    }
)
