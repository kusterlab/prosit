from setuptools import setup

setup(
    name="prosit",
    version="1.0",
    description="prediction",
    url="http://github.com/kusterlab/prosit",
    author="Siegfried Gessulat",
    author_email="s.gessulat@gmail.com",
    packages=["prosit"],
    zip_safe=False,
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pylint"],
)
