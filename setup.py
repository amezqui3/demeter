from distutils.core import setup

setup(
    name                           = "demeter",
    author                         = "Erik Amezquita",
    author_email                   = "amezqui3@msu.edu",
    packages                       = ["demeter"],
    description                    = "the Euler Characteristic Transform applied to barley data",
    long_description_content_type  = "text/markdown",
    long_description               = open("README.md", "r").read(),
    url                            = "https://github.com/amezqui3/demeter",
    classifiers                    = ("Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent"),
)
