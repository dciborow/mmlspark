# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import os
from setuptools import setup, find_packages

setup(
    name="mmlspark",
    version=os.environ["MML_PY_VERSION"],
    description="Microsoft ML for Spark",
    long_description="Microsoft ML for Apache Spark contains Microsoft's open source " +
                     "contributions to the Apache Spark ecosystem",
    license="MIT",
    packages=find_packages() + ['pyspark.jars'],

    # Project's main homepage.
    url="https://github.com/Azure/mmlspark",
    # Author details
    author="Microsoft",
    author_email="mmlspark-support@microsoft.com",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3"
    ],

    zip_safe=True,
    include_package_data=True,
    data_files=[('lib', ['deps/jars/*.jar'])],
    package_dir={
            'pyspark.jars': 'deps/jars',
    },
    package_data={
            'pyspark.jars': ['*.jar'],
            "mmlspark": ["../LICENSE.txt", "../README.txt"]},
    install_requires=[
        "pyspark",
        "numpy"
    ]
)
