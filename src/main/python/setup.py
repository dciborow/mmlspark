# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import os
import shutil 
from setuptools import setup, find_packages

setup(
    name="mmlspark",
    version=os.environ.get("MML_PY_VERSION","0.0.0"),
    description="Microsoft ML for Spark",
    long_description="Microsoft ML for Apache Spark contains Microsoft's open source " +
                     "contributions to the Apache Spark ecosystem",
    license="MIT",
    packages=find_packages() + ['mmlspark_pyspark.jars'],

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
    package_dir={
            'mmlspark_pyspark.jars': 'deps/jars',
    },
    package_data={
            'mmlspark_pyspark.jars': ['**.jar'],
            "mmlspark": ["../LICENSE.txt", "../README.txt"]},
    install_requires=[
        "pyspark",
        "numpy"
    ]
)
spark_home = os.environ["SPARK_HOME"]
shutil.copy("deps/jars/mmlspark_2.11-"+os.environ.get("MML_PY_VERSION")+".jar", spark_home + "/jars/")
