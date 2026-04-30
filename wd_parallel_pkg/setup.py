from setuptools import setup, find_packages

setup(
    name="wd_parallel",
    version="0.1.0",
    description="Windows-native TP+SP training framework (no NCCL required)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
    ],
)
