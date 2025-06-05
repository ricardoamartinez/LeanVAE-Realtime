"""
Setup script for Active Inference Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="active-inference-agent",
    version="1.0.0",
    author="Active Inference Agent Team",
    description="A revolutionary physics-informed neural architecture for real-time video processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ActiveInferenceAgent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.940",
        ],
        "experiments": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "active-inference-demo=active_inference_agent.demo:main",
        ],
    },
)