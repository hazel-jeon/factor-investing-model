from setuptools import setup, find_packages

setup(
    name="factor_investing",
    version="0.1.0",
    description="Value, Momentum, Size factor investing model with walk-forward backtest",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/yourusername/factor-investing",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "yfinance>=0.2.38",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "scipy>=1.12.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "jupyter>=1.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Office/Business :: Financial",
    ],
)
