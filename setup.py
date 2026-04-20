from setuptools import setup, find_packages

setup(
    name="mlops-sentiment",
    version="1.0.0",
    description="End-to-end MLOps pipeline for sentiment analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "black", "ruff", "ipykernel"],
    },
    entry_points={
        "console_scripts": [
            "mlops-run=scripts.run_pipeline:main",
            "mlops-rollback=scripts.rollback:rollback",
        ]
    },
)
