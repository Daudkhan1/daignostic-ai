from setuptools import setup, find_packages

setup(
    name="maanz_medical_ai_models",
    version="1.0.0",
    description="Includes all of the models used by maanz",
    author="Ahmed Nadeem",
    author_email="ahmed.nadeem@maanz-ai.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "maanz_medical_ai_models": ["*.pth", "*.pt"],
    },
    install_requires=[],  # Add dependencies here
    python_requires=">=3.11",
)
