# Data Anonymisation and Differential Privacy

This project focuses on implementing data anonymization techniques and applying differential privacy principles to protect sensitive information in datasets. It includes the integration of the `diffprivlib` library for implementing differential privacy algorithms.

## Table of Contents

- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Dataset](#dataset)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the era of data-driven decision-making, ensuring privacy and confidentiality is crucial, especially when dealing with sensitive datasets. This project explores techniques for anonymizing data and preserving privacy using differential privacy methods. The aim is to provide a framework for securely analyzing datasets while protecting individuals' privacy.

## Directory Structure

- `.idea/`: Contains project-specific settings for your IDE.
- `diffprivlib/`: The differential privacy library used in the project.
- `research papers/`: Includes relevant research papers on data anonymization and differential privacy.
- `diabetes.csv`: Dataset used for demonstrating data anonymization and privacy preservation techniques.
- `feature_distributions_after_dp_diabetes.csv`: Output file containing feature distributions after applying differential privacy to the diabetes dataset.
- `naive-diabetes.py`: Python script for analyzing the diabetes dataset with and without applying differential privacy techniques.
- `requirements.txt`: List of required Python packages for running the project.

## Usage

To use this project:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the `main.py` script to analyze the diabetes dataset with and without applying differential privacy techniques.

## Dataset

The `diabetes.csv` file contains a dataset with information about individuals' health metrics, including age, BMI, blood pressure, and diabetes status. This dataset is used for demonstrating data anonymization and privacy preservation techniques.

## Files

- `naive-diabetes.py`: This Python script demonstrates the application of differential privacy techniques to the diabetes dataset.
- `feature_distributions_after_dp_diabetes.csv`: Output file containing feature distributions after applying differential privacy to the diabetes dataset.
- `requirements.txt`: This file lists the required Python packages for running the project.

## Contributing

Contributions to this project are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
