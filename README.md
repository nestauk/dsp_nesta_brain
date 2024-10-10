# Nesta brain

A research methods innovation project.

We're developing retrieval-augmented generation approaches to bolster internal knowledge management.

## Installation

We're using poetry for dependency management.

Run the following commands to install depedencies.

```
poetry install
poetry install --with lint
poetry insstall --with test
poetry run pre-commit install
```

To start an environment in your terminal

```
poetry env use python3.11
poetry shell
```

To add a new package, use poetry add:

```
poetry add package-name
```

## Repo structure

```
data/                 # Contains raw and processed datasets used for the project
dsp_nesta_brain/
├── notebooks/        # Jupyter notebooks for exploration and experimentation
├── pipeline/         # Data processing and analysis pipelines.
├── getters/          # Getter functions to get data from S3 or other sources
└── utils/            # Utility scripts and helper functions
```

Keep project related data in the `data` folder for local prototyping. When submitting code for PR reviews, best to store the data on S3 and add getter functions in `getters`.

Feel free to add other folders (eg for streamlit apps).
