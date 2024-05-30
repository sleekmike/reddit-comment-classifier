# Reddit Comment Classifier

## Project Overview

The Reddit Comment Classifier is a robust application designed to categorize Reddit comments into predefined labels: 'Medical Doctor', 'Veterinarian', or 'Other'. This tool leverages advanced machine learning techniques and NLP (Natural Language Processing) to analyze and classify textual data efficiently.

## Key Features

- **Model Training**: Utilizes XGBoost and Voting Classifier algorithms to predict the categories based on the content of comments.
- **Data Preprocessing**: Implements text cleaning, tokenization, and vectorization to prepare data for modeling.
- **Model Evaluation**: Employs cross-validation and grid search to fine-tune and evaluate model performance.
- **Web Interface**: Provides a Streamlit-based web interface for easy interaction with the classifier, allowing users to upload CSV files and receive classified outputs.

## Installation

Before setting up the project, ensure you have Python 3.8+ installed on your machine. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

Where `requirements.txt` includes:

```
imblearn
joblib
nltk
numpy
pandas
scikit-learn
streamlit
xgboost
psycopg2  # If interacting with PostgreSQL databases
```

## Usage

### Running the Streamlit App

To start the Streamlit web application, navigate to the project directory and run:

```bash
streamlit run app.py
```

### Command Line Scripts

Several Python scripts are provided for data processing and model operations:

- `download_data_to_csv.py`: Connects to a PostgreSQL database to download and save tables as CSV.
- `preprocessing.py`: Contains all preprocessing functions used to clean and prepare text data.
- `model.py`: Contains the model training and evaluation logic.

### Using the Web App

1. **Upload a CSV**: The CSV should contain at least two columns: 'username' and 'comments'.
2. **Classification**: The app processes the uploaded file, classifies the comments, and provides a downloadable CSV output with labels.

## Directory Structure

```
reddit-comment-classifier/
│
├── data/                   # Folder for datasets and outputs
├── models/                 # Trained model files and scalers
├── preprocessing           # Preprocessing script
├── train/                  # Training and utility script 
├── README.md               # Project documentation
├── app.py                  # Streamlit application entry point
└── requirements.txt        # Project dependencies
```

## Development

### Data Preprocessing

The `cleaner` function in `preprocessing.py` is central to data preparation, handling tasks like tokenization, stemming, and removal of stopwords.

### Model Training

Training scripts in the `train/` folder detail the setup and tuning of machine learning models, including cross-validation and hyperparameter optimization.

### Adding New Features

To contribute new features or models, extend the existing Python scripts or add new scripts in the `train/` directory. Ensure integration with the existing workflow, particularly data preprocessing and output.

## Contributing

Contributions to the Reddit Comment Classifier are welcome! Please fork the repository, make your changes, and submit a pull request.

## License

This project is open source, licensed under the MIT License.