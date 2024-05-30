import streamlit as st
import pandas as pd
import joblib
import base64
from preprocessing import cleaner 
import nltk

nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

# Function to refine predictions
def refine_predictions(comments, predictions, label_mapping):
    keywords = [
        'vet tech', 'vet technician', 'vet assistant', 'tech', 'assistant',
        'RVT', 'CVT', 'LVT', 'VTS', 'CVPM', 
        'AVMA', 'NAVTA','DACVS', 'DACVIM', 'javma',
        'DACVECC', 'DACVR', 'vca', 'va', 'CVA'
    ]
    refined_predictions = []
    for comment, prediction in zip(comments, predictions):
        if any(keyword.lower() in comment.lower() for keyword in keywords):
            refined_predictions.append(label_mapping.transform(['Other'])[0])
        else:
            refined_predictions.append(prediction)
    return refined_predictions

def load_model_xgboost():
    # Load the model
    loaded_model = joblib.load('./models/4xgboost_model.pkl')
    # Load the TfidfVectorizer
    loaded_vectorizer = joblib.load('./models/4xgboost_tfidf_vectorizer.pkl')
    # Load the LabelEncoder
    loaded_encoder = joblib.load('./models/4xgboost_label_encoder.pkl')
    return loaded_model, loaded_vectorizer, loaded_encoder

def load_model_voting1():
    # Load the VotingClassifier model
    loaded_voting_clf = joblib.load('voting_classifier_model.pkl')
    # Load the TfidfVectorizer
    loaded_vectorizer = joblib.load('voting_classifier_tfidf_vectorizer.pkl')
    # Load the LabelEncoder
    loaded_encoder = joblib.load('voting_classifier_label_encoder.pkl')
    # Load the MaxAbsScaler
    loaded_scaler = joblib.load('voting_classifier_maxabs_scaler.pkl')
    return loaded_voting_clf, loaded_vectorizer, loaded_encoder, loaded_scaler

# Classify a CSV File
def classify_csv_voting(df):
    # Load the model, vectorizer, label encoder, and scaler
    model, vectorizer, encoder, scaler = load_model_voting1()
    # Preprocess and clean data
    df = cleaner(df)
    # Vectorize the comments
    X_new_tfidf = vectorizer.transform(df['comments'])
    # Scale the features using the loaded scaler
    X_new_scaled = scaler.transform(X_new_tfidf)
    # Predict the labels
    y_pred = model.predict(X_new_scaled)
    # Decode the labels
    df['Label'] = encoder.inverse_transform(y_pred)
    # Save the results to a new CSV
    output_csv = "classified_voting.csv"
    df.to_csv(output_csv, index=False)
    print(f"Voting Classified data saved to {output_csv}")
    return df

# Classify a CSV File
def classify_csv(df):
    # Load the model, vectorizer, and label encoder
    model, vectorizer, encoder = load_model_xgboost()
    # Preprocess and clean data
    df = cleaner(df)
    # Vectorize the comment
    X_new_tfidf = vectorizer.transform(df['comments']) 
    # Predict the labels
    y_pred = model.predict(X_new_tfidf)
    # Refine the predictions
    #label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    #y_pred_refined = refine_predictions(df['comments'], y_pred, label_mapping)
    y_pred_refined = refine_predictions(df['comments'], y_pred, encoder)
    # Decode the labels
    df['Label'] = encoder.inverse_transform(y_pred_refined)
    # Save the results to a new CSV
    output_csv = "refined_classified.csv"
    df.to_csv(output_csv, index=False)
    print(f"Classified data saved to {output_csv}")
    return df

# Streamlit app
st.title("Reddit Comment Classification App")
st.write("Upload a CSV file with 'username' and 'comment' columns to classify the comments.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        # Check if the necessary columns are present
        if 'username' not in df.columns or 'comments' not in df.columns:
            st.error("CSV file must contain 'username' and 'comment' columns.")
        else:
            # Classify the comments
            classified_df = classify_csv(df)
            #classified_df = classify_csv_voting(df)
            # Display the classified DataFrame
            st.write("Classified Data:")
            st.dataframe(classified_df)
            # Convert DataFrame to CSV
            csv = classified_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            download_link = f'<a href="data:file/csv;base64,{b64}" download="classified_comments.csv">Download classified CSV file</a>'
            st.success("Comments classified successfully!")
            st.balloons()
            st.markdown(download_link, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Instructions and information
st.write("""
### Instructions
1. Ensure your CSV file has the columns 'username' and 'comment'.
2. Upload the CSV file using the file uploader above.
3. Once the file is uploaded and processed, you will see a link to download the classified CSV file.

### Notes
- The model will label each comment as either 'Medical Doctor', 'Veterinarian', or 'Other'.
- The output CSV file will contain the original data along with a new 'label' column.
""")