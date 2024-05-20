import streamlit as st
import pandas as pd
import joblib
import base64
from io import StringIO
from preprocessing import cleaner 

def load_model_xgboost():
    # Load the model
    loaded_model = joblib.load('xgboost_model.pkl')
    # Load the TfidfVectorizer
    loaded_vectorizer = joblib.load('xgboost_tfidf_vectorizer.pkl')
    # Load the LabelEncoder
    loaded_encoder = joblib.load('xgboost_label_encoder.pkl')
    return loaded_model, loaded_vectorizer, loaded_encoder


# Classify a CSV File
def classify_csv(input_csv):
    # Load the model, vectorizer, and label encoder
    model, vectorizer, encoder =  load_model_xgboost()
    # Preprocess and clean data 
    df = cleaner(input_csv)
    # Vectorize the comments
    X_new_tfidf = vectorizer.transform(df['comments'])
    # Predict the labels
    y_pred = model.predict(X_new_tfidf)
    # Decode the labels
    df['Label'] = encoder.inverse_transform(y_pred)
    # Save the results to a new CSV
    output_csv = "classified.csv"
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

# pip3 install streamlit 