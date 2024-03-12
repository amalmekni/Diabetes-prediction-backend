from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import fitz
import re

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'supersecret'

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl', 'rb'))

# Mapping full feature names to abbreviations
feature_mapping = {
    'Number of Pregnancies': 'pregs',
    'Glucose Level': 'gluc',
    'Blood Pressure': 'bp',
    'Skin Thickness': 'skin',
    'Insulin Level': 'insulin',
    'BMI': 'bmi',
    'Diabetes Pedigree Function': 'func',
    'Age': 'age',
}

def extract_data_from_pdf(file_path):
    # Open the PDF file
    with fitz.open(file_path) as pdf_document:
        # Extract text from all pages
        text = ''
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
          

        # Extract numerical values using regular expressions
        # pattern = re.compile(r'\b\d+\b')
        pattern = re.compile(r'\d+(?:\.\d+)?')
        matches = pattern.findall(text)

        print(matches)

        # Assuming the order of values is [pregs, gluc, bp, skin, insulin, bmi, func, age]
        extracted_data = [matches[i] for i in range(8)]

        return extracted_data
    
@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' in request.files:
            # If a file is uploaded, extract data from the PDF
            file = request.files['file']
            file.save('uploaded.pdf')
            input_features = extract_data_from_pdf('uploaded.pdf')

            # Map the feature names to abbreviations
            mapped_features = {feature_mapping[key]: value for key, value in zip(feature_mapping.keys(), input_features)}
            
            prediction = model.predict(scaler.transform([input_features]))
            response = {'prediction': int(prediction[0]), 'input_features': mapped_features}
        else:
            response = {'error': 'No file uploaded'}

        return jsonify(response)

    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Invalid input'}), 400    

@app.route('/', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()

        pregs = int(data.get('pregs', 0))
        gluc = int(data.get('gluc', 0))
        bp = int(data.get('bp', 0))
        skin = int(data.get('skin', 0))
        insulin = float(data.get('insulin', 0))
        bmi = float(data.get('bmi', 0))
        func = float(data.get('func', 0))
        age = int(data.get('age', 0))

        input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
        prediction = model.predict(scaler.transform(input_features))

        response = {'prediction': int(prediction[0])}
        return jsonify(response)

    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)
