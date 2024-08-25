# EyeBot

EyeBot is a comprehensive project aimed at diagnosing eye diseases and assisting users with relevant information through an AI-powered chatbot. The project consists of two main components: a computer vision model that classifies eye diseases and a fine-tuned language model chatbot designed to respond to queries about these diseases.

## Project Structure

The repository is organized as follows:

### 1. Eye Disease Classification
- **File**: `eye-classification.ipynb`
- **Description**: This notebook builds and trains a ResNet-101 model to classify eye diseases such as cataract, glaucoma, and diabetic retinopathy. The model uses TensorFlow and Keras, leveraging transfer learning to achieve high accuracy on medical image data.

### 2. Fine-Tuned Language Model Chatbot
- **File**: `eyebot-fine-tuned-llm.ipynb`
- **Description**: This notebook fine-tunes the LLaMA 2-7B Chat HF model using a custom dataset of query-response pairs related to eye diseases. The model is trained using efficient techniques (e.g., PEFT) to respond accurately to user queries.

### 3. Data Transformation
- **File**: `dataset_transformation.ipynb`
- **Description**: This notebook prepares the raw dataset into a format suitable for fine-tuning the language model. It transforms the data into the query-response pairs that the LLaMA model requires for training.

### 4. Flask API for Language Model
- **File**: `eyebot_api.ipynb`
- **Description**: This notebook deploys the fine-tuned language model on a server using Flask. The API serves as the backend, processing user queries and returning responses generated by the language model.

### 5. Streamlit Application
- **File**: `app.py`
- **Description**: The Streamlit app provides a user interface for interacting with EyeBot. Users can upload eye images for classification and input text queries to receive information about eye diseases from the chatbot.

## Setup Instructions

### Prerequisites
Before running the project, ensure that you have the following installed:
- Python 3.8 or higher
- Jupyter Notebook
- Flask
- Streamlit

### Installation Steps

## 1. Clone the Repository
   ```bash
   git clone https://github.com/yourusername/EyeBot.git
   cd EyeBot
```
## 2. Install Required Packages
   ```bash
   pip install -r requirements.txt
```
## 3. Run the Notebooks
Open the Jupyter notebooks (`eye-classification.ipynb`, `eyebot-fine-tuned-llm.ipynb`, etc.) and execute the cells to train and deploy the models.

## 4. Deploy the Flask API
Run the `eyebot_api.ipynb` notebook to start the Flask server. The API will be accessible for processing text queries.

## 5. Launch the Streamlit Application
   ```bash
   streamlit run app.py
```
## 6. Access the Application
After running `app.py`, open the provided local URL to interact with EyeBot's interface.

# Usage
- **Image Classification**: Upload an eye image to classify potential diseases.
- **Chatbot Interaction**: Ask questions related to eye diseases, and receive AI-generated responses.

# Demo

[Watch the demo video](https://drive.google.com/uc?export=download&id=14W4QuNkJwPwVCF8oK-fy46PkCzxNVpaq)

# Future Work
- Extend the classification model to include additional eye conditions.
- Expand the chatbot's knowledge base.
- Deploy the system on cloud platforms for enhanced accessibility.

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Acknowledgments
- Thanks to the developers of ResNet, LLaMA, and the various libraries used in this project.
- Special recognition to the AI and healthcare communities for ongoing support and resources.
