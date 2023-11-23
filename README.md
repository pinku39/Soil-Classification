# Soil Type Classification using Deep Learning 🌱🔍

This project classifies soil types using deep learning techniques. The model predicts soil type from images.



## Install dependencies:
pip install -r requirements.txt 
git clone https://github.com/pinku/soil-type-classification.git
cd soil-type-classification


## Usage
## Dataset Preparation
The dataset contains images of various soil types.
Organize the data into folders for each soil type:
Black Soil
Cinder Soil
Laterite Soil
Peat Soil
Yellow Soil
## Training the Model
Run the train_model.py script to train the deep learning model.
Adjust hyperparameters in the script as needed.
## Evaluating the Model
Evaluate the model's performance using the test dataset.
## Making Predictions
Use the trained model to make predictions on new images.
Run predict_soil_type.py and provide the path to the image.
## Model Details
The model architecture is a Convolutional Neural Network (CNN).
It consists of several convolutional and pooling layers followed by dense layers.
The model uses the RMSprop optimizer and categorical cross-entropy loss function.
## Results
Training Accuracy :96%

## Figure: Training accuracy with epochs

## File Structure
train_model.py: Script for training the model
predict_soil_type.py: Script to predict soil type from an image
requirements.txt: List of Python dependencies
images/: Folder containing images used in the README
Contributing
Feel free to contribute by forking the repository and submitting pull requests. Please open an issue first to discuss significant changes.

## Credits
The dataset used in this project is sourced from [kaggel]
## License
This project is licensed under [MIT]. See the LICENSE file for details.
