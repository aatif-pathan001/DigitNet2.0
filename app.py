
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import time
import pickle
import torch
import torch.nn as nn

#Page Layout is set

st.set_page_config(page_title="DigitNet2.0", layout="wide")
st.title(" DigitNet \n A Handwritten Digit ✏️ Recognition Model")


# function to make image compatible to feed into model

class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet,self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(1,8,5), # 8X28x28 -> 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(2) # 8x24x24 -> 8x12x12
        )

        self.classifier = nn.Sequential(
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10))


    def forward(self,x):
        x = self.Conv1(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

        return x

def process_image(image):
    grayscale_img = image.convert('L')
    grayscale_img = ImageOps.invert(grayscale_img)
    resized_img = grayscale_img.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(resized_img)
    normalized_array = image_array.astype('float32') / 255.0
    
    pro_image = np.where(normalized_array < 0.4, 0, normalized_array)
    pro_image = np.where(pro_image>=0.5,1,pro_image)
    final_array = pro_image.reshape(1, 1, 28, 28)

    return pro_image, final_array

# Loading the model

with open('digitnet3.pkl','rb') as f:
    model_net = pickle.load(f)


def predict(image,model):

    with st.spinner('🤖 Model is thinking...'):
        time.sleep(2)

    im = torch.tensor(image)
    output = model(im)
    _, pred = torch.max(output,1)

    return pred.item()
  

st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # ---x Display Uploaded and Processed Images x---
    try:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file)
        im1 , im2= process_image(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Upload")
            st.image(image, caption="Uploaded Image")

        with col2:
            st.subheader("Model Input")
            st.image(im1, caption="Processed Image")

        # --- Prediction Section ---
        st.header("🧠 Prediction")

        # Add a button to trigger the prediction
        if st.button("Recognize Digit", key="predict_button"):
            # Get the prediction from our placeholder model
            prediction = predict(im2, model_net)

            # Display the prediction in a styled box
            st.success(f"**Recognized Digit:** {prediction}")

    except Exception as e:
        st.error(f"Error: Could not open or process the image. Please upload a valid image file. Details: {e}")

else:
    st.info("Please upload an image using the sidebar to get started.")

# --- Instructions ---
st.sidebar.markdown("---")
st.sidebar.subheader("How to Use")
st.sidebar.info(
    """
    1.  **Upload an image** using the file uploader on the left.
    2.  The uploaded image will be displayed.
    3.  Click the **'Recognize Digit'** button.
    4.  The predicted digit from the model will appear below.
    """
)
