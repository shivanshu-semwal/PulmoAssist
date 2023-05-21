import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os

interpreter = None
labels = {
    0: 'Covid',
    1: 'Normal',
    2: 'TB',
    3: 'Pneumonia'
}


def input_covid_classifier():
    """
      function to read the model from disk
    """
    global interpreter
    interpreter = tf.lite.Interpreter(
        model_path=os.path.join(os.getcwd(), 'model/quantized_model.tflite'))
    interpreter.allocate_tensors()


def predict(image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = image.convert("RGB")
    image = image.resize((256, 256))
    img = np.array(image, dtype='float32')
    # img = img / 255
    img = img.reshape((1, 256, 256, 3))

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(predictions[0])
    result = {
        'class': labels[pred],
        'class_probability': np.round(predictions[0][pred] * 100, 2)
    }
    chart_data = pd.DataFrame(
        predictions[0].tolist(),
        labels.values()
    )
    return (result, chart_data)


if __name__ == '__main__':
    st.sidebar.header("PulmoAssist")
    st.sidebar.markdown(
        "PulmoAssist uses a CNN to detect COVID-19, Pneumonia, TB in Chest X-Ray Images with an accuracy of 95%.")
    st.sidebar.image('img/sidebar.jpeg', use_column_width=True)
    st.sidebar.subheader("Upload an image to get a diagnosis")
    st.sidebar.write('Made with ❤ by Shivanshu')
    file_uploaded = st.file_uploader(
        "Choose the Image File", type=['jpg', 'jpeg', 'png'])

    if file_uploaded is not None:
        if interpreter is None:
            input_covid_classifier()
        image = Image.open(file_uploaded)
        result, chart_data = predict(image)

        col1, col2 = st.columns(2)
        col2.image(image, caption="The image is classified as " + result['class'], width=300)
        col1.header("Classification Result")
        col1.write("The image is classified as " + result['class'])
        col1.write("The class probability is " +
                   str(result['class_probability']) + "%")

        st.bar_chart(chart_data)
