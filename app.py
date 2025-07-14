import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# CARREGAR O MODELO TFLITE
@st.cache_resource
def carrega_modelo():
    #https://drive.google.com/file/d/1PlmzkNN6rrNPVghtxVpIIjvSchIu_r4m/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1PlmzkNN6rrNPVghtxVpIIjvSchIu_r4m'

    # Baixar e salvar localmente
    gdown.download(url, 'modelo_animais_final.tflite', quiet=False)

    # Carregar modelo TFLite
    interpreter = tf.lite.Interpreter(model_path='modelo_animais_final.tflite')
    interpreter.allocate_tensors()
    return interpreter

# CARREGAR IMAGEM DO USUÁRIO
def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste ou clique para enviar um desenho', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert('L')  # Grayscale
        image = image.resize((28, 28))  # Redimensionar para 28x28

        st.image(image, caption='Imagem carregada', width=150)
        st.success('Imagem carregada com sucesso!')

        # Pré-processamento
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.reshape(1, 28, 28, 1)
        return image

# FAZER PREVISÃO
def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    st.write("Input Details:", input_details)
    st.write("Output Details:", output_details)


    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['Rinoceronte', 'Panda', 'Papagaio', 'Porco', 'Coelho']

    df = pd.DataFrame({
        'Animal': classes,
        'Probabilidade (%)': (output_data[0] * 100).round(2)
    })

    fig = px.bar(
        df.sort_values('Probabilidade (%)'),
        y='Animal',
        x='Probabilidade (%)',
        orientation='h',
        text='Probabilidade (%)',
        title='Previsão de Animal (QuickDraw)'
    )

    st.plotly_chart(fig)

# APP PRINCIPAL
def main():
    st.set_page_config(page_title="Classificador de Desenhos (Animais)")

    st.write("# 🐾 Classificador de Desenhos - Animais")
    st.write("Desenhe ou envie uma imagem e descubra qual animal o modelo reconhece!")

    interpreter = carrega_modelo()
    image = carrega_imagem()

    if image is not None:
        previsao(interpreter, image)

if __name__ == "__main__":
    main()
