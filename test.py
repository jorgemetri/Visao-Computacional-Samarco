import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image

def ExibirInferencia(image_path, output_image_path):
    """
    Realiza a inferência em uma imagem usando YOLOv11, desenha as detecções e salva o resultado.

    Parâmetros:
    - image_path (str): Caminho para a imagem de entrada.
    - output_image_path (str): Caminho para salvar a imagem com as detecções.
    """
    # Caminho padrão do modelo YOLO treinado
    model_path = "runs/detect/train/weights/best.pt"

    # Carregar o modelo YOLOv11
    model = YOLO(model_path)

    # Mapear cores para classes
    class_colors = {
        "capacete": (0, 255, 0),  # Verde
        "colete": (0, 0, 255)    # Vermelho
    }

    # Ler a imagem de entrada
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"A imagem no caminho {image_path} não foi encontrada.")

    # Realizar a inferência na imagem
    results = model.predict(source=image, save=False, conf=0.5)

    # Desenhar as detecções na imagem
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas da caixa delimitadora
        conf = result.conf[0]  # Confiança da detecção
        cls = int(result.cls[0])  # Classe detectada
        class_name = model.names[cls]  # Nome da classe detectada

        # Selecionar cor baseada na classe
        color = class_colors.get(class_name, (255, 255, 255))  # Branco padrão se classe não mapeada

        # Adicionar a caixa delimitadora e o rótulo à imagem
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Caixa delimitadora com cor da classe
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Rótulo

    # Salvar a imagem processada
    cv2.imwrite(output_image_path, image)
    return output_image_path

tab1, tab2,tab3 = st.tabs(["📊 Aplicação", "📥 Imagens Upadas","📥 Imagens Resultado"])
with tab1 :
    # Streamlit interface
    st.title("YOLOv11 - Inferência em Imagem")

        # Input do usuário para a imagem
    uploaded_file = st.file_uploader("Carregar uma imagem para inferência:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Salvar a imagem carregada pelo usuário
        input_image_path = f"./{uploaded_file.name}"
        with open(input_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_image_path = "./resultado.jpg"

        # Realizar a inferência
        processed_image_path = ExibirInferencia(input_image_path, output_image_path)

        # Exibir a imagem processada
    
        

        
with tab2:
    st.write("teste")  
    st.image(Image.open(processed_image_path), caption="Resultado da Inferência", use_container_width=True)  
