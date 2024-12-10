import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import altair as alt
from streamlit_extras.metric_cards import style_metric_cards
import cv2
from ultralytics import YOLO
import os
from PIL import Image, ExifTags
from collections import Counter
@st.fragment
def apagar_imagens(caminho_pasta):
    try:
        # Verifica se o caminho é válido
        if not os.path.exists(caminho_pasta):
            print("O caminho especificado não existe.")
            return
        
        # Lista todos os arquivos na pasta
        arquivos = os.listdir(caminho_pasta)
        
        # Extensões de imagens que deseja apagar
        extensoes_imagens = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        
        # Lista para armazenar os arquivos de imagens encontrados
        imagens_encontradas = []
        
        # Itera sobre os arquivos e verifica as imagens
        for arquivo in arquivos:
            caminho_completo = os.path.join(caminho_pasta, arquivo)
            
            # Verifica se é um arquivo e se possui extensão de imagem
            if os.path.isfile(caminho_completo) and os.path.splitext(arquivo)[1].lower() in extensoes_imagens:
                imagens_encontradas.append(caminho_completo)
        
        # Apaga as imagens encontradas ou retorna mensagem se nenhuma imagem foi encontrada
        if imagens_encontradas:
            for imagem in imagens_encontradas:
                os.remove(imagem)
                print(f"Imagem apagada: {os.path.basename(imagem)}")
            return 1
        else:
            return 0
    
    except Exception as e:
        print(f"Ocorreu um erro: {e}")






def UploadImage():
    # Define o caminho onde as imagens serão salvas
    save_path = "aplication/images_upload/"
    os.makedirs(save_path, exist_ok=True)  # Cria a pasta se não existir

    # Upload de arquivos
    uploaded_files = st.file_uploader(
        "Escolha as imagens", type=['png', 'jpg'], accept_multiple_files=True
    )
    
    col1,col2,col3 = st.columns([1,3,1])
    with col1:
            # Botão para salvar as imagens
        if st.button("Salvar Imagens"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(save_path, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())  # Salva a imagem na pasta
                    st.write(f"Imagem '{uploaded_file.name}' salva com sucesso em '{file_path}'!")
            else:
                st.error("Nenhuma imagem selecionada!")
    with col2:
        if st.button("Limpar Imagens"):
            num1=apagar_imagens("aplication/images_download")
            num2=apagar_imagens("aplication/images_upload")
            if num1+num2 > 0:
                st.success("Imagens apagadas!")
            else:
                st.error("Não existem imagens para serem apagadas!")
    with col3:
        if st.button("Limpar Cache!"):
            st.session_state["datas"] = {}
            st.success("Cache Limpada")
def listar_caminhos_arquivos(pasta):
    """
    Itera pelos arquivos de uma pasta e retorna os caminhos completos.
    
    Args:
        pasta (str): Caminho da pasta a ser analisada.
    
    Returns:
        list: Lista de caminhos completos dos arquivos.
    """
    caminhos = []
    for raiz, _, arquivos in os.walk(pasta):
        for arquivo in arquivos:
            caminho_completo = os.path.join(raiz, arquivo)
            caminhos.append(caminho_completo)
    return caminhos



def TreinarModelo(data,epochs,img):
  model = YOLO("yolo11n.pt")#Carrega um modelo pre-treiando
  results = model.train(data=data, epochs=epochs, imgsz=img)
def ValidacaoModelo(pathmodel):
  model = YOLO(pathmodel)
  #Validando o modelo
  metrics = model.val()
  metrics.box.map  # map50-95
  metrics.box.map50  # map50
  metrics.box.map75  # map75
  metrics.box.maps  # a list contains map50-95 of each category
@st.cache_data
def InferirModelo(pathweights,img,conf):
  model = YOLO(pathweights)
  results = model.predict(source=img,conf=conf)
  return results
    
def Tabela():
    pass


  # Função para corrigir orientação da imagem
def correct_image_orientation(image):
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(orientation, 1)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except AttributeError:
            # Se a imagem não tiver EXIF ou falhar na leitura
            pass
        return image


#Menu de exibição da aplicação-----------------------------------------------------------------
#SideBar()
def ExecutarModeloFotos1(pathimage):
    """
    Realiza a inferência em uma imagem usando YOLOv11, salva o resultado e exibe a imagem no Streamlit.
    """
    # Caminho do modelo YOLO treinado
    model_path = "runs/detect/train/weights/best.pt"

    # Carregar o modelo YOLO
    model = YOLO(model_path)

    # Mapear cores para classes
    class_colors = {
        "capacete": (0, 255, 0),  # Verde
        "colete": (0, 0, 255)    # Vermelho
    }

    # Ler a imagem de entrada
    img = cv2.imread(pathimage)
    if img is None:
        raise FileNotFoundError(f"A imagem no caminho {pathimage} não foi encontrada.")

    # Realizar a inferência
    results = model.predict(source=img, save=False, conf=0.25)

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
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Caixa delimitadora
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Rótulo

    # Preparar o caminho para salvar a imagem processada
    x = pathimage.split("/")[-1]
    output_path = f"aplication/images_download/{x}"

    # Salvar a imagem processada
    cv2.imwrite(output_path, img)
    print(f"Imagem salva em: {output_path}")
#----------------------------------------------------

def ExecutarModeloFotos2(pathimage):
    """
    Realiza a inferência em uma imagem usando YOLOv11, salva o resultado, exibe a imagem no Streamlit
    e exibe a contagem de objetos detectados.
    """
    # Caminho do modelo YOLO treinado
    model_path = "runs/detect/train/weights/best.pt"

    # Carregar o modelo YOLO
    model = YOLO(model_path)

    # Mapear cores para classes
    class_colors = {
        "capacete": (0, 255, 0),  # Verde
        "colete": (0, 0, 255)    # Vermelho
    }

    # Ler a imagem de entrada
    img = cv2.imread(pathimage)
    if img is None:
        raise FileNotFoundError(f"A imagem no caminho {pathimage} não foi encontrada.")

    # Realizar a inferência
    results = model.predict(source=img, save=False, conf=0.25)

    # Contador de classes
    class_count = Counter()

    # Desenhar as detecções na imagem
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Coordenadas da caixa delimitadora
        conf = result.conf[0]  # Confiança da detecção
        cls = int(result.cls[0])  # Classe detectada
        class_name = model.names[cls]  # Nome da classe detectada

        # Incrementar contador de classes
        class_count[class_name] += 1

        # Selecionar cor baseada na classe
        color = class_colors.get(class_name, (255, 255, 255))  # Branco padrão se classe não mapeada

        # Adicionar a caixa delimitadora e o rótulo à imagem
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Caixa delimitadora
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Rótulo

    # Adicionar texto com contagem de objetos no canto inferior esquerdo
    text = f"Quantidade de capacetes: {class_count.get('capacete', 0)}\nQuantidade de coletes: {class_count.get('colete', 0)}"
    y_offset = img.shape[0] - 40
    for i, line in enumerate(text.split("\n")):
        cv2.putText(
            img,
            line,
            (10, y_offset + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # Branco
            1
        )

    # Preparar o caminho para salvar a imagem processada
    x = pathimage.split("/")[-1]
    output_path = f"aplication/images_download/{x}"

    # Salvar a imagem processada
    cv2.imwrite(output_path, img)
    print(f"Imagem salva em: {output_path}")

   

tab1, tab2,tab3 = st.tabs(["📊 Aplicação", "📥 Imagens Upadas","📥 Imagens Resultado"])

with tab1:
    st.title("Aplicação :chart_with_upwards_trend:")
    st.divider()

    with st.container(height=200):
        UploadImage()
    
    
    
    if st.button("Rodar Modelo"):
        pasta = "aplication/images_upload"
        caminhos_dos_arquivos = listar_caminhos_arquivos(pasta)
        array = []
        if len(caminhos_dos_arquivos) > 0:
            for images in caminhos_dos_arquivos:
                ExecutarModeloFotos2(images)
            st.success("Modelo Inferiu em todas as imagens com sucesso!")
        else:
            st.error("Não existem imagens para realizar a inferência!")
    
    #ExibirValores() 
   

    
with tab2:
    st.write("📥 Imagens Upadas")
    pasta = "aplication/images_upload"
    caminhos_dos_arquivos = listar_caminhos_arquivos(pasta)
    array = []
    if len(caminhos_dos_arquivos)> 0:
        for images in caminhos_dos_arquivos:
            image = correct_image_orientation(Image.open(images))
            image = image.resize((image.width, image.height), Image.LANCZOS)
            array.append(image)
    # Exibir as imagens no Streamlit
        caption = []
        for caminho in caminhos_dos_arquivos:
            caption.append(caminho.split("\\")[-1])
        print(array)
        st.image(array, caption=caption, use_container_width=True)
      
with tab3:
    st.write("")
    pasta = "aplication/images_download"
    caminhos_dos_arquivos = listar_caminhos_arquivos(pasta)
    array = []
    if len(caminhos_dos_arquivos) > 0:
        print("Foram adicionadas imagens em download")
        for images in caminhos_dos_arquivos:
            image = correct_image_orientation(Image.open(images))
            image = image.resize((image.width, image.height), Image.LANCZOS)
            array.append(image)
        # Exibir as imagens no Streamlit
        caption = []
        for caminho in caminhos_dos_arquivos:
            caption.append(caminho.split("/")[-1])
        print(array)
        st.image(array, caption=caption, use_container_width=True)
