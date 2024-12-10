import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import string
import time
#Definindo função para pegar dados

#Definindo as pages
def Logo(url):
    st.logo(
        url,
        link="https://innovatechgestao.com.br/",size="large"
    )

st.set_page_config(layout="wide")
modelo1 =  st.Page("model/model.py",title="Modelo",icon=":material/dashboard:")
aplication = st.Page("aplication/aplication.py",title="Aplicação",icon=":material/dashboard:")
test = st.Page("test.py",title="test")

def Logo(url):
    st.logo(
        url,
        link="https://streamlit.io/gallery",size="large"
    )

LOGO_URL_LARGE="./samarco.png"
Logo(LOGO_URL_LARGE)


pg = st.navigation(
    {
        "Aplicação":[aplication],
        "Modelo":[modelo1],

    }
)
pg.run()