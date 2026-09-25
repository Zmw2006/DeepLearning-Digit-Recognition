import streamlit as st
from PIL import Image
from utils.inference import load_model, predict

st.set_page_config(page_title="手写数字识别", page_icon="✍️")
st.title("手写数字识别")
st.write("上传一张只含一个数字的图片。清晰、居中的数字效果更好。")

@st.cache_resource
def cached_model():
    return load_model()

file = st.file_uploader("选择 PNG/JPG 图片", type=["png", "jpg", "jpeg"])
if file:
    try:
        with Image.open(file) as source:
            image = source.copy()
        st.image(image, caption="上传的图片", width=220)
        digit, confidence = predict(image, cached_model())
        st.success(f"预测数字：{digit}")
        st.write(f"模型置信度：{confidence:.2%}")
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as error:
        st.error(str(error))
