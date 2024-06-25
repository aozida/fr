import streamlit as st
import pickle
from fastai.vision.all import *
import pandas as pd
import os
import platform
import pathlib
from litellm import completion
import joblib

# 处理 Windows 和非 Windows 平台的路径问题
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# 加载鲜花特征数据
data = pd.read_excel('flowers_data.xlsx')

def recommend_flowers(selected_data, data):
    # 计算匹配度
    data['match_score'] = selected_data[selected_data.columns[:-2]].min(axis=1)  # 排除 'flower' 和 '花语' 列
    # 按匹配度排序
    sorted_data = data.sort_values(by='match_score', ascending=False)
    return sorted_data

# 加载模型
learn = load_learner('flower_recognition_model.pkl')

# 加载推荐结果
with open('recommended_flowers.pkl', 'rb') as f:
    recommended_flowers = pickle.load(f)

# 图片文件夹路径
image_folder = '图片'

# Streamlit应用程序
st.title('鲜花推荐系统')

# 图像识别模块
st.header('图像识别模块')
with st.expander("查看可识别的花种类"):
    st.write(", ".join(data['flower'].unique()))

uploaded_file = st.file_uploader("选择一张鲜花图片", type=["jpg", "png"])
if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption='上传的图片', use_column_width=True)  # 显示上传的图片
    pred, pred_idx, probs = learn.predict(img)
    flower_info = data[data['flower'] == pred]
    st.write(f'预测结果: {pred}')
    st.write(f'花语: {flower_info["花语"].values[0]}')
    image_path = os.path.join(image_folder, pred, f'{pred}.jpg')  # 假设图片格式为jpg
    st.image(image_path, caption=pred, use_column_width=True)
    st.write(f'概率: {probs[pred_idx]:.2f}')

    # 调用大预言模型描述用户上传的花朵
    response = completion(model='deepseek/deepseek-coder',
                          messages=[
                              {"content": "你是一个优秀的python编辑助手,\n请根据用户上传的花朵,\n描述其特点和花语",
                               "role": "system"},
                              {"content": f'花朵: {pred}\n花语: {flower_info["花语"].values[0]}',
                               "role": "user"}
                          ])
    try:
        description = response['choices'][0]['message']['content']
        st.write('花朵描述:', description)
    except (KeyError, IndexError):
        st.error('描述过程中出现错误，请检查输入或模型是否可用。')

# 特征选择模块
st.header('特征选择模块')
features = st.multiselect('选择特征', data.columns[2:-1])  # 排除 'flower' 和 '花语' 列
if features:
    selected_data = data[features + ['flower', '花语']]
    recommended_flowers = recommend_flowers(selected_data, data)
    st.write('推荐鲜花:')
    for _, row in recommended_flowers[recommended_flowers['match_score'] > 0].iterrows():
        st.write(f'{row["flower"]} - 花语: {row["花语"]}')
        image_path = os.path.join(image_folder, row['flower'], f'{row["flower"]}.jpg')  # 假设图片格式为jpg
        st.image(image_path, caption=row['flower'], use_column_width=True)
        st.write(f'匹配度: {row["match_score"]}')

        # 调用大预言模型描述推荐的花朵
        response = completion(model='deepseek/deepseek-coder',
                              messages=[
                                  {"content": "你是一个优秀的python编辑助手,\n请根据推荐的花朵,\n描述其特点和花语",
                                   "role": "system"},
                                  {"content": f'花朵: {row["flower"]}\n花语: {row["花语"]}',
                                   "role": "user"}
                              ])
        try:
            description = response['choices'][0]['message']['content']
            st.write('花朵描述:', description)
        except (KeyError, IndexError):
            st.error('描述过程中出现错误，请检查输入或模型是否可用。')


from sklearn.impute import SimpleImputer

# KNN推荐系统模块
st.header('KNN推荐系统模块')
knn_model = joblib.load('knn_model.joblib')

# 加载KNN特征数据
knn_data = pd.read_excel('flowers.xlsx', usecols=[0, 1, 2, 3])  # 读取四列数据
print(knn_data.columns)  # 打印列名以确认读取的列数

# 重命名列名
knn_data.columns = ['flower', '花期', '花香浓度', '鲜艳程度']

# 确保特征列是数值型
for feature in [ '花期', '花香浓度', '鲜艳程度']:
    knn_data[feature] = pd.to_numeric(knn_data[feature], errors='coerce')

# 使用SimpleImputer填充NaN值
imputer = SimpleImputer(strategy='mean')  # 使用均值填充NaN值
knn_data[[ '花期', '花香浓度', '鲜艳程度']] = imputer.fit_transform(knn_data[[ '花期', '花香浓度', '鲜艳程度']])

# 选择特征喜好程度
st.subheader('选择特征喜好程度')
feature1_preference = st.slider('花期 喜好程度', 1, 5, 3)
feature2_preference = st.slider('花香浓度 喜好程度', 1, 5, 3)
feature3_preference = st.slider('鲜艳程度 喜好程度', 1, 5, 3)

# 计算匹配度
knn_data['match_score'] = (knn_data['花期'] * feature1_preference +
                           knn_data['花香浓度'] * feature2_preference +
                           knn_data['鲜艳程度'] * feature3_preference)

# 按匹配度排序
sorted_data = knn_data.sort_values(by='match_score', ascending=False)

# 显示推荐结果
st.subheader('推荐鲜花')
for _, row in sorted_data.iterrows():
    st.write(f'{row["flower"]}')
    image_path = os.path.join(image_folder, row['flower'], f'{row["flower"]}.jpg')  # 假设图片格式为jpg
    st.image(image_path, caption=row['flower'], use_column_width=True)
    st.write(f'匹配度: {row["match_score"]}')