# 모듈 호출
import cv2
import os
import pandas as pd
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from cloth_image_preprocessing import image_preprocess
from werkzeug.utils import secure_filename
from recommend import *
from datetime import datetime
import pyautogui
from otherFunc import return_hex

app = Flask(__name__)  # 플라스크 인스턴스 생성

@app.route('/')
@app.route('/Main')  # 기본 홈 경로 설정
def Main():  # 경로에 대한 요청이 있을 때 실행될 함수 정의
    return render_template('Main.html')  # 저장된 html 템플릿 렌더링

@app.route('/Aicodi')
def Aicodi():
    return render_template('Aicodi.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/productDetail')
def productDetail():
    return render_template('productDetail.html')

@app.route('/productDetailTemp')
def productDetailTemp():
    return render_template('productDetailTemp.html')

@app.route('/recommendMatch')
def recommendMatch():
    return render_template('recommendMatch.html')

@app.route('/Mycloset')
def Mycloset():
    user = pd.read_pickle('./static/data/user.pkl')
    if user.empty:
        return render_template('Mycloset.html')
    else:
        season_dict = {'봄': 0, '여름': 1, '가을': 2, '겨울': 3}
        season_list = set(str([ss.replace(' ', ',') for ss in user['season']]).translate(str.maketrans("[ ]'", '    ')).replace(' ', '').split(','))
        unique_season = sorted(list(set(season_list)), key=lambda x: season_dict[x])
        unique_color = list(set(user['color']))
        color_hex = return_hex(unique_color)

        return render_template('Mycloset.html',
                               unique_season=unique_season,
                               unique_color=unique_color,
                               color_hex=color_hex,
                               fname=list(user.fname),
                               cloth_cat=list(user.cloth_cat),
                               color=list(user.color),
                               season=list(user.season),
                               favor=list(user.favor),
                               description=list(user.description))
@app.route('/StoreCloset')
def StoreCloset():
    return render_template('StoreCloset.html')
# @app.route('/StoreCloset')
# def StoreCloset():
#     store = pd.read_pickle('./static/data/store.pkl') # 추후 sql 구문으로 변경
#     season_dict = {'봄': 0, '여름': 1, '가을': 2, '겨울': 3}
#     unique_season = sorted(list(set(store['season'])), key=lambda x: season_dict[x])
#     unique_color = list(set(store['color']))
#     color_hex = return_hex(unique_color)
#
#     return render_template('StoreCloset.html',
#                            unique_season=unique_season,
#                            unique_color=unique_color,
#                            color_hex=color_hex,
#                            fname=list(store.fname),
#                            cloth_cat=list(store.cloth_cat),
#                            color=list(store.color),
#                            season=list(store.season),
#                            favor=list(store.favor),
#                            cost=list(store.cost),
#                            count=list(store.count),
#                            description=list(store.description))

@app.route('/StoreproductDetailTemp')
def StoreproductDetailTemp():
    return render_template('StoreproductDetailTemp.html')

@app.route('/modifystorecloth', methods=['POST'])
def modifystorecloth():
    # 받아온 데이터의 정보로 기존 데이터 정보 수정하는 코드 짜야함 last_save_date
    data = request.get_json()
    print(data)

    # print(user.drop_duplicates(['fname'], keep='first'))
    return jsonify(result="success", result2=data)

@app.route('/storeclothadd', methods=['POST'])
def storeclothadd():
    # 받아온 데이터의 정보로 기존 데이터 정보 수정하는 코드 짜야함 last_save_date
    data = request.get_json()
    print(data)

    # print(user.drop_duplicates(['fname'], keep='first'))
    return jsonify(result="success", result2=data)

@app.route('/ajax', methods=['POST'])
def ajax():
    # 받아온 데이터의 정보로 기존 데이터 정보 수정하는 코드 짜야함 last_save_date
    data = request.get_json()
    print(data)

    # print(user.drop_duplicates(['fname'], keep='first'))
    return jsonify(result="success", result2=data)

@app.route('/productDetail_upload', methods = ['POST', 'GET'])
def productDetail_upload():
    if request.method == 'POST':
        f = request.files['file']
        fname = secure_filename(f.filename) # 파일명을 보호하기 위한 함수
        f.save('./static/images/cloths/origin_' + fname +'.png')  # 지정된 경로에 파일 저장
        color_name = image_preprocess(fname)

        model = load_model('./static/model/MobileNet.h5')
        img = cv2.imread('./static/images/cloths/pp_' + fname + '.png')  # 저장한 이미지 불러오기
        img = cv2.resize(img, (224, 224)) / 255.0  # 모델에 맞는 input_shape로 리사이즈
        img = img.reshape((1,) + img.shape)  # 입력 데이터로 사용하기 위해 데이터 reshape
        pred = model.predict(img)

        # 인덱스로 의류 카테고리 추출
        class_dict = {0:'Bottom', 1:'One-Piece', 2:'Outer', 3:'Top'}

        pred_class = class_dict[np.argmax(pred, axis=1)[0]]

        # 모든 예측 로직 사용이 끝났으므로 원본파일 삭제
        os.remove(f'./static/images/cloths/origin_' + fname +'.png')
        # 가입조건에 따른 조건문 작성 필요
        # 현재 사용자 기준
        return render_template('productDetailTemp.html',
                               fname=f'pp_{fname}.png',
                               color_cat=color_name,
                               cloth_cat=pred_class)
        # 판매자
        # return render_template('StoreproductDetailTemp.html',
        #                        fname=f'pp_{fname}.png',
        #                        color_cat=color_name,
        #                        cloth_cat=pred_class)

@app.route('/clothadd', methods=['POST'])
def clothadd():
    data = request.get_json()

    user = pd.read_pickle('./static/data/user.pkl')

    input_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    user_cloth_no = f"{data['cloth_cat']}_{datetime.now().strftime('%m%d%Y%H%M%S')}"

    new_data = {
        'user_cloth_no': user_cloth_no,
        'fname': data['fname'],
        'cloth_cat': data['cloth_cat'],
        'color': data['color'],
        'season': str(data['season']).translate(str.maketrans("[ ]'", '    ')).replace(' ','').replace(',',' '),
        'favor': data['favor'],
        'description': data['description'],
        'input_date': input_date,
        'last_save_date': input_date
    }

    user = user.append(new_data, ignore_index=True)
    user.to_pickle('./static/data/user.pkl')

    return jsonify(result="success", result2=data)

@app.route('/recommend', methods = ['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        temperature = int(float(request.form['temperature']))
        situation = request.form['situation']
        keyword = request.form.getlist('keyword')

        ### 상황, 온도, 스타일 키워드 선택에 따른 데이터셋 생성
        data = pd.read_pickle('./static/data/data.pkl')
        # 상황 선택하기
        pyautogui.PAUSE = 1
        select_data = select_condition(data, situation)

        # 온도 선택하기
        pyautogui.PAUSE = 5
        select_data = select_temperature(select_data, temperature)

        # 스타일 키워드 선택하기
        pyautogui.PAUSE = 5
        select_data = select_style(select_data, keyword, situation)

        # 추천해주기
        pyautogui.PAUSE = 15
        user = pd.read_pickle('./static/data/user.pkl')
        # 선택한 온도 반영한 사용자 데이터 출력하기
        user = return_user_temp(user,  temperature)
        select_data = create_user_faver(select_data, user)
        select_data = pre_processing_adjmatrix(select_data)
        adj_matrix = make_adj_matrix(select_data)

        recommend_data = make_recommend_list(adj_matrix, select_data)

        # 추천리스트와 사용자 비교하여 보여주기
        pyautogui.PAUSE = 10
        match_path, match_fail_path = return_match_path_list(recommend_data, user, data)

        return render_template('recommendMatch.html',
                                fileID_onCloset_path_list=match_path,
                                fileID_onStore_path_list=match_path)


if __name__ != '__main__':
    pass
else:
    app.run(debug=True)
