# 모듈 불러오기
import numpy as np
import pandas as pd

"""## 함수 정의 ##"""

# 선택한 상황을 반영하여 데이터셋 반환하는 함수
def select_condition(data, select_situation):
    if int(select_situation) == 1:
        condition = (data['situation'] == '공')
    elif int(select_situation) == 2:
        condition = (data['situation'] == '사')
    elif int(select_situation) == 3:
        condition = (data['situation'] == '운동')
    elif int(select_situation) == 4:
        condition = (data['situation'] == '기타')
    else:
        print('1 - 4 번 상황 중 선택해주세요')
        return None
    data = data[condition]
    return data

# 온도에 따라 fabric, cloth_cat, arm_length 조건 반영하여 데이터셋 반환하는 함수
def select_temperature(data, temperature):
    if temperature < 5:
        fabric_list = ['가죽', '울캐시미어', '니트', '퍼', '패딩', '벨벳', '코듀로이','플리스', '무스탕', '자카드', '데님','스웨이드','헤어']
        cloth_cat_list = ['Top', 'Bottom','Outer', 'One-Piece']
        arm_length_list = ['긴팔',None]
    elif temperature in range(5,9):
        fabric_list = ['데님', '니트', '스웨이드', '플리스', '트위드','울캐시미어','퍼', '벨벳', '스웨이드','코듀로이','헤어']
        cloth_cat_list = ['Top', 'Bottom','Outer', 'One-Piece']
        arm_length_list = ['긴팔',None]
    elif temperature in range(9,12):
        fabric_list = ['우븐', '가죽', '데님', '니트', '벨벳', '스웨이드','코듀로이','트위드','헤어']
        cloth_cat_list = ['Top', 'Bottom','Outer', 'One-Piece']
        arm_length_list = ['긴팔',None]
    elif temperature in range(12,17):
        fabric_list = ['우븐', '데님', '시폰', '레이스', '실크', '니트','시폰']
        cloth_cat_list = ['Top', 'Bottom','Outer', 'One-Piece']
        arm_length_list = ['긴팔',None]
    elif temperature in range(17,20):
        fabric_list = ['우븐', '가죽', '데님', '네오프렌','실크','시폰']
        cloth_cat_list = ['Top', 'Bottom','Outer', 'One-Piece']
        arm_length_list = ['7부소매','긴팔','반팔',None]
    elif temperature in range(20,23):
        fabric_list = ['린넨', '데님','시폰']
        cloth_cat_list = ['Top', 'Bottom', 'Outer', 'One-Piece']
        arm_length_list = ['7부소매','긴팔','반팔',None]
    elif temperature in range(23,28):
        fabric_list = ['린넨', '데님','메시','스판덱스']
        cloth_cat_list = ['Top', 'Bottom', 'One-Piece']
        arm_length_list = ['민소매','반팔','캡',None]
    elif temperature >= 28:
        fabric_list = ['린넨', '데님','메시','스판덱스']
        cloth_cat_list = ['Top', 'Bottom', 'One-Piece']
        arm_length_list = ['민소매','반팔','캡',None]
    else:
        print('잘못된 온도를 입력하셨습니다. 다시 입력해주세요.')
        return None

    data = data[data['fabric'].map(lambda x: any(x in select_fabric for select_fabric in fabric_list))]
    data = data[data['cloth_cat'].isin(cloth_cat_list) & data['arm_length'].isin(arm_length_list)]
    return data

# 선택한 상황에 따른 스타일 키워드 리스트를 반환하는 함수
def return_style_keyword(select_situation):
    # 스타일 키워드 리스트
    public_list=['개성있는', '고전적', '남성적', '단순한', '도시적', '독특한', '모던함', '미래지향적', '산뜻함', '세련된', '여성스러운','우아함', '중성적', '지적인', '편견없는', '편안한', '혁신적', '현대적인']
    other_list=['고급스러운', '고전적인', '다채로운', '레트로', '복고풍', '세련된', '슬림한', '자유분방한', '장식이과한', '화려한']
    privacy_list=['귀여운', '내추럴한', '단정한', '도시적', '독특한', '러블리한', '반항적인', '부드러운', '사랑스러운', '세련된', '소녀적인', '소박한', '야성적', '여성스러운', '자유분방한', '차분한', '캐주얼한', '편한', '현대적인', '힙한']

    if int(select_situation) == 1: # public
        return public_list
    elif int(select_situation) == 2:   # privacy
        return privacy_list
    elif int(select_situation) == 4:   # other
        return other_list

# 선택한 상황에 따라 스타일 딕셔너리를 반환하는 함수
def return_situation_style_keyword(select_situation):
    # 상황별 스타일 키워드 딕셔너리
    public_style = dict(매니시=['남성적', '중성적', '우아함', '산뜻함'],
                        모던=['도시적', '미래지향적', '모던함'],
                        소피스트케이티드=['세련된', '도시적', '지적', '여성스러운'],
                        아방가르드=['독특한', '혁신적'],
                        젠더리스=['중성적인', '개성있는', '편견없는'],
                        클래식=['고전적', '세련된', '우아함'],
                        톰보이=['남성적', '편안한', '단순한', '현대적인'])

    other_style = dict(레트로=['복고풍', '레트로'],
                    오리엔탈=['화려한', '고급스러운', '다채로운', '고전적인'],
                    키치=['장식이과한',  '자유분방한'],
                    히피=['자유분방한', '세련된', '화려한', '슬림한'])

    privacy_style = dict(로맨틱=['여성스러운', '귀여운', '사랑스러운', '소녀적인'],
                        스트리트=['현대적인', '자유분방한', '힙한'],
                        웨스턴=['도시적', '세련된', '야성적'],
                        컨트리=['내추럴한', '편한', '소박한', '소녀적인'],
                        펑크=['반항적인', '자유분방한', '독특한'],
                        페미닌=['여성스러운', '부드러운', '러블리한', '차분한'],
                        프레피=['캐주얼한', '단정한', '세련된'])

    if int(select_situation) == 1:      # public
        return public_style
    elif int(select_situation) == 2:   # privacy
        return privacy_style
    elif int(select_situation) == 4:   # other
        return other_style

# 상황별 스타일 키워드 선택에 따라 점수(가중치)을 반영한 데이터셋 반환하는 함수
def select_style(data, select_style_list, select_situation):
    # 선택한 상황에 따른 스타일 전체 키워드 리스트 생성
    style_keyword_list = return_style_keyword(select_situation)

    # 선택한 상황에 따른 스타일 키워드 딕셔너리 생성
    situation_style = dict(return_situation_style_keyword(select_situation))

    # 선택한 스타일 키워드에 따른 매트릭스 생성
    select_style_list = [int(i) for i in str(select_style_list).replace('[', "").replace(']', "").replace("'", "").split(', ')]
    style_select_metrix = pd.DataFrame(np.zeros(shape=(len(data.fileID.unique()), len(style_keyword_list))))
    style_select_metrix['fileID'] = data.fileID.unique()

    # 선택한 키워드 : 1 / 미선택 키워드 : 0 반영
    for select in select_style_list:
        style_list = [k for k, v in situation_style.items() if style_keyword_list[select] in v]
        fileID_list = data.loc[data['style'].isin(style_list), 'fileID'].unique()
        style_select_metrix.loc[style_select_metrix.fileID.isin(fileID_list), select] += 1.

    # 스타일링 파일마다 선택된 키워드들의 합산한 값만 저장
    style_select_metrix['score'] = style_select_metrix.loc[:, select_style_list].sum(axis=1)
    style_select_metrix = style_select_metrix.loc[:, ['fileID', 'score']]

    # 키워드 점수를 원본데이터와 merge
    data = pd.merge(left=data, right=style_select_metrix, how="inner", on="fileID")

    # 최종적으로 사용할 컬럼만 반영하여 반환
    return data.loc[:, ['fileID', 'situation', 'style', 'cloth_cat', 'color', 'score', 'clothID']]

# 사용자 선호도 반영한 데이터셋 반환하는 함수
def create_user_faver(data, user_data):
    for user_cloth, user_color, user_score in zip(user_data.cloth_cat, user_data.color, user_data.favor):
        data.loc[((data.cloth_cat == user_cloth) & (data.color == user_color)), 'score'] += float(user_score)
    return data

def pre_processing_adjmatrix(data):
    # fileID별 중복이름(cloth_cat 다름) 제거를 위해 index 재설정
    df_name = pd.DataFrame(data.fileID.unique(), columns=['fileID'])
    df_name.reset_index(inplace=True)
    data = pd.merge(left=data, right=df_name, how="inner", on="fileID")

    # array 위치값으로 사용하기 위한 clothID index 재설정
    i = 0
    for id in data.clothID.unique():
        data.loc[data.clothID==id, 'clothID'] = i
        i+=1

    return data.loc[:, ['fileID', 'situation', 'style', 'cloth_cat', 'color', 'score', 'index', 'clothID']]

# 사용자 데이터 중 선택된 온도에 따라 데이터셋 반환하기
def return_user_temp(user_data, temperature):
    if float(temperature) < 10:
        return user_data[user_data['season'].str.contains('겨울')]
    elif float(temperature) in range(10,24):
        return user_data[(user_data['season'].str.contains('봄'))|(user_data['season'].str.contains('가을'))]
    elif float(temperature) >= 24:
        return user_data[user_data['season'].str.contains('여름')]

# 인접행렬 생성하기
def make_adj_matrix(data):
    data_array = np.array([[int(fileid), int(clothID), int(score), int(fname)] for fileid, clothID, score, fname in zip(data.index, data.clothID, data.score, data.fileID)])

    data_array[:,0] -= 1 # 0부터 시작하기 위해 1를 뺌
    fileid=int(max(data_array[:, 0])) # fileID 총 갯수
    clothID= int(max(data_array[:, 1]))#  clothID의 총 갯수
    shape = (clothID+1, fileid+1) # 전체 매트릭스 크기 확인

    adj_matrix=np.zeros(shape, dtype=int)
    for fileid, clothid, score, fname in data_array:
        adj_matrix[int(clothid)][int(fileid)] = int(score)

    return adj_matrix

# 코사인 유사도 구하기
def compute_cos_similarity(v1, v2):
  norm1 = np.sqrt(np.sum(np.square(v1)))
  norm2 = np.sqrt(np.sum(np.square(v2)))
  dot = np.dot(v1, v2)
  return (dot/(norm1*norm2))

# 코사인 유사도를 계산하여 최종 TOP10 리스트 반환
# 거리가 가까울 수록(값이 작을 수록) 선택한 스타일 조건과 유사한 사용자

def make_recommend_list(adj_matrix, select_data):
    my_id, my_vector=0, adj_matrix[0]
    best_match, best_match_id, best_match_vector = 9999, -1, []

    for clothid, fileid in enumerate(adj_matrix):
        if my_id != clothid:
            cos_similarity = compute_cos_similarity(my_vector, fileid)
            if cos_similarity < best_match:
                best_match=cos_similarity
                best_match_id=clothid
                best_match_vector=fileid
    #print('best_match:',best_match, ',best_match_id:',best_match_id)

    recommend_list = []
    for i, log in enumerate(zip(my_vector, best_match_vector)):
        log1, log2 = log
        if log1 < 1. and log2 > 0.:
            recommend_list.append(i)

    return select_data[(select_data['index'].isin(list(recommend_list)))].sort_values('score', ascending=False).drop_duplicates('clothID')[:10]

# 최종데이터와 유저데이터 비교하여 옷장안의 유무 리스트(파일명) 반환
def make_match_path(rec_cat, recommend_data, user_data, origin_data): # store_data 입력값 추가 필요
    name_list = []

    for cat in rec_cat:
        final_data = origin_data[(origin_data.fileID.isin(recommend_data['fileID'])) & (origin_data.cloth_cat==cat)]

        match_list = []
        match_fail_list = []

        for fcat, fcolor, fname in zip(final_data.cloth_cat, final_data.color, final_data.fileID):
            if user_data[(user_data.cloth_cat==fcat)&(user_data.color==fcolor)].empty==False:
                match_list.append(fname)
            else:
                match_fail_list.append(fname)

        # 우선순위 반영하여 재정렬
        globals()['match_{}'.format(cat)] = [rec_fid for rec_fid in recommend_data.fileID for match in match_list if match==rec_fid]
        globals()['match_fail_{}'.format(cat)] = [rec_fid for rec_fid in recommend_data.fileID for matchfail in match_fail_list if matchfail==rec_fid]

        name_list.append([f'match_{cat}', f'match_fail_{cat}'])

    # 의류 카테고리별 조합하기
    match_list = []
    for i in range(0, min([len(globals()[a]) for a in [name_list[k][0] for k in range(len(rec_cat))]])):
        match_list.append([globals()[a][i] for a in [name_list[s][0] for s in range(len(rec_cat))]])

    match_fail_list = []
    for i in range(0, min([len(globals()[a]) for a in [name_list[k][1] for k in range(len(rec_cat))]])):
        match_fail_list.append([globals()[a][i] for a in [name_list[s][1] for s in range(len(rec_cat))]])

    # 조합한 의류데이터 순으로 의류 경로 리스트 제작하기
    match_path_list = []
    for match in match_list[:3]:
        match_df = pd.DataFrame()
        for i in range(0, len(rec_cat)):
            match_df = pd.concat([match_df, origin_data[(origin_data.fileID==match[i]) & (origin_data.cloth_cat==rec_cat[i])]])
        try: match_path_list.append([f'cloths/{str(user_data[(user_data.cloth_cat==mtcat)&(user_data.color==mtcolor)].fname.unique()[0])}' for mtcat, mtcolor in zip(match_df.cloth_cat, match_df.color)])
        except: pass


    match_fail_path_list = []
    # for match_fail in match_fail_list[:3]:
    #     match_fail_df = pd.DataFrame()
    #     for i in range(0, len(rec_cat)):
    #         match_fail_df = pd.concat([match_fail_df, origin_data[(origin_data.fileID==match_fail[i]) & (origin_data.cloth_cat==rec_cat[i])]])
    #     try: match_path_list.append([f'cloths/{str(store_data[(store_data.cloth_cat==mtfcat)&(user_data.color==mtfcolor)].fname.unique()[0])}' for mtfcat, mtfcolor in zip(match_fail_df.cloth_cat, match_fail_df.color)])
    #     except: pass
    #
    return match_path_list, match_fail_path_list

def return_match_path_list(recommend_data, user_data, origin_data): # store_data 입력값 추가 필요
    rec_cat = list(recommend_data.cloth_cat.unique())
    match_path_list = []
    match_fail_path_list = []

    # 원피스의 경우 상의,하의를 대신할 수 있으므로 조건문 처리 진행
    if ('One-Piece' in rec_cat):  # 원피스가 있고
        if ('Outer' in rec_cat):  # 아우터가 있는 경우
            a, b = make_match_path(['Outer', 'One-Piece'], recommend_data, user_data, origin_data)
            rec_cat.remove('One-Piece')  # 불필요한 컬럼 삭제
            match_path_list, match_fail_path_list = make_match_path(rec_cat, recommend_data, user_data, origin_data)

            match_path_list = match_path_list + a
            match_fail_path_list = match_fail_path_list + b
        else:  # 아우터가 없는 경우
            a, b = make_match_path('One-Piece', recommend_data, user_data, origin_data)
            rec_cat.remove('One-Piece')  # 불필요한 컬럼 삭제
            match_path_list, match_fail_path_list = make_match_path(rec_cat, recommend_data, user_data, origin_data)

            match_path_list = match_path_list + a
            match_fail_path_list = match_fail_path_list + b
    else:  # 원피스가 없는 경우
        match_path_list, match_fail_path_list = make_match_path(rec_cat, recommend_data, user_data, origin_data)

    return match_path_list, match_fail_path_list