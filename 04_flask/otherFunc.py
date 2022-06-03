def return_hex(user_color_list):
    color_kor_hex = [{'hex': '#FFFFFF', 'name': '화이트'},
                        {'hex': '#9C9C9B', 'name': '그레이'},
                        {'hex': '#000', 'name': '블랙'},
                        {'hex': '#E8416', 'name': '레드'},
                        {'hex': '#F7119E', 'name': '핑크'},
                        {'hex': '#F7441B', 'name': '오렌지'},
                        {'hex': '#FBEA2B', 'name': '옐로우'},
                        {'hex': '#40C1AB', 'name': '민트'},
                        {'hex': '#2AAC14', 'name': '그린'},
                        {'hex': '#5B5A3A', 'name': '카키'},
                        {'hex': '#5BC1E7', 'name': '스카이블루'},
                        {'hex': '#241EFC', 'name': '블루'},
                        {'hex': '#01F62', 'name': '네이비'},
                        {'hex': '#A77BCA', 'name': '라벤더'},
                        {'hex': '#4E86C', 'name': '퍼플'},
                        {'hex': '#76222F', 'name': '와인'},
                        {'hex': '#6C2A16', 'name': '브라운'},
                        {'hex': '#E8C381', 'name': '베이지'},
                        {'hex': '#FFD70', 'name': '골드'},
                        {'hex': '#C0C0C0', 'name': '실버'}]

    unique_hex = [ch['hex'] for user in user_color_list for ch in color_kor_hex if ch['name']==user]

    return unique_hex