
def image_preprocess(target_item):
    color_name = ['화이트', '그레이', '블랙', '레드', '핑크', '오렌지', '옐로우', '민트', '그린', '카키', '스카이블루', '블루', '네이비', '라벤더', '퍼플', '와인', '브라운', '베이지', '골드',  '실버']

    color_chip_hsv = [[160, 0, 240], [40, 1, 146],
                         [160, 0, 0], [237, 232, 111], [215, 224, 124], [7, 224, 129],
                         [37, 231, 138], [113, 122, 121], [74, 190, 90], [39, 53, 70],
                         [131, 179, 152], [161, 234, 133], [147, 240, 46], [182, 102, 153],
                         [188, 207, 55], [234, 133, 72], [22, 121, 57], [26, 166, 170]]

    color_chip_rgb = [[255, 255, 255], [156, 156, 155], [0, 0, 0], [232, 4, 22],
                         [247, 17, 158], [247, 68, 27], [251, 234, 43], [64, 193, 171],
                         [42, 172, 20], [91, 90, 58], [91, 193, 231], [36, 30, 252],
                         [0, 31, 98], [167, 123, 202], [78, 8, 108], [118, 34, 47],
                         [108, 42, 22], [232, 195, 129], [255, 215, 0], [192, 192, 192]]



    # image에 target_item을 받아와야 로직 실행이 가능
    # image = cv2.imread(os.path.join(file_path , str(target_item) + '.jpg'), cv2.IMREAD_UNCHANGED)
    filepath = './static/cloths/origin_' + target_item +'.png'
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    ## 외곽선 검출 및 배경 제거
    blur = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(blur, 10, 250)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 경계선 내부 255로 채우기
    height, width, channel = image.shape
    mask = np.zeros((height, width, 4), np.uint8)
    cv2.fillPoly(mask, contours, (255,) * image.shape[2], )
    masked_image = cv2.bitwise_and(image, mask)
    new_img = cv2.bitwise_and(image, mask)

    ## trim
    contours_xy = np.array(contours)

    x_min, x_max = 0, 0
    y_min, y_max = 0, 0
    xs = []
    ys = []
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            xs.append(contours_xy[i][j][0][0])
            ys.append(contours_xy[i][j][0][1])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    img_trim = new_img[y:y + h, x:x + w]

    # 정방형으로 만들기
    height, width, channel = img_trim.shape
    size = max(height, width)
    resize_image = np.zeros((size, size, 4), np.uint8)
    if height == size:
        diff = (height // 2) - (width // 2)
        for i in range(size):
            for j in range(size):
                if (j > diff) and (j < diff + width):
                    resize_image[i][j] = img_trim[i][j - diff]
    else:
        diff = (width // 2) - (height // 2)
        for i in range(size):
            for j in range(size):
                if (i > diff) and (i < diff + height):
                    resize_image[i][j] = img_trim[i - diff][j]

    thumbnail = cv2.resize(resize_image, dsize=(300, 300), interpolation=cv2.INTER_AREA)
    new_img = Image.fromarray(thumbnail)
    new_img.save('./static/cloths/pp_' + target_item +'.png')

    return return_colorname(return_match_rgb('./static/cloths/pp_' + target_item +'.png'))
