#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '15YaKTONZlz1bSJKaQJs-_FB0NIWuCEat'


# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #F3F2F2; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #F781BE; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #FE2E64; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_column_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
         'images': [
            "https://i.ibb.co/FW8FLkG/2.jpg",
            "https://i.ibb.co/z4BRyWV/1.jpg",
            "https://i.ibb.co/Vv5wQjb/image.jpg"
        ],
             
        'videos': [
            "https://youtu.be/gxDmLEpEqAU?feature=shared",
            "https://youtu.be/HpY_tx-o_wY?feature=shared",
            "https://youtu.be/IPFpLwKiHtY?feature=shared"
        ],
        'texts': [
            "수질관리: 물은 일반적으로 매주 30% 정도의 물을 교체하는 것이 좋다. 물의 온도는 24도에서 26도 사이가 적절하며, pH는 6.8에서 7.8 사이를 유지해야 한다.",
            "먹이관리: 초기에는 구피 전용 사료를 주는 것이 좋으며, 여기에 냉동 새우나 브라인 쉬림프를 추가하면 더욱 좋다. 하루에 1~2회, 1분 이내에 소화할 수 있는 양으로 조절해야 한다.",
            "번식관리: 일반적으로 암컷 2~3마리에 대해 수컷 1마리가 적당하다. 새끼가 태어난 후에 다른 구피들에게 잡아먹힐 수 있으므로, 새끼를 따로 분리해 주는 것이 좋다."
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/Xysk8dc/2.jpg",
            "https://i.ibb.co/PCZRRQk/1.jpg",
            "https://i.ibb.co/NLkPPnG/image.jpg"
        ],
        'videos': [
            "https://youtu.be/a2FQRmWYaWo?feature=shared",
            "https://youtu.be/PPK-skmAqJI?feature=shared",
            "https://youtu.be/fGuIfABYimc?feature=shared"
        ],
        'texts': [
            "수질관리: 수온 22도에서 28도, pH 범위 6.0에서 7.5에서 잘 자란다.",
            "먹이관리: 네온 테트라는 혼합식을 먹는다. 작은 입자 형태의 플레이크나 그래뉼 타입의 사료가 이상적이며 주기적으로 아르테미아나 브라인 새우와 같은 살아 있는 먹이를 제공하면 네온 테트라의 건강에 좋다.",
            "번식관리: 번식이 까다로운 편으로, 어두운 환경과 적절한 수온, pH조절이 필요하다. 수정된 알은 수족관 바닥에 떨어지며, 부모가 알을 먹지 않도록 분리해주는 것이 좋다."
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/X3zPPhX/2.jpg",
            "https://i.ibb.co/Wg0dwX1/1.jpg",
            "https://i.ibb.co/mqPrL6Y/image.jpg"
        ],
        'videos': [
            "https://youtu.be/j3BZ8qEi8L4?feature=shared",
            "https://youtu.be/WB524aPvlvw?feature=shared",
            "https://youtu.be/-wTqziGbiLM?feature=shared"
        ],
        'texts': [
            "수질관리: 26도에서 잘 살 수 있는 열대어로, 공기호흡을 하러 수면 위로 올라간다. 점프를 자주 하므로 어항의 수위는 80%만 채워두는 것이 좋다.",
            "먹이관리: 물에 뜨는 플레이크 타입의 사료가 좋다. 어항에 적응이 된 베타는 일반 사료도 대부분 잘 먹고, 어항 안에 새우가 있다면 잡아먹기도 한다.",
            "번식관리: 투쟁심이 강해 보통 한 마리 단독 사육을 하며 산란기에만 암컷과 수컷을 합사시켜준다."
        ]
    },
      labels[3]: {
        'images': [
            "https://i.ibb.co/x3BydvS/2.jpg",
            "https://i.ibb.co/25FPND6/1.jpg",
            "https://i.ibb.co/C8T6ZkD/image.jpg"
        ],
        'videos': [
            "https://youtu.be/UD334WRM9uM?feature=shared",
            "https://youtu.be/HOeLIe8Wc1w?feature=shared",
            "https://youtu.be/SwUc5Kq6CzA?feature=shared"
        ],
        'texts': [
            "수질관리: 중성 또는 알칼리성 물을 선호하며, 적정수온은 24~27도 사이이다. 영역다툼이 심한 종으로 과밀사육을 하거나 단독으로 키우는 것이 좋다.",
            "먹이관리: 시클리드 전용 사료를 먹이거나 작은 물고기류와 갑각류, 수서곤충류를 특식으로 주어도 좋다.",
            "번식관리: 한번에 50~200여개의 알을 낳으며 암컷이 알을 입에 넣어두고 새끼가 부화할 때까지 먹이를 아무것도 먹지 않는다."
        ]
    },
      labels[4]: {
        'images': [
            "https://i.ibb.co/nLZrTzf/image.jpg",
            "https://i.ibb.co/pyC98jc/2.jpg",
            "https://i.ibb.co/Bj6gw6v/1.jpg"
        ],
        'videos': [
            "https://youtu.be/CwfZXmlhN6c?feature=shared",
            "https://youtu.be/xa7qWLkqbHY?feature=shared",
            "https://youtu.be/G6XpUFOPONc?feature=shared"
        ],
        'texts': [
            "수질관리: 정적 수온은 22~26도이거, 일주일에 한 번 정도 전체 수량의 20~30%를 교체해주는 것이 좋다.",
            "먹이관리: 주로 바닥에서 먹이를 찾기 때문에 가라앉는 형태의 사료를 주는 것이 좋다. 코리도라스 전용 사료 외에도 냉동 또는 생먹이를 제공해 영양 균형을 맞추는 것이 좋다.",
            "번식관리: 코리도라스는 수초나 수조의 유리면에 알을 낳는 습성이 있다. 산란기에는 수컷이 암컷을 쫓아다니기 때문에 산란이 쉽도록 수초는 유목이나 바위에 단단히 고정하는 것이 안전하다."
        ]
    },
      labels[5]: {
        'images': [
            "https://i.ibb.co/72v5pF1/2.jpg",
            "https://i.ibb.co/x255C5f/1.jpg",
            "https://i.ibb.co/6JGQRh5/image.jpg"
        ],
        'videos': [
            "https://youtu.be/rVtLXMy8OSo?feature=shared",
            "https://youtu.be/MCuh723Gmzg?feature=shared",
            "https://youtu.be/gJpKobEit2M?feature=shared"
        ],
        'texts': [
            "수질관리: 수온 22~28도, pH 7~8 사이의 수질에서 잘 자란다.",
            "먹이관리: 기본 먹이는 플레이크 사료나 알갱이 사료이고, 생먹이와 채소를 제공해 영양을 골고루 섭취하도록 하면 좋다.",
            "번식관리: 플래티는 난태생 물고기로, 암컷이 알을 몸속에서 부화시켜 살아있는 새끼를 낳는다. 새끼 플래티는 충분한 은신처를 제공해 주거나 치어통에 격리해서 키우는 것이 좋다."
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

