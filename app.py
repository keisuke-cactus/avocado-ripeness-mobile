from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet18


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'best_model.pth'


@st.cache_resource
def load_model():
    # ① ResNet18の構造を用意する
    model = resnet18(weights=None)
    # ② 5クラス分類の構造にする
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    # ③ 保存した重みを読み込み、モデルに設定する
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    )
    # ④ 予測に使うモードにする
    model.eval()
    return model


# 学習・検証時と同じ前処理
transform = transforms.Compose([
    transforms.Resize(256), # 画像のサイズを整える
    transforms.CenterCrop(224), # 中央を切り取り、224x224にそろえる
    transforms.ToTensor(), # 画像をPyTorch用の数値データに変換する
    transforms.Normalize( # RGBそれぞれの数値の尺度を整える
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

st.set_page_config(page_title='アボカド熟成度アプリ', page_icon='🥑')
st.title('🥑 アボカド熟成度アプリ')
st.write('アボカドの画像から、公開データの熟成段階1〜5を推定します。')
st.caption('段階の数字が大きいほど熟成が進んだ分類です。')

if not MODEL_PATH.exists():
    st.error('app.pyと同じフォルダにbest_model.pthを置いてください。')
    st.stop()

st.caption('アボカドを1個ずつ、画面の中央に大きく写してください。')
mode = st.radio('画像の入力方法', ['写真を選ぶ', 'カメラで撮影'], horizontal=True)
if mode == 'カメラで撮影':
    uploaded = st.camera_input('アボカドを撮影してください')
else:
    uploaded = st.file_uploader('画像を選択してください', type=['jpg', 'jpeg', 'png'])
st.caption('画像は推論のためにサーバーへ送信されます。このアプリのコードでは画像をファイル保存しません。')

if uploaded is not None:
    try:
        with Image.open(uploaded) as original:
            picture = ImageOps.exif_transpose(original).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error('画像を読み込めませんでした。別のJPEGまたはPNG画像を選んでください。')
        st.stop()

    st.image(picture, caption='入力画像', width=360)
    if st.button('熟成度を判定する', type='primary'):
        with st.spinner('判定しています…'):
            model = load_model()
            inputs = transform(picture).unsqueeze(0)
            with torch.inference_mode():
                probabilities = torch.softmax(model(inputs), dim=1)[0]
            stage = int(probabilities.argmax().item()) + 1
        stage_names = {
            1: "未熟",
            2: "熟し始め",
            3: "食べ頃の初期",
            4: "十分に熟した状態",
            5: "熟しすぎ（過熟）",
        }

        st.subheader(
            f"推定結果：{stage_names[stage]}（段階{stage}）"
        )
    
        st.bar_chart({
            '段階': [f'段階{i}' for i in range(1, 6)],
            'モデルの出力（%）': (probabilities * 100).tolist(),
        }, x='段階', y='モデルの出力（%）')
        st.caption('グラフはモデルの出力分布です。予測が正しい確率を保証するものではありません。')

st.info('学習用の試作アプリです。食用の安全性や内部の状態は判断できません。撮影条件の異なる画像での性能は未検証です。')

with st.expander('モデルと評価について'):
    st.write('ResNet18の学習済み部分を固定し、最後の分類層を5クラス用に学習しました。')
    st.write('478個体・14,710枚を使用。同一個体が混ざらないよう、学習286個体、検証96個体、テスト96個体に分割しました。')
    st.write('テスト3,026枚の正解率：65.7%。段階別の判別性能には差があります。')
    st.markdown('データ出典：[Hass Avocado Ripening Photographic Dataset](https://data.mendeley.com/datasets/3xd9n945v8/1)')
    st.caption('データ作成者：Pedro Xavier、Pedro Rodrigues、Cristina L. M. Silva。ライセンス：CC BY 4.0。画像をサイズ変更・正規化して学習に使用。')
