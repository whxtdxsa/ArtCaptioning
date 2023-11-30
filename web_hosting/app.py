from flask import Flask, request, render_template, send_from_directory, url_for
import os
# from your_model_module import process_image  # 이 부분은 모델에 맞게 수정해야 합니다.
from gtts import gTTS
from torchvision import transforms
import final_model.model as model
import vocabulary 
import pickle

vocab_path = "./final_model/vocab.pkl"  # Path to the preprocessed vocabulary file
# Load vocabulary file
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# 파일 관리를 위한 함수
def manage_files(directory, max_files):
    files = os.listdir(directory)
    while len(files) > max_files:
        full_paths = [os.path.join(directory, file) for file in files]
        oldest_file = min(full_paths, key=os.path.getctime)
        os.remove(oldest_file)

# 현재 스크립트 파일의 디렉토리 경로를 얻기
current_directory = os.path.dirname(os.path.abspath(__file__))

# 현재 스크립트 파일의 디렉토리로 이동
os.chdir(current_directory)

app = Flask(__name__)

# 모델 파라미터 설정
embed_size = 256
hidden_size = 512
num_layers = 1

# 모델 초기화 및 가중치 로드
encoder = model.EncoderCNN(embed_size).eval()
decoder = model.DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
encoder.load_state_dict(model.torch.load('./final_model/encoder-10-40.ckpt'))
decoder.load_state_dict(model.torch.load('./final_model/decoder-10-40.ckpt'))

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device = model.torch.device('cuda' if model.torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)
decoder = decoder.to(device)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['AUDIO_FOLDER'] = 'audio/'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return '파일이 첨부되지 않았습니다.'
        file = request.files['file']
        if file.filename == '':
            return '파일이 선택되지 않았습니다.'
        
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            manage_files(app.config['UPLOAD_FOLDER'], 5)  # 이미지 폴더 관리
            # 이미지 처리 및 문장 도출
            # sentence = process_image(image_path)
            image_file = file

            image = model.load_image(image_file, transform)
            image_tensor = image.to(device)

            # 캡션 생성
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()

            # 단어 변환
            predicted_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                if word == '<start>':
                    continue
                if word == '<end>':
                    break
                predicted_caption.append(word)
            sentence = ' '.join(predicted_caption).capitalize()

            image_url = url_for('uploaded_file', filename=file.filename)

            # TTS로 음성 변환
            tts = gTTS(text=sentence, lang='ko')
            audio_filename = os.path.splitext(file.filename)[0] + '.mp3'
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts.save(audio_path)
            manage_files(app.config['AUDIO_FOLDER'], 5)  # 오디오 폴더 관리

            # 음성 파일 재생 및 다운로드 링크 제공
            return render_template('audio.html', image_url=image_url, sentence=sentence, audio_file=url_for('audio_file', filename=audio_filename))
    return render_template('upload.html')

@app.route('/audio/<filename>')
def audio_file(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    app.run(debug=True)
