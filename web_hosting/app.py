from flask import Flask, request, render_template, send_from_directory, url_for
import os
# from your_model_module import process_image  # 이 부분은 모델에 맞게 수정해야 합니다.
from gtts import gTTS
import final_model.model as model

app = Flask(__name__)

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
            model.manage_files(app.config['UPLOAD_FOLDER'], 5)  # 이미지 폴더 관리
            # 이미지 처리 및 문장 도출
            encoder, decoder = model.load_model('./final_model/art-encoder.ckpt', './final_model/art-decoder.ckpt')
            sentence = model.generate_caption(image_path, encoder, decoder)

            print("Caption:", sentence)

            image_url = url_for('uploaded_file', filename=file.filename)

            # TTS로 음성 변환
            tts = gTTS(text=sentence, lang='ko')
            audio_filename = os.path.splitext(file.filename)[0] + '.mp3'
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts.save(audio_path)
            model.manage_files(app.config['AUDIO_FOLDER'], 5)  # 오디오 폴더 관리

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
