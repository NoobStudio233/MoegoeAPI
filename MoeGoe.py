from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from torch import no_grad, LongTensor
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import uuid

logging.getLogger('numba').setLevel(logging.WARNING)
app = Flask(__name__)
CORS(app)

def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


def get_voice(text,out_path):
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

    model = "module/G_13000.pth"
    config = "module/config.json"

    hps_ms = utils.get_hparams_from_file(config)
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    while True:
        text = text#input('Text to read: ')
        if text == '[ADVANCED]':
            text = input('Raw text:')
            print('Cleaned text is:')
            ex_print(_clean_text(
                text, hps_ms.data.text_cleaners), escape)
            continue

        length_scale, text = get_label_value(
            text, 'LENGTH', 1, 'length scale')
        noise_scale, text = get_label_value(
            text, 'NOISE', 0.667, 'noise scale')
        noise_scale_w, text = get_label_value(
            text, 'NOISEW', 0.8, 'deviation of noise')
        cleaned, text = get_label(text, 'CLEANED')

        stn_tst = get_text(text, hps_ms, cleaned=cleaned)

        print_speakers(speakers, escape)
        speaker_id = 0

        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)])
            sid = LongTensor([speaker_id])
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                   noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                0, 0].data.cpu().float().numpy()

        write(out_path, hps_ms.data.sampling_rate, audio)
        return out_path

@app.route('/generate_audio', methods=['POST'])
def generate_audio_endpoint():
    try:
        data = request.get_json()
        text = data['text']

        unique_filename = str(uuid.uuid4()) + ".mp4"
        audio_file = get_voice(text,out_path="voice/"+unique_filename)

        return jsonify({"audio_url": audio_file}), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    file_path = "voice/" + filename
    response = send_file(file_path, as_attachment=True)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Expires'] = datetime(2000, 1, 1).strftime('%Y-%m-%d %H:%M:%S')
    return response

if __name__ == '__main__':
    app.run(debug=True)
