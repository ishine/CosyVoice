import torchaudio
import os, sys, re
import torch
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
from funasr import AutoModel as ASRModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
Matcha_path = 'third_party/Matcha-TTS'
sys.path.append(Matcha_path)

ref_audios_zh = [
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_1_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_2_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_3_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_4_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_5_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_6_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_7_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_8_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_9_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_10_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_11_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_12_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_13_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_14_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_15_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_16_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_17_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_18_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_19_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_20_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_21_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/CN_22_30s.wav",
]

ref_audios_en = [
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_1_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_2_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_3_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_4_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_5_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_6_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_7_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_8_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_9_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_10_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_11_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_12_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_13_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_14_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_15_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_16_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_17_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_18_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_19_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_20_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_21_30s.wav",
    "/data/megastore/SHARE/TTS/ref_audios/test/EN_22_30s.wav",
]


texts_zh = open('cntts_vc_text.txt', 'r', encoding='utf-8').readlines()
texts_en = open('entts_vc_text.txt', 'r', encoding='utf-8').readlines()

model_dir = "iic/SenseVoiceSmall"
sensevoice = ASRModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="/data/megastore/Projects/DuJing/code/SenseVoice/model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)


def process(ref_audio, texts, start_idx=1):
    print(f'Cloning the reference audio: {ref_audio}')
    id = os.path.basename(ref_audio)
    id = os.path.splitext(id)[0]
    prompt_speech_16k = load_wav(ref_audio, 16000)
    print(prompt_speech_16k.size())
    if prompt_speech_16k.size(1) > 160000:
        prompt_speech_16k = prompt_speech_16k[:, :160000]
    
    
    prompt_speech_24k = load_wav(ref_audio, 24000)
    if prompt_speech_24k.size(1) > 240000:
        prompt_speech_24k = prompt_speech_24k[:, :240000]
    torchaudio.save(f'tmp.wav', prompt_speech_24k, 24000)

    # 
    res = sensevoice.generate(
        input=prompt_speech_16k[0],
        cache={},
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    # prompt_text = rich_transcription_postprocess(res[0]["text"])
    prompt_text = re.sub(r'<[^>]*>', '', res[0]["text"])
    print("prompt_text:", prompt_text)
    
    for idx, text in enumerate(texts):
        text = text.strip()
        cur_wave = []
        for i,j in enumerate(cosyvoice.inference_zero_shot(
                text,
                "You are a helpful assistant.<|endofprompt|>"+prompt_text,
                'tmp.wav',
                stream=True)):

            cur_wave.append(j['tts_speech'])

        wave = torch.cat(cur_wave, dim=-1)
        true_idx = start_idx + idx
        torchaudio.save(f'{true_idx}_{id}_{text[0:10]}.wav', wave, 24000)

def run_vc(ref_audios, total_texts, txt_per_spk=15):
    for idx, ref_audio in enumerate(ref_audios):
        texts = total_texts[idx*txt_per_spk: (idx+1)*txt_per_spk]
        process(ref_audio, texts, start_idx = idx*txt_per_spk+1)


if __name__ == "__main__":
    run_vc(ref_audios_zh, texts_zh)
    run_vc(ref_audios_en, texts_en)
    