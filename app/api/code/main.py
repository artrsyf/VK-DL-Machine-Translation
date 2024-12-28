from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from translator import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./seq2seq_lstm_model.pth"

input_lang, output_lang, _ = prepareData('eng', 'rus', False)

input_lang_vocab_size = input_lang.n_words
output_lang_vocab_size = output_lang.n_words

try:
    seq2seq_lstm_model = Seq2SeqLstmModel(
        input_lang_vocab_size,
        output_lang_vocab_size,
        HIDDEN_SIZE
    ).to(device)

    load_model(seq2seq_lstm_model, 'seq2seq_lstm_model.pth')
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

@app.options("/translate")
async def handle_options():
    return JSONResponse(status_code=200, content={})

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        input_sentence = request.text.rstrip('\n')
        print(f"Input: {input_sentence}, Length: {len(input_sentence)}")
        
        translated_text = translateSentence(input_sentence, seq2seq_lstm_model, input_lang, output_lang)
        translated_text = translated_text.replace("<EOS>", "").strip()

        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке текста: {e}")

@app.get("/")
async def root():
    return {"message": "API для перевода текста"}
