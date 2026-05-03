
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os, uuid, shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "MP3 para Partitura API"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".mp3") and "audio" not in file.content_type:
        raise HTTPException(400, "Envie um arquivo MP3.")

    tmp_dir = tempfile.mkdtemp()
    mp3_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.mp3")
    midi_path = mp3_path.replace(".mp3", ".mid")

    try:
        with open(mp3_path, "wb") as f:
            f.write(await file.read())

        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        model_output, midi_data, note_events = predict(
            mp3_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=50,
        )

        midi_data.write(midi_path)

        import music21
        score = music21.converter.parse(midi_path)

        notes = []
        for el in score.flat.notesAndRests:
            if hasattr(el, 'pitch'):
                notes.append({
                    "note": el.pitch.name.replace("-", "b"),
                    "octave": el.pitch.octave,
                    "duration": str(el.duration.type),
                    "offset": float(el.offset),
                })

        bpm = 120
        for el in score.flat:
            import music21.tempo as mt
            if isinstance(el, mt.MetronomeMark):
                bpm = int(el.number)
                break

        key = score.analyze('key')
        key_str = f"{key.tonic.name} {key.mode}"

        return JSONResponse({
            "status": "ok",
            "bpm": bpm,
            "key": key_str,
            "notes": notes[:64],
            "total_notes": len(notes),
        })

    except Exception as e:
        raise HTTPException(500, f"Erro na transcrição: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
