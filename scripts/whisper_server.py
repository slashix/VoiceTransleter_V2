import sys, json, os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from faster_whisper import WhisperModel

model: WhisperModel | None = None

def write_msg(obj: dict):
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

def load_model():
    global model
    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")
    try:
        write_msg({"type": "log", "message": f"Loading faster-whisper {model_size} ({device}, {compute_type})...", "id": 0})
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        write_msg({"type": "ready"})
    except Exception as e:
        if device == "cuda":
            # CUDA недоступна или упала инициализация — откатываемся на CPU, чтобы сервис не падал насмерть
            write_msg({"type": "log", "message": f"CUDA load failed ({e}), falling back to CPU int8...", "id": 0})
            try:
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                write_msg({"type": "ready"})
                return
            except Exception as e2:
                e = e2
        _fail_load(e)


def _fail_load(e: Exception):
    import traceback
    tb = traceback.format_exc()
    write_msg({"type": "error", "message": str(e) + "\n" + tb})
    sys.exit(1)

def transcribe(audio_path: str, language: str, req_id: int = 0):
    global model
    if model is None:
        write_msg({"type": "result", "status": "error", "message": "Model not loaded", "id": req_id})
        return
    try:
        lang = None if language == "auto" else language
        segments, info = model.transcribe(audio_path, language=lang, beam_size=5)
        detected = info.language if lang is None else lang
        result = []
        for seg in segments:
            result.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })
        write_msg({
            "type": "result",
            "status": "ok",
            "segments": result,
            "detected_language": detected,
            "id": req_id,
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        write_msg({"type": "result", "status": "error", "message": str(e) + "\n" + tb, "id": req_id})

if __name__ == "__main__":
    load_model()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            write_msg({"type": "result", "status": "error", "message": "invalid json", "id": 0})
            continue
        t = req.get("type")
        rid = req.get("id", 0)
        if t == "transcribe":
            transcribe(req.get("audio_path", ""), req.get("language", "auto"), rid)
        elif t == "shutdown":
            break
        else:
            write_msg({"type": "result", "status": "error", "message": f"unknown type: {t}", "id": rid})
