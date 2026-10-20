import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
	device="cpu"
)
texto = "Hola, Bienvenidos a inteligencia artificial generativa, vamos a crear audios, imagenes, textos y videos"

print("Generando audio...")

audio = modelo.generate(
	texto,
	language_id="es",
	exaggeration=0.7, # Expresión o emocion
	cfg_weight=0.4, # nos ajusta el ritmo, si la voz va rapido
	temperature=0.8, # Control la variabilidad, mas alta = mas variabilidad
	repeticion_penalty=1.2, # penaliza repeticiones extrañas
	main_p=0.05, #Filtra opciones improbables durante las operaciones
	top_p=1.0 #COntrola el muestreo probabilistico, 1.0 lo deja abierto
)

ta.save(
	"audio.wav",
	audio,
	modelo.sr
)

print("Audio guardado")