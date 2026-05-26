# Marker che rende `app.services` un package importabile. I servizi sono
# moduli con logica di business "pura": ricevono input + dipendenze come
# argomenti, ritornano output. Niente accesso diretto a stato globale,
# niente conoscenza di FastAPI o HTTP. Questa separazione li rende
# facilmente testabili.
