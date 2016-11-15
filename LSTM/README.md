Los datasets (train, cv y test) se ponen en la carpeta "data" (usar ../data_init.py para crearlos).
El script main.py se encarga de "limpiarlos" y hacer las distintas conversiones usando data_init.py

Crear ademas una carpeta "models" (donde se guardan los modelos entrenados a medida que mejoran
el MSE en el set de validacion) y otra "logs" (para los logs de TensorFlow).

El modelo pre-entrenado de word2vec se guarda en la carpeta "pretrained" y se puede bajar del link
https://code.google.com/archive/p/word2vec/   (GoogleNews-vectors-negative300.bin.gz)
descomprimir

Recomendado usar ipython para ejecutar los scripts:

    from main import *

    # copiar el model a entrenar en la seccion "if __name__ == '__main__'"
    %paste


El modelo se crea, inicializa los datos, entrena y genera la salida en "results.csv"
