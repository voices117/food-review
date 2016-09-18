Bibliografia
============

Las citas a bibliografia se agregan en informe.bib (pueden tomar como ejemplo las que ya estan hechas)

para referenciar la cita en el texto, ponen

    \cite{nombre_de_la_referencia}


Capitulos
=========

Para agregar un nuevo capitulo, hay que meter en la carpeta Chapters un nuevo archivo con el nombre
    <nombre_capitulo.tex>.

Despues, en main.tex agregan

    \include{Chapters/nombre_capitulo} en donde estan las demas lineas similares

Hay un archivo en la carpeta que se llama "ChapterTemplate.tex". Pueden copiar el contenido en el nuevo
archivo del capitulo para usarlo como base.


Imagenes
========

Las imagenes las van en la carpeta Figures, y despues se referencian desde Latex de la siguiente forma

    \begin{figure}[h]
    \centering
    \includegraphics{Figures/<NombreImagen>}
    \decoRule
    \caption[Una imagen]{Descripcion de la imagen.}
    \label{fig:Nombre de la imagen}
    \end{figure}
