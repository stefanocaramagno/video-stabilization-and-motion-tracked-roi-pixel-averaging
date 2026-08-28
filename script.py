import numpy as np # Importa il modulo numpy per la gestione degli array
import cv2 # Importa il modulo OpenCV per la gestione delle immagini e dei video

# Definisce il raggio di smoothing
RAGGIO_LISCIA = 50  

# Funzione per calcolare la media mobile su una curva
def mediaMobile(curva, raggio): 
    # Calcola la dimensione della finestra di smoothing
    dimensione_finestra = 2 * raggio + 1
    # Crea un filtro uniforme per la media mobile
    f = np.ones(dimensione_finestra) / dimensione_finestra 
    # Aggiunge il padding ai bordi della curva per evitare artefatti
    curva_padded = np.lib.pad(curva, (raggio, raggio), 'edge') 
    # Applica la convoluzione tra la curva pad e il filtro
    curva_lisciata = np.convolve(curva_padded, f, mode='same') 
    # Rimuove il padding aggiunto precedentemente
    curva_lisciata = curva_lisciata[raggio:-raggio]
    return curva_lisciata 

# Funzione per lisciare una traiettoria
def smooth(traiettoria):
    # Crea una copia della traiettoria originale 
    traiettoria_lisciata = np.copy(traiettoria) 
    # Applica la media mobile su ciascuna delle tre colonne (dx, dy, da)
    for i in range(3):
        traiettoria_lisciata[:, i] = mediaMobile(traiettoria[:, i], raggio=RAGGIO_LISCIA)
    return traiettoria_lisciata

# Funzione per correggere i bordi di un frame
def correggiBordo(frame):
    # Ottiene la forma del frame
    s = frame.shape
    # Calcola la matrice di rotazione
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    # Applica la trasformazione affine al frame
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# Funzione per stabilizzare il video
def stabilizzaVideo(percorso_input, percorso_output):
    # Apre il file video
    cap = cv2.VideoCapture(percorso_input)
    # Verifica se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'aprire il file video.")
        return
    
    # Ottiene il numero totale di frame
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # Ottiene la larghezza del frame
    larghezza = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    # Ottiene l'altezza del frame
    altezza = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Ottiene il frame rate del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Definisce il codec e crea il VideoWriter per salvare il video stabilizzato
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Creazione del video output per la stabilizzazione
    out = cv2.VideoWriter(percorso_output, fourcc, fps, (2 * larghezza, altezza))

    # Lettura del primo frame
    _, precedente = cap.read() 
    # Converte il frame in scala di grigi
    precedente_gray = cv2.cvtColor(precedente, cv2.COLOR_BGR2GRAY) 
    # Inizializza l'array per le trasformazioni
    trasformazioni = np.zeros((n_frame - 1, 3), np.float32) 

    # Iterazione sui frame per calcolare e applicare le trasformazioni
    for i in range(n_frame - 1):
        # Trova i punti di interesse nel frame precedente
        punti_precedenti = cv2.goodFeaturesToTrack(precedente_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        # Legge il frame corrente
        success, corrente = cap.read() 
        if not success: 
            break 
        # Converte il frame corrente in scala di grigi
        corrente_gray = cv2.cvtColor(corrente, cv2.COLOR_BGR2GRAY) 
        # Calcola il flusso ottico per trovare i punti corrispondenti
        punti_correnti, status, err = cv2.calcOpticalFlowPyrLK(precedente_gray, corrente_gray, punti_precedenti, None) 
        # Filtra i punti validi
        idx = np.where(status == 1)[0]
        punti_precedenti = punti_precedenti[idx]
        punti_correnti = punti_correnti[idx]
        # Stima la trasformazione affine tra i due set di punti
        m, _ = cv2.estimateAffinePartial2D(punti_precedenti, punti_correnti)
        if m is None:
            continue
        # Estrae le componenti di traslazione e rotazione
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        trasformazioni[i] = [dx, dy, da]
        # Aggiorna il frame precedente
        precedente_gray = corrente_gray

    # Calcola la traiettoria cumulativa
    traiettoria = np.cumsum(trasformazioni, axis=0) 
    # Liscia la traiettoria
    traiettoria_lisciata = smooth(traiettoria) 
    # Calcola la differenza tra traiettoria lisciata e originale
    differenza = traiettoria_lisciata - traiettoria
    # Applica la differenza alle trasformazioni originali
    trasformazioni_lisciate = trasformazioni + differenza
    # Riporta il video al primo frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

    # Ciclo per applicare le trasformazioni lisciate ai frame
    for i in range(n_frame - 1):
        # Legge il frame corrente
        success, frame = cap.read() 
        if not success:
            break
        # Estrae le trasformazioni lisciate
        dx = trasformazioni_lisciate[i, 0]
        dy = trasformazioni_lisciate[i, 1]
        da = trasformazioni_lisciate[i, 2]
        # Crea la matrice di trasformazione
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        # Applica la trasformazione al frame
        frame_stabilizzato = cv2.warpAffine(frame, m, (larghezza, altezza))
        # Corregge i bordi del frame stabilizzato
        frame_stabilizzato = correggiBordo(frame_stabilizzato) 
        # Unisce il frame originale e stabilizzato per confronto
        frame_output = cv2.hconcat([frame, frame_stabilizzato])
        # Ridimensiona il frame se troppo grande
        if frame_output.shape[1] > 1920: 
            frame_output = cv2.resize(frame_output, (frame_output.shape[1] // 2, frame_output.shape[0] // 2))

        # Mostra il frame prima e dopo la stabilizzazione
        cv2.imshow("Prima e Dopo", frame_output)
        cv2.waitKey(10)
        # Scrive il frame nel video di output
        out.write(frame_output)

    # Rilascia il video e chiude tutte le finestre
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Funzione per selezionare la ROI (Region Of Interest)
def selezionaRoi(frame):
    # Consente all'utente di selezionare una ROI sul frame
    roi = cv2.selectROI("Seleziona ROI", frame, fromCenter=False, showCrosshair=True)
    # Chiude la finestra di selezione ROI
    cv2.destroyWindow("Seleziona ROI")
    return roi

# Funzione per calcolare la media dei pixel in una ROI del video
def calcolaMediaVideo(percorso_video, roi, frame_min, frame_max):
    # Apre il file video
    cap = cv2.VideoCapture(percorso_video)
    
    # Verifica se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'aprire il file video.")
        return None
    
    # Ottiene il numero totale di frame
    numero_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Limita il frame_min al valore minimo consentito
    frame_min = max(frame_min, 0)
    # Limita il frame_max al valore massimo consentito
    frame_max = min(frame_max, numero_frame - 1)
    
    # Estrae le coordinate e dimensioni della ROI
    x, y, larghezza, altezza = roi
    # Inizializza un array per la somma dei pixel
    somma_pixel = np.zeros((altezza, larghezza, 3), np.float64)
    
    # Posiziona il video al frame iniziale
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_min)
    _, primo_frame = cap.read()
    frame_precedente_gray = cv2.cvtColor(primo_frame, cv2.COLOR_BGR2GRAY)
    # Posizione iniziale del punto chiave nella ROI
    punti_precedenti = np.array([[x + larghezza // 2, y + altezza // 2]], dtype=np.float32)
    
    # Iterazione sui frame selezionati per calcolare la media dei pixel nella ROI
    for numero_frame in range(frame_min, frame_max + 1):
        # Leggi il frame successivo
        ret, frame = cap.read()
        if not ret:
            break

        # Converti il frame in scala di grigi
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        punti_correnti, status, err = cv2.calcOpticalFlowPyrLK(frame_precedente_gray, frame_gray, punti_precedenti, None)

        # Verifica lo stato del punto chiave
        if status[0] == 1:

            # Calcola lo spostamento del punto chiave nella ROI
            dx, dy = punti_correnti[0] - punti_precedenti[0]
            x += int(dx)
            y += int(dy)
            punti_precedenti = punti_correnti

        # Estrai la ROI corrente dal frame
        frame_roi = frame[y:y+altezza, x:x+larghezza]

        # Aggiungi i pixel della ROI alla somma accumulata
        somma_pixel += frame_roi

        # Aggiorna il frame precedente in scala di grigi
        frame_precedente_gray = frame_gray
    
    # Calcola il numero di frame selezionati
    numero_frame_selezionati = frame_max - frame_min + 1
    # Calcola la media dei pixel
    media_pixel = somma_pixel / numero_frame_selezionati
    # Arrotonda i valori dei pixel e converte a uint8
    media_pixel = np.array(np.round(media_pixel), dtype=np.uint8)
    
    # Rilascia il video
    cap.release()
    
    return media_pixel

# Funzione per salvare la media dei pixel come immagine
def salvaMediaComeImmagine(media_pixel, percorso_output):
    if media_pixel is None:
        return
    
    # Salva l'immagine della media dei pixel
    cv2.imwrite(percorso_output, media_pixel)
    print(f"Immagine media salvata con successo: {percorso_output}")

# Funzione per creare un report HTML con le immagini originali e medie
def creaReportHtml(percorso_img_originale, percorso_img_media, percorso_output_html):
    # Crea il contenuto HTML del report
    contenuto_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Report</title>
    </head>
    <body>
        <h1>Report di Stabilizzazione Video</h1>
        <h2>Immagine Originale</h2>
        <img src="{percorso_img_originale}" alt="Immagine Originale" width="50%">
        <h2>Immagine Media</h2>
        <img src="{percorso_img_media}" alt="Immagine Media" width="50%">
    </body>
    </html>
    """
    # Scrive il contenuto HTML nel file di output
    with open(percorso_output_html, 'w') as file:
        file.write(contenuto_html)
    print(f"Report HTML salvato con successo: {percorso_output_html}")

# Funzione principale
def main():
    # Definisce i percorsi dei file
    percorso_video = 'video_originale.mp4'
    percorso_output_stabilizzato = 'video_stabilizzato.mp4'
    percorso_output_media = 'foto_media.jpg'
    percorso_output_html = 'report.html'
    
    # Stabilizza il video
    stabilizzaVideo(percorso_video, percorso_output_stabilizzato)
    
    # Apre il video stabilizzato e legge il frame_di_apertura
    cap = cv2.VideoCapture(percorso_output_stabilizzato)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
    _, frame_di_apertura = cap.read()
    roi = selezionaRoi(frame_di_apertura)
    cap.release()

    # Definisce i frame di inizio e fine per il calcolo della media del video
    frame_min = 2
    frame_max = 8
    
    # Calcola la media dei pixel della ROI nei frame selezionati
    media_pixel = calcolaMediaVideo(percorso_output_stabilizzato, roi, frame_min, frame_max)
    
    # Salva l'immagine della media dei pixel
    salvaMediaComeImmagine(media_pixel, percorso_output_media)
    
    # Salva il frame_min originale con la ROI
    percorso_img_originale = percorso_output_media.replace(".jpg", "_originale.jpg")
    cv2.imwrite(percorso_img_originale, frame_di_apertura[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]) # 'roi' è una tupla (x, y, larghezza, altezza)
    
    # Crea il report HTML con le immagini
    creaReportHtml(percorso_img_originale, percorso_output_media, percorso_output_html)

# Avvia la funzione principale se lo script è eseguito come programma principale
if __name__ == "__main__":
    main()
