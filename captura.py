import cv2

# Carregar o arquivo de reconhecimento
classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# especificando por onde ele vai capturar o video por número / 0 para webcam por estar em notebook
camera = cv2.VideoCapture(0)

while True:
    # variaveis que vão receber a leitura da webcam
    conectado, imagem = camera.read()

    # fazer detecção em uma escala de imagens cinzas/ Imagem que deseja escolher e tipo de converção
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # passar imagens que deseja detectar / todas que ele conseguir identificar
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     # scala da imagem
                                                     scaleFactor=1.5,
                                                     # tamanho da imagem
                                                     minSize=(100, 100))
    # fazer contorno / x, y largura altura
    # dentro de facesDetectadas existe uma matriz onde identifica onde começa e termina uma face
    for (x, y, l, a) in facesDetectadas:
        # fazer o retangulo / onde ele exibira(webcam),
        cv2.rectangle(imagem,
                      # parametros de posicionamento
                      (x, y),
                      # onde ele fechara o contorno
                      (x + l, y + a),
                      # cor
                      (0, 0, 255),
                      # tamanho da borda
                      2)
    # mostrar a imagem capturada na webcam
    cv2.imshow("FACE", imagem)
    cv2.waitKey(1)

# Salvar na memoria
camera.release()
cv2.destroyAllWindows()
