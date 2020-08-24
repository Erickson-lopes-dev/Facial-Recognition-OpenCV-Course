import cv2

# carregar o arquivo de detecção de faces
classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# indicar captura das imagens / 0 para webcam
camera = cv2.VideoCapture(0)

# controla todos tiradas
amostra = 1

# total de fotas
numeroAmostra = 25

id = input('Digite seu identificador: ')

# tamanho da foto tirada
altura, largura = 220, 220

print('Capturando as faces...')

while True:
    # transferindo para as variaveis a leitura da camera/imagens
    conectado, imagem = camera.read()

    # com imagens cinza o desenpenho do algoritmo tende a sewr melhor / converte a imagem recebida para Gray
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2BGRA)

    # detecta as faces
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     # escala da imagem
                                                     scaleFactor=1.5,
                                                     # tamanho da imagem
                                                     minSize=(150, 150)
                                                     )

    # retem o posicionamento da face encontrada repassando suas coordenadas
    for x, y, l, a in facesDetectadas:
        # desenhar um retangulo onde a face foi detectada
        cv2.rectangle(imagem,
                      # onde começa
                      (x, y),
                      # onde termina
                      (x + l, y + a),
                      # valor que deseja pintar e tamanho da borda
                      (0, 0, 255), 2
                      )

        # Toda vez que apertar a tecla 'q' ele executara
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Redimencionar a imagem
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (altura, largura))
            # Gravas as imagens (nome / imagem redimencionada)
            cv2.imwrite(f"fotos/pessoa_{str(id)}_{str(amostra)}.jpg", imagemFace)

            print(f'foto {str(amostra)} capturada com sucesso')

            # incremento
            amostra += 1

    # mostrar a imagem capturada
    cv2.imshow("Face", imagem)
    cv2.waitKey(1)

    # se o número de fotos capturada for maior que o numero de amostra
    if amostra >= numeroAmostra + 1:
        # para o loop
        break

camera.release()
cv2.destroyAllWindows()
