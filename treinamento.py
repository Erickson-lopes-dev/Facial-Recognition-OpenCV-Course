import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    # ler todas as imagens
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []

    # percorer cada imagem
    for caminhoImagem in caminhos:
        # pegar as imagens ja convertendo para escala cinza
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)

        # pegar os ids
        id = int(os.path.split(caminhoImagem)[-1].split('_')[1])

        print(id)

        # preenchenco lista com ids
        ids.append(id)

        # preenchendo lista de faces
        faces.append(imagemFace)

    return np.array(ids), faces


ids, faces = getImagemComId()

print('Treinando...')

# treina o algoritmo / aprendizagem supervisionada
eigenface.train(faces, ids)
# gera um arquivo com esta estenção
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento concluido.')