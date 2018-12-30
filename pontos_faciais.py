import cv2
import dlib
import numpy as np

def imprimePontos(imagem, pontosFaciais):
    for p in pontosFaciais:
        cv2.circle(imagem, (p.x, p.y), 1, (0,255,0), 2)
        #imagem, posição dos pontos, raio do círculo, cor e tamanho da borda

def imprimeNumeros(imagem, pontosFaciais):
    for i, p in enumerate(pontosFaciais):
        cv2.putText(imagem, str(i), (p.x, p.y), fonte, .55, (0,0,255),1)
        #imagem, número, posição dos números, fonte, tamanho da fonte, cor e expessura

def imprimeLinhas(imagem, pontosFaciais):
    p68 =[[0,16,  False], # pontos linhas do queixo, false = ele não faz ligação das linhas
          [17,21, False], #sobrancelha direita
          [22,26, False],#sobrancelha esquerda
          [27,30, False],# ponte nasal
          [30,35, True], #nariz inferior
          [36,41, True], #olho esquerdo
          [42,47, True], #olho direito
          [48,59, True], #labio externo
          [60,67, True]] #labio interno
    for k in range(0, len(p68)):
        pontos = []
        for i in range(p68[k][0], p68[k][1]+1):
            ponto = [pontosFaciais.part(i).x, pontosFaciais.part(i).y]
            pontos.append(ponto)
        pontos = np.array(pontos, dtype=np.int32)
        cv2.polylines(imagem, [pontos], p68[k][2], (255,0,0),2)


fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
imagem = cv2.imread("fotos/treinamento/ronald.0.1.jpg")
detectorFace = dlib.get_frontal_face_detector()
detectorDePontosFaciais = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
facesDetectadas = detectorFace(imagem, 2)

for face in facesDetectadas:
    pontos = detectorDePontosFaciais(imagem, face)
    #imprimePontos(imagem, pontos.parts())
    #imprimeNumeros(imagem,pontos.parts())
    imprimeLinhas(imagem, pontos)
cv2.imshow('Pontos Faciais',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
