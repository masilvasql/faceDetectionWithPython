import cv2
import dlib

def imprimePontos(imagem, pontosFaciais):
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 1, (0,255,0), 2)
        #imagem, posição dos pontos, raio do círculo, cor e tamanho da borda

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
imagem = cv2.imread("fotos/treinamento/ronald.0.0.jpg")
detectorFace = dlib.get_frontal_face_detector()
detectorDePontosFaciais = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
facesDetectadas = detectorFace(imagem, 2)
for face in facesDetectadas:
    pontos = detectorDePontosFaciais(imagem, face)
    print(pontos.parts())
    print(len(pontos.parts()))
    imprimePontos(imagem, pontos)

cv2.imshow('Pontos Faciais',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
