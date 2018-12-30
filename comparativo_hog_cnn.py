import cv2
import dlib

imagem = cv2.imread('fotos/teste.3.jpg')

detectorHog = dlib.get_frontal_face_detector()
facesDetectadasHog, pontuacao , idx = detectorHog.run(imagem, 2)

detectorCnn = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
facesDetectadasCnn = detectorCnn(imagem, 2)

for i, d in enumerate(facesDetectadasHog):
    print('HOG', pontuacao[i])

print("")

for face in facesDetectadasCnn:
    print('CNN',face.confidence)