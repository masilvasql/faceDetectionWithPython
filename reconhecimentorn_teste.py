import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace =dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor('recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('recursos/dlib_face_recognition_resnet_model_v1.dat')
indices = np.load("recursos/indices_rn.pickle")
descritoresFaciais = "recursos/descritores_rn.npy"

for arquivo in glob.glob(os.path.join("fotos","*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace(imagem,2)
    for face in facesDetectadas:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [fd for fd in descritorFacial]
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        cv2.rectangle(imagem, (e,t), (d,b), (0,255,255),2)
    cv2.imshow("Detector hog ", imagem)
    cv2.waitKey(0)
cv2.destroyAllWindows()