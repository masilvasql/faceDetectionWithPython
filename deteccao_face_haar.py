import cv2 #importação do open CV

imagem = cv2.imread("fotos/teste.jpg")
classificador = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # função para converter imagem para cinza
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.2, minSize=(80,80))
#scaleFactor = aumentar a escala da imagem
# minSize = tamanho mínimo da face que se deseja detectar

print(facesDetectadas)
print("Faces detectadas: ", len(facesDetectadas))

for(x, y, l,  a) in facesDetectadas:
    cv2.rectangle(imagem, (x,y),(x + l, y + a), (0,255,0),2)
    # os parametros passados são:
    # imagem original
    # x = primeiro valor da matriz
    # y = segundo valor da matriz
    # l = terceiro valor da matriz (largura)
    # a = quarto valor da matriz (altura)
    # 0,255,0 = valor RGB e o ultimo 2 é a largura da borda

cv2.imshow("detector har cascade",imagem) #1º parâmetro é o título da janela
cv2.waitKey(0)
cv2.destroyAllWindows()