import cv2
import mediapipe as mp
import time

vid = cv2.VideoCapture(0) 
pTime = 0

#Créez un fichier de sortie à l'aide de la méthode cv2.VideoWriter_fourcc()
output = cv2.VideoWriter( 
        "output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 30, (1080, 1920)) 

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

frameWidth = 1920
frameHeight = 1080
vid.set(3, frameWidth)
vid.set(4, frameHeight)
vid.set(10, 150)

while(True): 
      
	ret, frame = vid.read() 
    
	imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
	results = faceMesh.process(imgRGB)
	if results.multi_face_landmarks:
		for faceLms in results.multi_face_landmarks:
			mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
			
	cTime = time.time()
	fps = 1 / (cTime - pTime)
	pTime = cTime
	
	cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
	
	#Display form on video
	# cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 3)

	#Write text on video
	# font = cv2.FONT_HERSHEY_SIMPLEX 
	# cv2.putText(frame,  
    #             'TEXT ON VIDEO',  
    #             (50, 50),  
    #             font, 1,  
    #             (0, 255, 255),  
    #             2,  
    #             cv2.LINE_4) 
	

	output.write(frame)
	cv2.imshow('frame', frame) 
      
	if (cv2.waitKey(1) & 0xFF == ord('q')) | (cv2.waitKey(1) == 27): 
		break
  
vid.release()
output.release() 
cv2.destroyAllWindows() 