import cv2
from simple_facerec import SimpleFacerec

# Encode faces from the folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")



# Load CAmera
cap =cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Frame", frame)
    
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


