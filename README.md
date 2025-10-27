# project

# Face-detection

# Aim:
To develop a Python program using OpenCV that detects human faces in images or video streams by employing the Haar Cascade Classifier technique.

# Algorithm:
~~~

Step-1:Import Libraries
Step-2:Load the Classifier and Image/Video
Step-3:Convert to Grayscale
Step-4:Detect Faces
Step-5:Display Results
~~~
# Program:
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline
```

```
model = cv2.imread('image_01.png',0)
withglass = cv2.imread('image_02.png',0)
group = cv2.imread('image_03.png',0)
```
```
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.resize(model, (1000, 1000)), cmap='gray')
plt.title('Model')
plt.subplot(1, 3, 2)
plt.imshow(cv2.resize(withglass, (1000, 1000)), cmap='gray')
plt.title('With Glass')
plt.subplot(1, 3, 3)
plt.imshow(cv2.resize(group, (1000, 1000)), cmap='gray')
plt.title('Group')
plt.show()
```
<img width="1362" height="453" alt="image" src="https://github.com/user-attachments/assets/635cd422-8b9d-4409-9544-fe816851168e" />

# Cascade Files
# Face Detecetion
```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
```
def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img
```

```
result = detect_face(model)
```
```
plt.imshow(result,cmap='gray')
plt.show()
```

<img width="588" height="502" alt="image" src="https://github.com/user-attachments/assets/62267115-01e8-40c9-b040-256b4f4224a1" />

```
result = detect_face(withglass)
```
```
plt.imshow(result,cmap='gray')
plt.show()
```
<img width="637" height="527" alt="image" src="https://github.com/user-attachments/assets/d71b5d7b-1ba8-4f84-946b-f3da8ee3b919" />

```
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```

<img width="758" height="437" alt="image" src="https://github.com/user-attachments/assets/394e8a19-e6a2-42e9-b1aa-b6c769102112" />

```
def adj_detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img
```
<img width="726" height="442" alt="image" src="https://github.com/user-attachments/assets/3aae4733-f171-49fb-b317-6a72634c92f9" />


# Eye Cascade file
```
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
```

```
def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img
```

```
result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()
```

<img width="497" height="512" alt="image" src="https://github.com/user-attachments/assets/bd468c35-7c18-4b9a-a44b-f4874d793e42" />

```
eyes = eye_cascade.detectMultiScale(withglass)
```
```
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```
<img width="657" height="537" alt="image" src="https://github.com/user-attachments/assets/08f28303-b0a1-4d1a-86ff-ba60b761e2e4" />

# Conjuction with video

```
cap = cv2.VideoCapture(0)

# Set up matplotlib
plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read(0)
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Video Face Detection')

while True:
    ret, frame = cap.read(0)

    frame = detect_face(frame)

    # Update matplotlib image
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)

   

cap.release()
plt.close()
```
<img width="687" height="547" alt="image" src="https://github.com/user-attachments/assets/60790bdb-9522-4fe2-9a27-0126809110e0" />



# Result:
The system successfully detects faces and highlights them by drawing square around each detected face.
