import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
from datetime import date


# For Finding the Encodings
def findEncodings(images):
    encodeList = []
    for img in images:

        if len(encodeList) == int(len(images)/2):
            print("We are halfway there...")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        encode = face_recognition.face_encodings(imgS)[0]
        encodeList.append(encode)
        # print(f'{classNames[len(encodeList)-1]} Encoded...')
    return encodeList


# For Marking Attendance in csv file
def markStudentAttendance(name):
    with open(f'{subject_name}_{date.today()}.csv', 'r+') as f:
        all_attendance = f.readlines()
        present_students = []

        for single_record in all_attendance:
            values = single_record.split(',')
            present_students.append(values[0])

        # We should check that no student in marked present twice.
        if name not in present_students:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {time_string}')


path = "StudentsImages"
images = []
classNames = []

print("Please wait while we collect Student Details...")

myList = os.listdir(path)
# print(myList)


for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print("Student Details Fetched..!!!")
time.sleep(1)
# print(classNames)


print("\n\nPlease hold up while we Generate Encodings...\n"
      "This might take a while...\n")

encodeListKnown = findEncodings(images)

print("Encoding Completed..!!!")

subject_name = input("\nEnter Subject Name: ")
f = open(f'{subject_name}_{date.today()}.csv', 'w+')
f.write("Name, Time")
f.close()


# markStudentAttendance('Vivek')
# markStudentAttendance('Kunal')
# markStudentAttendance('Sahil')

cap = cv2.VideoCapture(0)

time_end = time.time() + 60

print("Attendance will be captured till 1 minute.")

success, img = cap.read()
while time.time() < time_end:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 233), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            markStudentAttendance(name)

    cv2.imshow("WebCam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"The Attendance for {subject_name} has been marked and saved..!!")
print("Thank You..!!!")
time.sleep(10)

cap.release()
