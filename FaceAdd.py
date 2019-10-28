video = cv2.VideoCapture(0)
video.set(3,1920)
video.set(4,1080)


for i in range(0:7):

    res, frame = video.read()
    faceLocations = face_recognition.face_locations(cropImg,model="cnn", number_of_times_to_upsample=1)
    faceID = face_recognition.face_encodings(cropImg, known_face_locations=faceLocations)
    faceID = str(faceID).replace("[array([","").replace("])]","").replace('       ', '').replace('\n', '')

    insertPeopleTable(faceID, 0)
    sleep(0.5)


for i in range(0:7):

    res, frame = video.read()
    faceLocations = face_recognition.face_locations(cropImg,model="cnn", number_of_times_to_upsample=1)
    faceID = face_recognition.face_encodings(cropImg, known_face_locations=faceLocations)
    faceID = str(faceID).replace("[array([","").replace("])]","").replace('       ', '').replace('\n', '')

    insertPeopleTable(faceID, 0)
    sleep(0.5)


for i in range(0:7):

    res, frame = video.read()
    faceLocations = face_recognition.face_locations(cropImg,model="cnn", number_of_times_to_upsample=1)
    faceID = face_recognition.face_encodings(cropImg, known_face_locations=faceLocations)
    faceID = str(faceID).replace("[array([","").replace("])]","").replace('       ', '').replace('\n', '')

    insertPeopleTable(faceID, 0)
    sleep(0.5)


def insertPeopleTable(faceID, cat):
### Inserts Face Distence and the catagory in to the people table ###
        #cheeks that faceID is not 0
    if len(faceID) > 0:
            #Connects to the mysql server
        connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
        cursor = connection.cursor()
            #Query to send
        sql_insert_query = """ INSERT INTO `people` (`face`, `cat`) VALUES (%s,%s)"""
        insert = (faceID, cat)
        result  = cursor.execute(sql_insert_query, insert)
        connection.commit()
            #Closes the conection
        if(connection.is_connected()):
            cursor.close()
            connection.close()
    print("===========Added!===========")
