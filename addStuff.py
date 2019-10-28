import datetime
from datetime import datetime
from datetime import timedelta
import numpy as np
import cv2
import face_recognition
import mysql.connector
from mysql.connector import Error

def insertPeopleTable(faceID, cat):
### Inserts Face Distence and the catagory in to the people table ###
        #cheeks that faceID is not 0
    if len(faceID) > 0:
            #Connects to the mysql server
        connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
        cursor = connection.cursor()
            #Query to send
        sql_insert_query = """ INSERT INTO `faces` (`face`, `cat`) VALUES (%s,%s)"""
        insert = (faceID, cat)
        result  = cursor.execute(sql_insert_query, insert)
        connection.commit()
            #Closes the conection
        if(connection.is_connected()):
            cursor.close()
            connection.close()

def addToMasterFirst(faces):
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()

    sql_insert_query = """INSERT INTO master (faceid,cat) VALUES (%s,%s)"""
    insert = (str(faces),1)

    cursor.execute(sql_insert_query, insert)
    connection.commit()
    if(connection.is_connected()):
        cursor.close()
        connection.close()

def addToMasterNfirst(faces):
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
        #Query to send
    sql_insert_query = """UPDATE master set faceid = %s where personid = %s"""
    holdSting=""
    for j in range(len(faces2Add)):
        holdSting += (str(faces2Add[j])+",")
    insert =(holdSting,str(name[0][0]).replace(',',''))

    result  = cursor.execute(sql_insert_query, insert)

    connection.commit()
        #Closes the connection
    if(connection.is_connected()):
        cursor.close()
        connection.close()


def getFaceID(face):

    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
    sql_select_query = "select faceID from faces where face = '{0}'".format(face)

    cursor.execute(sql_select_query)
    record = cursor.fetchall()
    listOutFinal =[]
        #Loop splits the output of the SQL query and saves it to an array.
    for records in record:
        holdRecords = str(records).replace("(u'","").replace('\n', '').replace("',)","").replace("(","").replace("([","").replace("])","").replace("array","").replace(")","")
        listHolding = holdRecords.split(',')
        listHolding = listHolding[:-1]
        for i in range(len(listHolding)):
            if int(listHolding[i]) != 0:
                listOutFinal.append(int(listHolding[i]))
    return(listOutFinal)
        #Closes the connection
    if (connection.is_connected()):
            cursor.close()
            connection.close()

def checkFaces(faceHold,name):
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
    faces = []
    print(name)
    if len(name) > 0:
        sql_select_query = "select faceid from master where personid = '{}'".format(name[0][0])
        cursor.execute(sql_select_query)
        record = cursor.fetchall()

        listOutFinal =[]
            #Loop splits the output of the SQL query and saves it to an array.
        for records in record:
                #Cleans the string of returned data
            holdRecords = str(records).replace("(u'","").replace('\n', '').replace("',)","").replace("',)","").replace("([","").replace("])","").replace("array","").replace("[","").replace("]","")
                #Creates a list from the cleaned string
            listHolding = holdRecords.split(',')
            listHolding = listHolding[:-1]

            for i in range(len(listHolding)):
                listOutFinal.append(int(listHolding[i]))
            faces = listOutFinal
            #Closes the connection
        if (connection.is_connected()):
            cursor.close()
            connection.close()

    faces2Add = faces
    print(faces2Add)

    for i in faceHold:
        if i not in faces:
            faces2Add.append(i)

    if len(name) == 0 and len(faces)>0:
        connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
        cursor = connection.cursor()

        sql_insert_query = """INSERT INTO master (faceid,cat) VALUES (%s,%s)"""
        insert = (str(faces),0)

        cursor.execute(sql_insert_query, insert)
        connection.commit()
        if(connection.is_connected()):
            cursor.close()
            connection.close()

    if len(name) > 0 and len(faces)>0:

            #Connects to the mysql server
        connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
        cursor = connection.cursor()
            #Query to send
        sql_insert_query = """UPDATE master set faceid = %s where personid = %s"""
        holdSting=""
        for j in range(len(faces2Add)):
            holdSting += (str(faces2Add[j])+",")
        insert =(holdSting,str(name[0][0]).replace(',',''))

        result  = cursor.execute(sql_insert_query, insert)

        connection.commit()
            #Closes the connection
        if(connection.is_connected()):
            cursor.close()
            connection.close()

def getName(ID):
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()

    names = []
    for s in range(len(ID)):
        sql_select_query = "select personid, cat from master where faceid like '%{0}%'".format(str(ID[s]))
        print(sql_select_query)
        cursor.execute(sql_select_query)
        names = cursor.fetchall()
        if len(names) > 0:
            break
    return(names)


video = cv2.VideoCapture(0)
masterList=""
for i in range(0,100):
    print(str(i) + " of 250")
    res, frame = video.read()
    faceLocations = face_recognition.face_locations(frame,model="hog", number_of_times_to_upsample=1)
    faceID = face_recognition.face_encodings(frame, known_face_locations=faceLocations)
    if len(faceID) > 0:
        print("added")
        faceID = str(faceID).replace("[array([","").replace("])]","").replace('       ', '').replace('\n', '')
        insertPeopleTable(faceID, 1)
        faceid = getFaceID(faceID)
        masterList += str(faceid[0]) + ","
    else:
        print "fail"
addToMasterFirst(masterList[0:-1])
