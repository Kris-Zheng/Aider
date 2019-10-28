import datetime
from datetime import datetime
from datetime import timedelta
import numpy as np
import mysql.connector
from mysql.connector import Error
#from ctypes import *


def getTrck(timeS,now):
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
    sql_select_query = """select trackid from log where time > %s and time < %s"""
    ver = (timeS,now)

    cursor.execute(sql_select_query,ver)
    record = cursor.fetchall()
    listOutFinal =[]
        #Loop splits the output of the SQL query and saves it to an array.
    for records in record:
        holdRecords = str(records).replace("(u'","").replace('\n', '').replace("',)","").replace("(","").replace("([","").replace("])","").replace("array","").replace(")","")
        listHolding = holdRecords.split(',')
        listHolding = listHolding[:-1]

        for i in range(len(listHolding)):
            if int(listHolding[i]) not in listOutFinal:
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


def getFaceID(trackid,timeS,now):

    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
    sql_select_query = """select faceid from log where trackid = %s and time > %s and time < %s"""
    ver = (trackid,timeS,now)
    cursor.execute(sql_select_query,ver)
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

def getName(ID):
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()

    names = []
    for s in range(len(ID)):
        sql_select_query = "select personid, cat from master where faceid like '%{0}%'".format(str(ID[s]))
        cursor.execute(sql_select_query)
        names = cursor.fetchall()
        if len(names) > 0:
            break
    return(names)


def addToCleanLog(time,name,cat):
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
        #Query to send

    sql_insert_query = """INSERT INTO clean (time, personID, cat) VALUES (%s,%s,%s)"""
    insert = (time,name,cat)
    result  = cursor.execute(sql_insert_query, insert)
    connection.commit()
        #Closes the connection
    if(connection.is_connected()):
        cursor.close()
        connection.close()

timeS =  datetime.now()


def clearLog(time):
     connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
     cursor = connection.cursor()
         #Query to send
     sql_insert_query = "DELETE FROM log WHERE time < '{0}'".format(time)
     result  = cursor.execute(sql_insert_query)
     connection.commit()
         #Closes the connection
     if(connection.is_connected()):
         cursor.close()
         connection.close()

def common_data(list1, list2):
    result = False

    # traverse in the 1st list
    for x in list1:

        # traverse in the 2nd list
        for y in list2:

            # if one common
            if x == y:
                result = True
                return result

    return result

trackingHold = []
while True:
    name= []
    now = datetime.now()
    nowStr = now.strftime("%Y/%m/%d, %H:%M:%S")
    tempholdTime = timeS
    tempholdTime= (tempholdTime + timedelta(seconds=10))
    tempholdTime = tempholdTime.strftime("%Y/%m/%d, %H:%M:%S")
    trackingHold =[]
    if (tempholdTime == nowStr):
        trackIDs = getTrck(timeS,now)
        trackIDsComp = trackIDs
        toSendNames = []
        faces=[]
        sentNames=[]
        for i in range(len(trackIDs)):
            faces = getFaceID(trackIDs[i],timeS,now)
            for k in range(len(trackIDs)):
                facehold = getFaceID(trackIDsComp[k],timeS,now)
                if facehold in faces:
                    trackIDsComp[k] = ''
                    faces.append(facehold)
                    facehold = []

            name = getName(faces)

            if len(name) ==0 :
                print("fire")
                templist = []
                for x in trackingHold:
                    print(x)
                    templist.append(x[0])
                for p in range(len(templist)):
                    if trackIDs[k] == templist[p]:
                        name = ([(trackingHold[p][1],trackingHold[p][2])])
                        break


            checkFaces(faces,name)
            name = getName(faces)


            if len(name) >0 :
                toSendNames.append(name)
                print(name)
                trackingHold.append([trackIDs[i],name[0][0],name[0][1]])

        for f in range(len(toSendNames)):
                addToCleanLog(timeS,toSendNames[f][0][0],toSendNames[f][0][1])

        print("Ran")
        clearLog(now)
        timeS = now
