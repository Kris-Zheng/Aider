from ctypes import *
import math
from sort import *
import random
import cv2
import numpy as np
from random import randint
import face_recognition
import mysql.connector
from mysql.connector import Error


def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

def getFaces():
### Returns the contents of the face category in the people Table on the SQL database. ###
        #Connects to the MySQL server
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
        #Query to send
    sql_select_query = """select face from faces"""
    cursor.execute(sql_select_query)
    record = cursor.fetchall()
    listOutFinal =[]
        #Loop splits the output of the SQL query and saves it to an array.
    for records in record:
            #Cleans the string of returned data
        holdRecords = str(records).replace("(u'","").replace('\n', '').replace("',)","").replace("',)","").replace("([","").replace("])","").replace("array","")
            #Creates a list from the cleaned string
        listHolding = holdRecords.split(',')
            #Creates an numpy array of the list
        outList = np.array([float(i) for i in listHolding])
            #Creates a 2d array of the array.
        listOutFinal.append(outList)
    return(listOutFinal)
        #Closes the connection
    if (connection.is_connected()):
            cursor.close()
            connection.close()


def getIds():
### Returns the contents of the ID category in the people Table on the SQL database. ###
    connection = mysql.connector.connect(host='localhost', database='testDB', user='main', password='Elephant1')
        #Connects to the mysql server
    cursor = connection.cursor()
        #Query to send
    sql_select_query = """select faceID from faces"""
    cursor.execute(sql_select_query)
    record = cursor.fetchall()
    return(record)
        #Closes the conection
    if (connection.is_connected()):
        cursor.close()
        connection.close()

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


def insertLogTable(ids,idnum):
        #Connects to the mysql server
    connection = mysql.connector.connect(host='localhost',database='testDB',user='main',password='Elephant1')
    cursor = connection.cursor()
        #Query to send
    sql_insert_query = """INSERT INTO log (trackid, faceid ) VALUES (%s,%s)"""
    insert = (int(ids),int(idnum))
    result  = cursor.execute(sql_insert_query, insert)
    connection.commit()
        #Closes the connection
    if(connection.is_connected()):
        cursor.close()
        connection.close()


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]




class IplROI(Structure):
    pass

class IplTileInfo(Structure):
    pass

class IplImage(Structure):
    pass

IplImage._fields_ = [
    ('nSize', c_int),
    ('ID', c_int),
    ('nChannels', c_int),
    ('alphaChannel', c_int),
    ('depth', c_int),
    ('colorModel', c_char * 4),
    ('channelSeq', c_char * 4),
    ('dataOrder', c_int),
    ('origin', c_int),
    ('align', c_int),
    ('width', c_int),
    ('height', c_int),
    ('roi', POINTER(IplROI)),
    ('maskROI', POINTER(IplImage)),
    ('imageId', c_void_p),
    ('tileInfo', POINTER(IplTileInfo)),
    ('imageSize', c_int),
    ('imageData', c_char_p),
    ('widthStep', c_int),
    ('BorderMode', c_int * 4),
    ('BorderConst', c_int * 4),
    ('imageDataOrigin', c_char_p)]


class iplimage_t(Structure):
    _fields_ = [('ob_refcnt', c_ssize_t),
                ('ob_type',  py_object),
                ('a', POINTER(IplImage)),
                ('data', py_object),
                ('offset', c_size_t)]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    """if isinstance(image, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, image = array_to_image(image)
        rgbgr_image(im)
    """
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    track_bbs_ids = mot_tracker.update(dets)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res



def runOnVideo(net, meta, vid_source, thresh=.8, hier_thresh=.5, nms=.45):
### Main program ###
    video = cv2.VideoCapture(vid_source)
        #sets video size

    video.set(3,1920)
    video.set(4,1080)


        #Creates an intance of sort
    motTracker = Sort()
        #Save all the faces from the Data base to memory.
    knowenPeople = getFaces()
        #Tags holds each trackers ID number.
    Tags = []
        #Tag tracks the detected people and increases each run.
    Tag = 0


    while video.isOpened():
            #Reads frame from webcam.
        res, frame = video.read()

        HDframe = frame
        frame = cv2.resize(frame, (960, 540))

        if not res:
            break
            #Converts the colors from fame to RGB - Posibly not needed.
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Create and array from rgbFrame.
        im, arr = array_to_image(rgbFrame)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
            #Creates a holder list for the bounding boxes to be passed to tracker.
        locations = []
        if (nms): do_nms_obj(dets, num, meta.classes, nms);

            #Set people count tracker to zero.
        peopleCount = 0
        IDlist = getIds()
        trackIds = []
        holdingFinal = []
        for j in range(num):
            for i in range(meta.classes):
                    #checks the probability of the detected box containing a person.
                if dets[j].prob[i] > 0:
                    Tag+=1
                        #TagHold saves the current count of tag.
                    TagHold = Tag
                        #Sets b to the boundry boxes of the detector.
                    b = dets[j].bbox

                        ##Sets x and y to the corners of the boundary boxes also checks that they are not negative as this corses CNN face detector to crash.
                    x1 = (int(b.x - b.w / 2.)+1)
                    if x1 < 0:
                        x1 = 0
                    y1 = (int(b.y - b.h / 2.)+1)
                    if y1 < 0:
                        y1 = 0
                    x2 = (int(b.x + b.w / 2.)+1)
                    if x2 < 0:
                        x2 = 0
                    y2 = (int(b.y + b.h / 2.)+1)
                    if y2 < 0:
                        y2 = 0

                        #Sets idNumber to a place holder.
                    idNumber = ''
                        #Sets probs to the probability of the detected box containing a person.
                    probs = dets[j].prob[i]
                        #Crops the image to the detected person, this should allow for targeted recognition

                    boxLocals = [x1,y1,x2,y2,probs]
                    locations.append(boxLocals)

        trackIds = motTracker.update(np.asarray(locations))


        for personDe in trackIds:

                #print(HDframe[yy1:yy2, xx1:xx2])
            yy1 = int(personDe[1])
            yy2 = int(personDe[3])
            xx1 = int(personDe[0])
            xx2 = int(personDe[2])


            yy1 = yy1*2
            xx1= xx1*2
            yy2 =yy2*2
            yy2 = int((yy1+(yy2/6))) #-----------8
            xx2 = (xx2*2)


            if xx1 <= 0: xx1 = 1
            if xx2 <= 0: xx2 = 2
            if yy1 <= 0: yy1 = 3
            if yy2 <= 0: yy2 = 4

            if xx1 >= 1920: xx1 = 1919
            if xx2 >=1920: xx2 = 1918
            if yy1 >= 1080: yy1 = 1079
            if yy2 >= 1080: yy2 = 1078



            cropImg = HDframe[yy1:yy2, xx1:xx2]

            #cropImg = frame[y1:y2, x1:x2]
                #Detects the face in a frame and aligns it, cnn is the detection module, number of times to upsample is the amount to exspand the image by (Slows processing).
            faceLocations = face_recognition.face_locations(cropImg,model="cnn", number_of_times_to_upsample=1)
                #Returns the distence between the points on the face.
            faceID = face_recognition.face_encodings(cropImg, known_face_locations=faceLocations)
                #Place holder for idNum
            idNum = ''

                ##If a face is found...
            if len(faceID) > 0:

                        #Cleaning Detected face
                    faceID = str(faceID).replace("[array([","").replace("])]","").replace('       ', '').replace('\n', '')
                        #Hold for face to be added if needed.
                    faceHoldForAdd = faceID
                        #More cleaning.
                    faceID = str(faceID).replace("(u'","").replace('\n', '').replace("',)","").replace("])","").replace("([","").replace("array","")
                        #Creates an array from the faceID.
                    listHold = (faceID.split(','))
                        #Cycals through listHold and adds the values to array.
                    outList = np.array([float(i) for i in listHold])
                        #Resit faceID to a list of numpy arrays with face distences.
                    faceID = [outList]
                        #Get Ids frome people table.

                        # checks to see if 128 face distances are recorded.
                    if str(outList.shape) == "(128,)":
                            #Compare the faces in the data base against the detected face, tolerance is the amount of error to return true.

                        compare = face_recognition.compare_faces(knowenPeople,outList,tolerance=0.3)
                        if sum(compare) == 0:
                                #Adds new face to database.
                            insertPeopleTable(faceHoldForAdd, 0)
                                #resets knowenPeople so that new face is included.
                            knowenPeople = getFaces()

                            #Gets the indexs of the posive faces
                        RowsMatches = [i for i, x in enumerate(compare) if x]
                            ##cheaks if more then one face in the data base.
                        if len(RowsMatches) > 1:
                                #finds the best matching face.
                            distances = face_recognition.face_distance(knowenPeople,outList)
                            for i in range(0,len(distances)):
                                RowsMatches = min(xrange(len(distances)), key=distances.__getitem__)

                                if IDlist[RowsMatches] == 0:
                                    distances[RowsMatches] = 0
                                else:
                                    #gets ID number of the matching face.
                                    idNum = str(IDlist[RowsMatches])
                                    IDlist[RowsMatches] = 0
                                    break
                        else:
                            for matchrun in RowsMatches:
                                    #gets ID number of the matching face.
                                idNum = str(IDlist[matchrun])
                                if IDlist[matchrun] == 0:
                                    pass
                                else:
                                    #gets ID number of the matching face.
                                    idNum = str(IDlist[matchrun])
                                    IDlist[matchrun] = 0


                    else:
                        #set tag to zero as a fail safe.
                        Tag = 0

                ##If there is no matching ID.
            if idNum == '':
                idNum = '0'
                    #set tag to zero as a fail safe.
                Tag = 0
                    #Draws a green rectangle around the detected object.

            peopleCount+=1
                    #resets Tag
            Tag = TagHold

            holdingFinal.append(np.append(personDe,str(idNum)))



        for number in holdingFinal:
                #Cleans Tags
            trackID =int(number[4].split(".")[0])
            IDNUM = number[5].split(".")[0]
                #update the log table

            x = int(number[0].split(".")[0])
            n1 = int(number[1].split(".")[0])
            n2 = int(number[2].split(".")[0])
            y = int((n1 + n2)/2)

                #update the log table

            IDNUM = only_numerics(IDNUM)
            insertLogTable(int(trackID),int(IDNUM))
            #Draws red box around Id people
            cv2.rectangle(frame, (int(number[0].split(".")[0]), int(number[1].split(".")[0])), (int(number[2].split(".")[0]), int(number[3].split(".")[0])), (0,0,255), 2)
            #Writes Id to the image
            cv2.putText(frame, str((number[4])), (int(number[0].split(".")[0]), int(number[1].split(".")[0])), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2, cv2.LINE_AA)

            if int(IDNUM) != 0:
                cv2.rectangle(frame, (int(number[0].split(".")[0]), int(number[1].split(".")[0])), (int(number[2].split(".")[0]), int(number[3].split(".")[0])), (255,0,0), 2)
                #Writes Id to the image
                cv2.putText(frame, str(IDNUM), (int(number[0].split(".")[0])+100, int(number[1].split(".")[0])), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0), 2, cv2.LINE_AA)


        #Reset people count
        peopleCount= 0
        Tags = []

            #show image
       # cv2.imshow('output', frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    net = load_net("cfg/yolov3.cfg".encode('utf-8'), "yolov3.weights".encode('utf-8'), 0)
    meta = load_meta("cfg/voc.data".encode('utf-8'))

    #vid_source = 0
       #Video source - Webcam
    #vid_source = 0
    vid_source = "b.mp4"
    #vid_source = 0
runOnVideo(net, meta, vid_source)
