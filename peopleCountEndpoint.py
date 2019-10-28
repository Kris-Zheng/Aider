from flask import Flask,render_template
import mysql.connector
import numpy as np
from flask import request
from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.debug = True

@app.route('/')
def display_WebPage():
    return render_template("face.html")


@app.route('/pass')
def hello_world():
    startDate = request.args.get('StartDate')
    startTime = request.args.get('StartTime')
    EndDate = request.args.get('EndDate')
    EndTime = request.args.get('EndTime')


    st = "{0} {1}:00".format(startDate,startTime)
    et = "{0} {1}:00".format(EndDate,EndTime)

    cus = getPeople(st,et,0)
    staff = getPeople(st,et,1)

    cus = len(np.unique(cus))
    staff = len(np.unique(staff))
    return "Cust = {0}, Staff= {1}".format(cus,staff)


def getPeople(datestart, dateEnd, cat):
### Returns the contents of the ID category in the people Table on the SQL database. ###
    connection = mysql.connector.connect(host='localhost', database='testDB', user='main', password='Elephant1')
        #Connects to the mysql server
    cursor = connection.cursor()
        #Query to send
    Query = "select personID from clean where time > '{0}' and time < '{1}' and cat = {2}".format(datestart,dateEnd, cat)
    sql_select_query = Query

    cursor.execute(sql_select_query)
    record = cursor.fetchall()
    return(record)
        #Closes the conection
    if (connection.is_connected()):
        cursor.close()
        connection.close()
if __name__ == '__main__':
    app.run()
#ppl = getPeople('2019-05-09 04:40:50','2019-05-09 04:49:56',)
