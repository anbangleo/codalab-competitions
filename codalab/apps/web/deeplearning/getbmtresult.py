from send_to_bmt import SendtoBMT
import csv
import urllib
import urllib2
import requests


sendtobmt = SendtoBMT()
token = sendtobmt.gettoken('anbang', '123456')
data, status = sendtobmt.getResult(token)

csv_reponse = '/app/codalab/static/img/partpicture/' + 'anbang' + '/bmtresponse.csv'

with open(csv_reponse, 'wb', encoding = 'utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'entity'])

urllib.urlretrieve(csv_reponse, 'bmtresult.csv')



