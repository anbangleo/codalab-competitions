from send_to_bmt import SendtoBMT
import json
sendtobmt = SendtoBMT('anbang', '123456')
datasetbmt = [['123'],
              ['aaaaa'],['bbbbbb'], ['dddd']]
visual_value = ['spam','ham']
truth_value = ['0','1']

# data, rec_status = sendtobmt.decidetosend('test1234', 0, datasetbmt, 'hello world', 1, '20180808080809', 'question',
#                                     visual_value, truth_value)
token = sendtobmt.gettoken('anbang', '123456')
data, status = sendtobmt.getResult(token, 19)
try:
    j = json.loads(data)
    for key, value in j.items():
        # askidlist.append(key)
        print value
    # print j['data'][0]['$answer']
except:
    print '123'
# print type(j)
