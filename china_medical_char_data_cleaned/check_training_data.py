#encoding:utf-8
import sys
import codecs
import json

def formatCheck(filename):
    '''check the form of training data'''
    flag = 0
    entitydict = {}
    with codecs.open(filename, 'r', 'utf8') as f:
        for idx, data in enumerate(f, 1):
            data = data.strip()
            datalis = data.split(' ')

            if data:
                entitydict[datalis[0]] = entitydict[datalis[0]] if datalis[0] in entitydict else dict()
                id = entitydict[datalis[0]].get(datalis[-1], 0) + 1
                if id == 1:
                    count = entitydict[datalis[0]].get('count', 0) + 1
                    entitydict[datalis[0]]['count'] = count
                entitydict[datalis[0]][datalis[-1]] = id

                if len(datalis) != 2:
                    flag = 1
                    print('\n line number: % d  not equal 2, line value: %s' % (idx, data))
    if flag:
        raise ValueError('\n Please adjust your format errors!!!')
    
    entitydict = filter(lambda x:x[1]['count']>1,entitydict.items())
    entitydict = sorted(entitydict, key=lambda x:x[1]['count'], reverse=True)
    #print 'entitydict', entitydict
    with codecs.open('Check_Result.json','w','gbk') as json_file:
        json_file.write(json.dumps(entitydict, ensure_ascii=False))
        #json_file.write(json.dumps(entitydict, encoding="UTF-8", ensure_ascii=False))

    print('\nfinished checkingdata!')


if __name__=="__main__":
    if len(sys.argv) != 2:
        raise ValueError("""\n usage: python check_training_data.py ['./route/filename']""")
    formatCheck(sys.argv[1])
