# encoding=utf8

import re
import codecs
import sys
import numpy as np

__doc__ = '''（1）将数据切分
             （2）将按列排列的训练语料转化成行排列的语料
             '''


def write_data(data, outfile):
    # print(data)
    with codecs.open(outfile, 'a+', 'utf8') as w_f:
        w_f.write(data)

        
def data_clean(strs):
    """
    :param strs:string documents 
    :return: string documents
    """
    pattern = [(u'，',','),(u'？','?'),(u'：',':'),(u'“','"'),(u'”','"'),(u"＞",">"),
               (u"‘","'"),(u"’","'"),(u"（","("),(u"）",")"),(u"《","<"),(u"＜","<"),
               (u"》",">"),(u"！","!"),(u"；",";"),(u'【',u'['),(u'】',u']'),
               (u'％',u'%'),(u'﹪',u'%')]
    for x,y in pattern:
        strs = re.sub(x,y,strs)
    return strs
        
def transfor_data(infile, word_file, tag_file):

    """:param infile:str, 输入文件路径 
        :param outfile:str, 输出文件路径
        :param counter: int,单次处理文件篇章数目
        :param sid: int,从sid篇开始处理长度为counter的篇章数据
        :return: null
    """

    sen, tag = list(), list()
    with codecs.open(infile, 'r', 'utf8') as in_f:
        indata = in_f.readline()
    #    len += len(indata)
        length = 0
        while indata: 
            length += 1
            data = data_clean(indata.strip())
			
            # print([indata])
            if not data:
                #print('***')
                datalist = data.split('\t')
#                sen.append(datalist[0] +str(length))
                sen.append(datalist[0])
                tag.append(datalist[-1])
                senline = ' '.join(sen).strip() + '\n'
                tagline = ' '.join(tag).strip() + '\n'

                write_data(senline, word_file)
                write_data(tagline, tag_file)
                sen, tag = list(), list()
                indata = in_f.readline()
                continue

            datalist = data.split('\t')
            sen.append(datalist[0])
            tag.append(datalist[-1])
            indata = in_f.readline()

        datalist = data.split('\t')
        sen.append(datalist[0])
        tag.append(datalist[-1])
        senline = ' '.join(sen).strip()
        tagline = ' '.join(tag).strip()

        write_data(senline, word_file)
        write_data(tagline, tag_file)
        print('finished transfor data {}'.format(infile))

        
def split_data(rawdatafile, test_ratio=0.25,
                savefile='train_data.txt testa_data.txt testb_data.txt'):
    """split data to train_data.txt, testa_data.txt, testb_data.txt
        rawdatafile: raw data file route
        testa_data = testb_data = test_ratio
        savefile: save data file route
    """
    with codecs.open(rawdatafile, 'r', 'utf8') as in_f:
        datalist = in_f.read().strip().split('\n\n')
        datalist = [data.strip() for data in datalist if data.strip()]
        
    #print(datalist)
    print('test_ratio', test_ratio)
    test_size = int(len(datalist)*test_ratio/2)
    train_size = len(datalist)-test_size*2
#    np.random.seed(10)
#    np.random.shuffle(datalist)
    train_data = datalist[:train_size]
    testa_data = datalist[train_size:train_size+test_size]
    testb_data = datalist[-test_size:]
    savefilelist = savefile.split()
    for idx, data in enumerate([train_data, testa_data, testb_data]):
        with codecs.open(savefilelist[idx], 'a+', 'utf8') as in_f:
            len_data = len(data)
            for id, row in enumerate(data,1):
                if id!=len_data:
                    in_f.write(row+'\n\n')
                else:
                    in_f.write(row)
                
        print('finish generate {}, total {} rows !'.format(savefilelist[idx], len(data)))

def main(infile, savefile='train_data.txt testa_data.txt testb_data.txt'):
    split_data(infile, savefile=savefile)
    for idx, data in enumerate(savefile.split()):
        dataname = re.sub('_data.txt', '', data)
        print('dataname',dataname)
        wordfile,tagfile = map(lambda x:"".join([dataname, x]), ['.words.txt','.tags.txt'])
        #wordfile,tagfile = map(lambda x:"".join(['./data/', dataname, x]),['.words.txt','.tags.txt'])
        transfor_data(data, wordfile, tagfile)

if __name__ == '__main__':
    data = sys.argv[1:]
    if len(data)<1:
        command = 'python data_transfor.py all_data.utf8'
        raise ValueError('please input the right format like: {}'.format(command))
    print(data)
    infile = data[0]
    main(infile)
