from bert_serving.client import BertClient
import datetime
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

bc = BertClient()
m = 2

#def trigger_func(m):
file_dir = './gcn/why_merged_2_set-Copy1.tsv'
trigger = []
with open(file_dir, 'r') as f:
    line = f.readline()
    while line:
        trigger.append(line[:-1])
        line = f.readline()


print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  

node_feat_vec_H0 = bc.encode(trigger)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

node_feat_vec_H0.tofile('./node_feat_vec_H0_cutoff_' + str(m) + '.txt')

print(node_feat_vec_H0.shape)