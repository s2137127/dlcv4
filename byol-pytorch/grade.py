import csv

import numpy as np

if __name__ == '__main__':
    ans_path = '/home/alex/Desktop/hw4-s2137127/hw4_data/office/val.csv'
    pred_path = './output.csv'
    with open(ans_path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        ans = [row[2] for row in rows]
    with open(pred_path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        pred = [row[2] for row in rows]
    cnt = 0
    for i in range(len(ans)):
        if ans[i] == pred[i]:
            cnt += 1
    print('grade: ',cnt/len(ans))