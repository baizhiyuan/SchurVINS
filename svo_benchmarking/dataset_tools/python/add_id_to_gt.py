#!/usr/bin/env python2

import os
import argparse
import numpy as np

def add_gt_id_and_save(input_file):
    # 获取输入文件所在目录并生成输出文件名
    outdir = os.path.dirname(input_file)
    outfn = os.path.join(outdir, 'groundtruth.txt')

    print("Going to add ID to {0} and write to {1}".format(input_file, outfn))

    # 加载输入的groundtruth数据
    data = np.loadtxt(input_file)
    print("Found {0} groundtruth data rows.".format(data.shape[0]))

    # 打开输出文件并写入带ID的数据
    with open(outfn, 'w') as fout:
        # 写入头部信息
        fout.write('# id timestamp(s) tx ty tz qx qy qz qw\n')

        # 为每一行数据添加ID并写入
        for idx, row in enumerate(data):
            # 写入ID和其他数据
            fout.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                (idx, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
            
            

    print("Written {0} rows with added IDs.".format(data.shape[0]))

if __name__ == '__main__':
    # 设置参数解析
    parser = argparse.ArgumentParser(
        description='Add incremental ID to the groundtruth data')
    parser.add_argument('groundtruth_file',
                        help="Groundtruth file that contains timestamp and pose data without id")
    args = parser.parse_args()

    # 确保输入文件存在
    if not os.path.exists(args.groundtruth_file):
        raise FileNotFoundError("Input file does not exist!")
    
    # 调用函数处理输入文件
    add_gt_id_and_save(args.groundtruth_file)
