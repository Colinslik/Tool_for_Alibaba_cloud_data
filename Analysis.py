#!/usr/bin/env python
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (c) 2013-2018 ProphetStor Data Services, Inc.
# All Rights Reserved.
#
"""
ProphetStor Predictive Data Adapter
"""

import collections
import copy
import csv
import math
import multiprocessing
import os
import re

import numpy as np
import pandas as pd


PARALLEL_NUM = 5

CPU_CORE = 1
#MEMORY_SIZE = 256 * 1024 * 1024
MEMORY_SIZE = 1
#DISK_SIZE = 1024 * 1024 * 1024
DISK_SIZE = 1

INTERVAL = 3600

'''
BREAKPOINT_PATH = os.getcwd() + "/"
BREAKPOINT_FILE = 'breakpoint.txt'

WORKDIR_PATH = '/alibaba_cloud_data'
SRC_PATH = '/alibaba_clusterdata_v2018/'
EVENT_FILE = 'container_meta.csv'
USAGE_FILE = 'container_usage.csv'
OUTPUT_PATH = '/Analysis/'
OUTPUT_PREFIX = 'container_usage_of_hour_'

ANALYSIS_OUTPUT_PATH = WORKDIR_PATH + '/Merged_files'
ANALYSIS_OUTPUT_FILE = 'container_total_output.csv'

usage_column = {'container_id': 0, 'machine_id': 1,
                'time_stamp': 2, 'cpu_util_percent': 3,
                'mem_util_percent': 4, 'cpi': 5,
                'mem_gps': 6, 'mpki': 7,
                'net_in': 8, 'net_out': 9,
                'disk_io_percent': 10}
meta_column = {'container_id': 0, 'machine_id': 1,
               'time_stamp': 2, 'app_du': 3,
               'status': 4, 'cpu_request': 5,
               'cpu_limit': 6, 'mem_size': 7}

fill_usage_column = {'container_id': "c_0", 'machine_id': "m_0",
                'time_stamp': 0, 'cpu_util_percent': 0,
                'mem_util_percent': 0, 'cpi': 0.0,
                'mem_gps': 0.0, 'mpki': 0,
                'net_in': 0.0, 'net_out': 0.0,
                'disk_io_percent': 0}

fill_meta_column = {'container_id': "c_0", 'machine_id': "m_0",
               'time_stamp': 0, 'app_du': "app_0",
               'status': "started", 'cpu_request': 0,
               'cpu_limit': 0, 'mem_size': 0}

search_key = 'container_id'
merge_entry_pair = [('cpu_util_percent', 'cpu_limit'), ('mem_util_percent',
                                                          'mem_size')]
correction_term = {'cpu_util_percent': 1.0 / 100.0, 'cpu_limit': 1.0 / 100.0, 'mem_util_percent': 1.0 /
                   100.0, 'mem_size': MEMORY_SIZE, 'net_in': 1.0, 'net_out': 1.0, 'disk_io_percent': 1.0 / 100.0}
output_entry_list = ['cpu_util_percent', 'mem_util_percent',
                     'net_in', 'net_out', 'disk_io_percent']
conditons_list = [{'disk_io_percent': (0.0, 100.0)}]
filter_list = [{'container_id': '[^0-9]'}]
'''

BREAKPOINT_PATH = os.getcwd() + "\\"
BREAKPOINT_FILE = 'breakpoint.txt'

WORKDIR_PATH = 'D:\\eclipse-workspace\\CloudSimAnalysis\\'
SRC_PATH = 'trace_201708\\'
EVENT_FILE = 'container_event.csv'
USAGE_FILE = 'container_usage.csv'
OUTPUT_PATH = '\\Analysis\\'
OUTPUT_PREFIX = 'container_usage_of_hour_'

ANALYSIS_OUTPUT_PATH = WORKDIR_PATH + 'Merged_files'
ANALYSIS_OUTPUT_FILE = 'container_total_output.csv'

usage_column = {'time_stamp': 0, 'instance_id': 1,
                'cpu_util': 2, 'mem_util': 3,
                'disk_util': 4, 'load1': 5,
                'load5': 6, 'load15': 7,
                'avg_cpi': 8, 'avg_mpki': 9,
                'max_cpi': 10, 'max_mpki': 11}
meta_column = {'time_stamp': 0, 'event': 1,
               'instance_id': 2, 'machine_id': 3,
               'plan_cpu': 4, 'plan_mem': 5,
               'plan_disk': 6, 'cpuset': 7}

fill_usage_column = {'time_stamp': 0, 'instance_id': 0,
                     'cpu_util': 0.0, 'mem_util': 0.0,
                     'disk_util': 0.0, 'load1': 0.0,
                     'load5': 0.0, 'load15': 0.0,
                     'avg_cpi': 0.0, 'avg_mpki': 0.0,
                     'max_cpi': 0.0, 'max_mpki': 0.0}

fill_meta_column = {'time_stamp': 0, 'event': "CREATE",
                    'instance_id': 0, 'machine_id': 0,
                    'plan_cpu': 0, 'plan_mem': 0,
                    'plan_disk': 0, 'cpuset': "0|1"}

search_key = 'instance_id'
merge_entry_pair = [('cpu_util', 'plan_cpu'), ('mem_util',
                                               'plan_mem'), ('disk_util', 'plan_disk')]
correction_term = {'cpu_util': 1.0 / 100.0, 'plan_cpu': 1.0, 'mem_util': 1.0 /
                   100.0, 'plan_mem': MEMORY_SIZE, 'disk_util': 1.0 / 100.0, 'plan_disk': DISK_SIZE}
output_entry_list = ['cpu_util', 'mem_util', 'disk_util']
conditons_list = [{'max_mpki': (3.0, 12.0)}]
filter_list = [{'instance_id': '[^0-9]'}]


def pre_processing():
    previous_line = -1
    previous_file = None
    previous_row = None
    count = 0

    try:
        if os.path.exists(BREAKPOINT_PATH + BREAKPOINT_FILE) or not os.path.exists(WORKDIR_PATH + OUTPUT_PATH):

            if not os.path.exists(WORKDIR_PATH + OUTPUT_PATH):
                os.makedirs(WORKDIR_PATH + OUTPUT_PATH)

            if os.path.exists(BREAKPOINT_PATH + BREAKPOINT_FILE):
                with open(BREAKPOINT_PATH + BREAKPOINT_FILE, 'r') as bp:
                    lines = bp.readlines()
                previous_line = int(lines[0][:-1])
                previous_file = str(lines[1][:-1])
                previous_row = str(lines[2][:-1])

                last_line = None
                with open(previous_file, 'r') as pf:
                    for row in pf:
                        last_line = row[:-1]
                if last_line == previous_row:
                    previous_line += 1

            with open(WORKDIR_PATH + SRC_PATH + USAGE_FILE, 'r') as f:
                for raw in f:
                    count += 1
                    if count >= previous_line:
                        line = raw[:-1].split(',')
                        index = int(
                            line[usage_column['time_stamp']]) / INTERVAL
                        bpfile = open(BREAKPOINT_PATH + '/' +
                                      BREAKPOINT_FILE, 'w')
                        bpfile.writelines([str(count), '\n',
                                           "{0}".format(WORKDIR_PATH + OUTPUT_PATH +
                                                        OUTPUT_PREFIX + str(index) + ".csv"), '\n',
                                           raw])
                        bpfile.close()
                        out = open(WORKDIR_PATH + OUTPUT_PATH +
                                   OUTPUT_PREFIX + str(index) + ".csv", 'a')
                        out.write(raw)  # Give your csv text here.
                        # Python will convert \n to os.linesep
                        out.close()

            os.remove(BREAKPOINT_PATH + BREAKPOINT_FILE)
            print "Preprocessing complete."
        else:
            print "Data exist. Skip preprocessing."

    except Exception as e:
        print e


def memory_manager_processing(filename, container_event_dict):
    try:
        csv_file = WORKDIR_PATH + OUTPUT_PATH + filename
        usage_file = pd.read_csv(
            csv_file, index_col=False, names=[str(v[0]) for v in sorted(
                usage_column.items(), key=lambda d: d[1])])

        usage_file.fillna(value=fill_usage_column, inplace=True)

        for condition in conditons_list:
            expression = "(usage_file.{0} < {1}) | (usage_file.{0} > {2})".format(
                condition.keys()[0], condition.values()[0][0], condition.values()[0][1])

            usage_file = usage_file.drop(
                usage_file[pd.eval(expression)].index).reset_index(drop=True)

        for usage_key, event_key in merge_entry_pair:
            usage_file[usage_key] = [(x * correction_term[usage_key]) *
                                     (container_event_dict[
                                         y][event_key] * correction_term[event_key])
                                     if y in container_event_dict else 0.0
                                     for x, y in zip(usage_file[usage_key], usage_file[search_key])]

        for filter in filter_list:
            usage_file[filter.keys()[0]] = [re.sub(filter.values()[0], "", str(x))
                                            for x in usage_file[filter.keys()[0]]]

        # index will be replaced by search_key
        usage_file.set_index(keys=search_key, inplace=True)
        usage_file = usage_file.groupby(level=0).mean()

        usage_file.to_csv(ANALYSIS_OUTPUT_PATH + OUTPUT_PATH + filename,
                          sep=',', header=False, index=True)

        print "File output to " + ANALYSIS_OUTPUT_PATH + OUTPUT_PATH + filename

    except Exception as e:
        print e


def post_processing_parallel(container_file_list):
    container_event_file = WORKDIR_PATH + SRC_PATH + EVENT_FILE
    container_event_dict = {}
    output_row = collections.OrderedDict()
    output_list = []

    try:
        event_file = pd.read_csv(
            container_event_file, index_col=False,
            names=[str(v[0]) for v in sorted(meta_column.items(), key=lambda d: d[1])])
        event_file.fillna(value=fill_meta_column, inplace=True)
        event_file.set_index(
            keys=search_key, inplace=True)
        event_file = event_file[~event_file.index.duplicated(keep='first')]
        container_event_dict = event_file.to_dict('index')

        for filename in container_file_list:
            process = multiprocessing.Process(target=memory_manager_processing, args=(
                filename, container_event_dict), name="AnalysisSub")
            process.daemon = True
            process.start()
            process.join()

    except Exception as e:
        print e

    print "Sub-process has ended."


def post_processing():
    try:
        if not os.path.exists(ANALYSIS_OUTPUT_PATH + OUTPUT_PATH):
            os.makedirs(ANALYSIS_OUTPUT_PATH + OUTPUT_PATH)

        container_usage_file = [f for f in os.listdir(
            WORKDIR_PATH + OUTPUT_PATH) if f.startswith(OUTPUT_PREFIX)]

        if PARALLEL_NUM <= 1:
            post_processing_parallel(container_usage_file)
        else:
            process_list = []
            count = int(
                math.ceil(len(container_usage_file) / float(PARALLEL_NUM)))
            for num in range(PARALLEL_NUM):
                start = num * count

                if ((num + 1) * count) > len(container_usage_file):
                    end = len(container_usage_file) - 1
                else:
                    end = (num + 1) * count
                process = multiprocessing.Process(target=post_processing_parallel, args=(
                    [container_usage_file[start: end]]), name="AnalysisParallel")
                process.start()
                process_list.append(process)

            for proc in process_list:
                proc.join()

        print "POSTPROCESSING DONE!"

    except Exception as e:
        print e


def main():
    container_total_output = ANALYSIS_OUTPUT_PATH + ANALYSIS_OUTPUT_FILE
    output_row = collections.OrderedDict()
    output_list = []

    try:
        container_usage_file = [f for f in os.listdir(
            ANALYSIS_OUTPUT_PATH + OUTPUT_PATH) if f.startswith(OUTPUT_PREFIX)]

        for filename in container_usage_file:
            output_row['timestamp'] = int(filename[len(OUTPUT_PREFIX):-4])
            csv_file = WORKDIR_PATH + OUTPUT_PATH + filename
            usage_file = pd.read_csv(
                csv_file, index_col=False, names=[str(v[0]) for v in sorted(
                    usage_column.items(), key=lambda d: d[1])])

            for usage_key, event_key in merge_entry_pair:
                output_row[usage_key + '_sum'] = usage_file[usage_key].sum()
                output_row[usage_key + '_max'] = usage_file[usage_key].max()
                output_row[usage_key + '_min'] = usage_file[usage_key].min()
                output_row[usage_key +
                           '_median'] = usage_file[usage_key].median()

            output_list.append(copy.deepcopy(output_row))

        df = pd.DataFrame(output_list)
        df.sort_values("timestamp", inplace=True)

        # row header timestamp, output_entry_list[0]_sum,
        # output_entry_list[0]_max, output_entry_list[0]_min,
        # output_entry_list[0]_median, output_entry_list[1]_sum, ...
        df.to_csv(ANALYSIS_OUTPUT_PATH + ANALYSIS_OUTPUT_FILE,
                  sep=',', header=False, index=False)

        print "DONE!"

    except Exception as e:
        print e


if __name__ == '__main__':
    pre_processing()
    post_processing()
    main()
