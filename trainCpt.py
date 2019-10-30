import sys
import pyspark
import os
import numpy as np
import getopt

#os.getenv("")

default_data_path = "/Users/moonknight/Project/python/TrainCPT/data/test_data.txt"
sc = None
ss = None


def initsc(appName="trainCPT"):
    global sc
    global ss
    if (sc is None) and (ss is None):
        tempSparkBuild = pyspark.sql.SparkSession.builder \
            .master("local[3]") \
            .appName(appName)
        ss = tempSparkBuild.getOrCreate()
        sc = ss.sparkContext
        sc.setLogLevel("WARN")



def preHandleData(filePath):
    if os.path.exists(filePath):
        initsc()
        dataFile = sc.textFile(filePath)
        filter_data = dataFile.filter(lambda line: line.find("cpuMax") == -1)

        # delete the index column
        def mapStrToArray(line):
            str_array = line.split(" ")
            return str_array[1:]

        array_data = filter_data.map(mapStrToArray)

        print(array_data.count())
        return array_data

    else:
        raise RuntimeError("the file is not exists")


def calculateCPT(currentIndex: list, parentIndex: list, filePath):
    """
    the original array which contain the elements name is start from zero
    :param currentIndex:  the currentIndex point the current columns index
    :param parentIndex: the parentIndex point the columns index of parents of current node
    :param filePath: the data file path
    :return: a np.array or other
    """
    initsc()
    array_data = preHandleData(filePath)
    # we need the count of all match and parent match
    currentIndex_bro = sc.broadcast(currentIndex)
    parentIndex_bro = sc.broadcast(parentIndex)

    def map_parent_match_topair(line_list):
        key = ""
        for index in parentIndex_bro.value:
            key += line_list[index] + ","
        return (key[:-1], float(line_list[-1]))

    parent_data = array_data.map(map_parent_match_topair).reduceByKey(lambda a, b: a + b)

    def map_current_match_to_pair(line_list):
        key = ""
        for index in currentIndex_bro.value:
            key += line_list[index] + ","
        return (key[:-1], 1)

    current_data = array_data.map(map_current_match_to_pair).reduceByKey(lambda a, b: a + b)

    # use map to construct a index
    # the columns represent the parent combination,the row represent the status
    output_arr = np.zeros(shape=[current_data.count(), parent_data.count()])
    current_data_index = {}
    array_current_index = 0
    row_list = []
    columns_list = []
    for value, _ in current_data.collect():
        row_list.append(value)
        current_data_index[value] = array_current_index
        array_current_index += 1

    parent_data_list = parent_data.collect()
    parent_data_index = 0
    print("the total count is: " + str(len(parent_data_list)))
    for parent_value, count in parent_data_list:
        columns_list.append(parent_value)
        print(parent_data_index)
        parent_value_bro = sc.broadcast(parent_value)

        def filter_no_match_parent(line):
            parent_value_list = parent_value_bro.value.split(",")
            match = True
            for index in parentIndex_bro.value:
                if line[index] != parent_value_list[index]:
                    match = False
                    break
                else:
                    continue
            return match

        match_parent_data = array_data.filter(filter_no_match_parent)

        def map_all_match_topair(line):
            key = ""
            for index in parentIndex_bro.value:
                key += line[index] + ","
            key = key[:-1] + "&"
            for index in currentIndex_bro.value:
                key += line[index] + ","
            return (key[:-1], float(line[-1]))

        match_all_data = match_parent_data.map(map_all_match_topair).reduceByKey(lambda a, b: a + b)
        for match_all_value, all_match_count in match_all_data.collect():
            current_value = match_all_value.split("&")[-1]
            current_array_index = current_data_index[current_value]
            output_arr[current_array_index][parent_data_index] = float(all_match_count) / float(count)
        parent_data_index += 1

    # write the result to file

    return (row_list, columns_list, output_arr)


if __name__ == "__main__":
    # (row_list, columns_list, output_arr) = calculateCPT([4], [0, 1, 2, 3], default_data_path)
    """
    this script could accept four arguments
    1.the data file path,if file path does not exists,the script will exit
    2.the path of output file
    3.the index of current node which you want to calculate CPT,the index should correspond to first line of data file
    this arguments could be a list
    4.a array which split by ",",it point the parents of current node,like 1,2,4,6
    """
    if os.path.exists(sys.argv[1]):
        if True:
            # handle current index
            current_index_list = sys.argv[3].split(",")
            current_index = []
            for ele in current_index_list:
                current_index.append(int(ele))
            parent_index_list = sys.argv[4].split(",")
            parent_index = []
            for ele in parent_index_list:
                parent_index.append(int(ele))
            (row_list, columns_list, output_arr) = calculateCPT(current_index, parent_index, sys.argv[1])
            np.savetxt(sys.argv[2], output_arr)

            with open(os.path.split(sys.argv[2])[0]+"/info.txt", "w") as file:
                row_in_file=""
                for ele in row_list:
                    row_in_file+=ele+" "
                columns_in_file=""
                for ele in columns_list:
                    columns_in_file+=ele+" "
                file.write(row_in_file+"\n")
                file.write(columns_in_file+'\n')
                file.write("\n")

        else:
            print("the output path is error")
    else:
        print("the data file path is error")
