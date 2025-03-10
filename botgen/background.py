import modin.pandas as pd
import swifter # Do not remove - this modified bindings for modin
import sys, os
import datetime
import csv
import random
import h5py
import numpy as np
import ipaddress
import datetime as datetime


def write_single_graph(f, graph_id, x, edge_index, y, attrs=None, **kwargs):
    '''
    store into hdf5 file
    '''
    f.create_dataset(f'{graph_id}/x', data=x, dtype = 'float32')
    f.create_dataset(f'{graph_id}/edge_index', data=edge_index, dtype = 'int64')
    f.create_dataset(f'{graph_id}/y', data=y, dtype = 'uint8')
    for key in kwargs:
        f.create_dataset(f'{graph_id}/{key}', data=kwargs[key])
    if attrs is not None:
        for key in attrs:
            f[f'{graph_id}'].attrs[key] = attrs[key]
    return None


def ip2int(ip):
    '''
    convert x.x.x.x into a number
    '''
    try:
        ip = ip.split(',')[0]
        ip = ipaddress.ip_address(ip)
        ip = int(ip)
        return ip
    except:
        return random.randint(0, 1<<32)

def search_dict(IP, IP_dict):
    '''
    use a dictionary to renumber the IPs into 0,1,2,...
    '''
    if IP not in IP_dict:
        IP_dict[IP] = len(IP_dict)
    return IP_dict[IP]


def prepare_background_(f, start_time, stop_time, NPARTS=30):
    #read data
    df = pd.read_csv(f, sep = '@')#, nrows = 10000)#
    df.columns = ["time", "srcIP", "dstIP"]

    # contains per-minute logs
    #filter time
    df['time'] = df['time'].swifter.set_npartitions(NPARTS).apply(lambda x: datetime.datetime.strptime(x[:21], "%b %d, %Y %H:%M:%S"))

    if start_time is not None:
        start_time_formated = datetime.datetime.strptime(start_time, "%Y%m%d%H%M%S")
        df = df[ df.time >= start_time_formated]
    
    if stop_time is not None:
        stop_time_formated = datetime.datetime.strptime(stop_time, "%Y%m%d%H%M%S")
        df = df[ df.time < stop_time_formated]

    #transform time and IP address into formal type
    df["srcIP"] = df["srcIP"].swifter.set_npartitions(NPARTS).apply(ip2int)
    df["dstIP"] = df["dstIP"].swifter.set_npartitions(NPARTS).apply(ip2int)
    
    #aggregate nodes according to /20, build dictionary
    df['srcIP'] = df['srcIP'].swifter.set_npartitions(NPARTS).apply(lambda x: x >> 12)
    df['dstIP'] = df['dstIP'].swifter.set_npartitions(NPARTS).apply(lambda x: x >> 12)

    # Drop time column and get rid of duplicates
    # Convert to pandas to drop (faster)
    df = df._to_pandas()
    df = df.drop(columns=['time'])
    df = df.drop_duplicates()
    
    # shared dictionary, using across threads will mess it up
    #renumber into 0, 1, 2, ..
    IP_dict = {}
    df["srcIP"] = df["srcIP"].apply(lambda x : search_dict(x, IP_dict))
    df["dstIP"] = df["dstIP"].apply(lambda x : search_dict(x, IP_dict))

    #write into h5py files
    num_nodes = len(IP_dict)
    num_edges = df.shape[0]

    edge_index = np.array(df[["srcIP", "dstIP"]]).T

    return edge_index, num_nodes, num_edges


def prepare_background(f, dst_dir, dst_name, graph_id, start_time, stop_time):
    '''
    Transform txt files into standard hdf5 format
    arg = [txt_file_name, subgroup of graphs]
    '''

    edge_index, num_nodes, num_edges = prepare_background_(f, start_time, stop_time)
    
    f_h5py = h5py.File(os.path.join(dst_dir,dst_name), 'a')
    write_single_graph(f_h5py, 
                        graph_id = graph_id, 
                        x = np.ones([num_nodes, 1]), 
                        edge_index = edge_index,
                        y = np.zeros(num_nodes), 
                        attrs={'num_nodes': num_nodes, 'num_edges': num_edges, 'num_evils':0})
    f_h5py.close()


if __name__ == '__main__':
    # prepare_background('equinix-nyc.dirA.20181220-131256.UTC.anon.pcap', '.', 'tmp.hdf5', 0, '20181220081256', '20181220081257')
    prepare_background('/p/adversarialml/as9rw/datasets/raw_botnet/temp.tmp',
                       '/p/adversarialml/as9rw/datasets/raw_botnet', 'tmp.hdf5', 0, None, None)
