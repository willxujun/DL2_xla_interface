import tensorflow as tf
import tf2xla_pb2
from common_tensor_pb2 import CommonTensor
from common_tensor_pb2 import CommonTensors
import google.protobuf.text_format as text
import numpy as np
import os

config = tf2xla_pb2.Config()
with open("test_graph_tfmatmulandadd.config.pbtxt") as f:
    config_str = f.read()
    text.Merge(text=config_str, message=config)

graph_def = tf.GraphDef()
names = []
with open('test_graph_tfmatmulandadd.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())

tf.graph_util.import_graph_def(graph_def=graph_def, name="")
graph = tf.get_default_graph()

writer = tf.summary.FileWriter('vis', graph)

# saves tensors to .npy, one file per tensor
def save(values, rel_target_dir_path):
    def prepare_common_tensor(val, name):
        ret = CommonTensor()
        ret.name = name
        ret.shape.extend(list(val.shape))
        ret.data.extend(val.flatten().tolist())
        return ret

    abs_curr_path = os.path.abspath(os.path.dirname(__file__))
    abs_target_dir_path = os.path.join(abs_curr_path, rel_target_dir_path)
    abs_target_file_path = os.path.join(abs_target_dir_path, "common_tensors.pb")
    
    tensors = []
    for (val, feed) in zip(values, config.feed):
        name = feed.id.node_name
        tensors.append(prepare_common_tensor(val,name))
    tensors_to_save = CommonTensors()
    tensors_to_save.data.extend(tensors)
    
    with open(abs_target_file_path, 'wb') as f:
        f.write(tensors_to_save.SerializeToString())

def list_common_tensors(common_tensors):
    return [{'name':t.name, 'data':t.data, 'shape':t.shape} for t in common_tensors.data]

def run_graph_and_save(in_1, in_2):
    with tf.Session() as sess:
        _fetches = [n.id.node_name+':0' for n in config.feed]
        _feed_dict = {graph.get_tensor_by_name("x_hold:0"):in_1, graph.get_tensor_by_name("y_hold:0"):in_2}
        _to_cpp = sess.run(fetches=_fetches, feed_dict=_feed_dict)
        print(_to_cpp)

        # assume order of list returned by sess.run() follows order of config.feed which follows order of config.pbtxt
        # save the list to individual .npy files
        save(_to_cpp, "save")

def main():
    print(graph.get_tensor_by_name("x_hold:0").shape)
    run_graph_and_save([[1,2],[3,4]], [[5,6],[7,8]])
    with open("save/common_tensors.pb", 'rb') as f:
        debug = CommonTensors()
        debug.ParseFromString(f.read())
        print(list_common_tensors(debug))

if __name__ == "__main__": main()