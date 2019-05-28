import argparse
import os
import sys

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import app
from tensorflow.python.training import saver as saver_lib

FLAGS = None

def tfmatmulandadd(_):
  # This tests multiple outputs.
  x = array_ops.placeholder(dtypes.float32, name='x_hold')
  y = array_ops.placeholder(dtypes.float32, name='y_hold')
  p = math_ops.matmul(x, y,name='x_y_prod')
  q = math_ops.add(x, y,name='x_y_sum')
  r = math_ops.matmul(p, q,name='prod')
  print(r)

def write_graph(build_graph, out_dir):
    """Build a graph using build_graph and write it out."""
    g = ops.Graph()
    with g.as_default():
        build_graph(out_dir)
        filename = os.path.join(out_dir, 'test_graph_%s.pb' % build_graph.__name__)
        with open(filename, 'wb') as f:
            f.write(g.as_graph_def().SerializeToString())

def main(_):
    tfmatmulandadd(0)
    # launch the default graph
    sess = tf.Session()

    writer = tf.summary.FileWriter('vis', sess.graph)

    write_graph(tfmatmulandadd, FLAGS.out_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--out_dir',
      type=str,
      default='',
      help='Output directory for graphs, checkpoints and savers.')
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
