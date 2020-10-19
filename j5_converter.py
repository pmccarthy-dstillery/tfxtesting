"""
Write sgd++ sparse files into equivalent TFRecord files containing sparse features

`python j5_converter.py myfile.bin`
outputs myfile.bin.tfrecord
"""

import argparse
import array
import logging
import numpy as np
import os
import scipy as sp
import struct
import sys
import tensorflow as tf

def bin_to_example_generator(filepath):

    with open(filepath,'rb') as f:
        f.seek(0)
        while True:
            try:            
                length = struct.unpack('I', f.read(4))[0]
                values = array.array('f')
                values.fromfile(f,length)                
                indices = array.array('i')
                indices.fromfile(f, length)                

                label = struct.unpack('f', f.read(4))[0]

                sparse_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                            'indices': tf.train.Feature(int64_list=tf.train.Int64List(value=indices)),
                            'values': tf.train.Feature(float_list=tf.train.FloatList(value=values))
                        }
                    )
                )
                yield sparse_example

            except EOFError as e:
                break

def line_limit(nlines: int):

    if nlines == -1:
        write_nlines = sys.maxsize
    else:
        write_nlines = nlines

    def bin2tfrecord(binfile,space=int(10e6)):
        """
        Interpret a binary file and on the fly write it to a compressed tfrecord with the same name.
        """
        line_str = ''
        if nlines != -1:
            line_str = str(nlines)

        tfrecord_file = f"{binfile}.{line_str}.tfrecord.gz"

        tf_generator = bin_to_example_generator(binfile) 

        logging.info("writing out %s...", tfrecord_file)
        with tf.io.TFRecordWriter(tfrecord_file, options='GZIP') as f:
            for i, record in enumerate(tf_generator):

                if i < write_nlines:
                    f.write(record.SerializeToString())
                else: 
                    break

            logging.info("%s lines written", i)
        return tfrecord_file

    return bin2tfrecord


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--lines",
        default=-1,
        type=int
    )
    parser.add_argument(
        "-i",
        "--infiles",
        nargs="*",
        required=True
    )

    args = vars(parser.parse_args())
    print(args)
    # sys.exit()


    ## Test that every member of the inputs is a valid file
    try:
        assert all(map(os.path.exists,args['infiles'])),"One or more input paths not valid"
    except AssertionError as e:
        logging.error("One or more input files not found")
        logging.error([x for x in zip(args['infiles'], map(os.path.exists, args['infiles']))])

    binarizer = line_limit(args['lines'])
    ## map a conversion function across inputs
    tf_record_files = map(binarizer, args['infiles'])

    logging.info("Following files are written out: %s",",".join([x for x in tf_record_files]))



