import test_caffe
import test_misc
import sys

if 'caffe' in sys.argv:
    test_caffe.run()

if 'misc' in sys.argv:
    test_misc.run()