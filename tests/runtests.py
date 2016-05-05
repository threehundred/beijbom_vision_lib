import test_misc
import test_caffe
import unittest

suite = unittest.TestSuite([
	unittest.TestLoader().loadTestsFromTestCase(test_caffe.TestCaffeTools),
	unittest.TestLoader().loadTestsFromTestCase(test_misc.TestMiscTools) 
	])
unittest.TextTestRunner(verbosity=2).run(suite)