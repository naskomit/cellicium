import logging

FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
logging.basicConfig(format = FORMAT, level = logging.INFO, datefmt = '%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('cellicium')
