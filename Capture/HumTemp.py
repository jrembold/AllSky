import Adafruit_DHT as dht
import time
import logging

import shared

# Set up logging
logger = logging.getLogger('weather')
formatter = logging.Formatter('[{asctime}] {message}',
        datefmt='%Y/%m/%d %H:%M:%S', style='{')
fhandler = logging.FileHandler('Logs/InternalConditions.log')
fhandler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(fhandler)

def log_weather(delay):
    while True:
        h,t = dht.read_retry(dht.DHT22, 24)
        shared.TEMP = t*9/5+32
        shared.HUMID = h
        logger.info('Temp: {:0.2f}F, Hum: {:0.2f}%'.format(t*9/5+32,h))
        time.sleep(delay)

if __name__ == '__main__':
    log_weather(10)
