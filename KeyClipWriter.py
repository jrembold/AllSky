# Importing all the necessities
from collections import deque
from threading import Thread
from queue import Queue
import time
import cv2

class KeyClipWriter:
    def __init__(self, bufSize=64, timeout=1.0):
        # Maximum number of frames to be kept in memory
        self.bufSize = bufSize
        # Sleep timeout for threading to prevent lock competition
        self.timeout = timeout

        # Initialize everything
        self.frames = deque(maxlen=bufSize)
        self.Q = None
        self.writer = None
        self.thread = None
        self.recording = False


    def update(self, frame):
        # Add frame to the frame buffer
        self.frames.appendleft(frame)

        # If we are also recording, add the frame to the write queue as well
        if self.recording:
            self.Q.put(frame)


    def start(self, outputPath, fourcc, fps):
        # Set recording flag
        self.recording = True
        # Initialize writer
        self.writer = cv2.VideoWriter(outputPath, fourcc, fps, (self.frames[0].shape[1], self.frames[0].shape[0]), True)
        # Initialize queue of frames to be written
        self.Q = Queue()

        # Add everything currently in the frames buffer to the queue
        for i in range(len(self.frames), 0, -1):
            self.Q.put(self.frames[i-1])

        # Start a thread to write the frames
        self.thread = Thread(target=self.write, args=())
        self.thread.daemon = True
        self.thread.start()


    def write(self):
        # Start up the writing loop
        while True:
            # If done recording, exit thread
            if not self.recording:
                return

            # Are there frames in the queue?
            if not self.Q.empty():
                # Then grab the next frame and write it!
                frame = self.Q.get()
                self.writer.write(frame)
            # Otherwise, sleep for a moment to not interfere with the update process
            else:
                time.sleep(self.timeout)


    def flush(self):
        # Empty the queue by writing out the rest of what is in it
        time1 = time.time()
        while not self.Q.empty():
            frame = self.Q.get()
            self.writer.write(frame)
        time2 = time.time()
        # print("Flush elapsed time was: {}".format(time2-time1))


    def finish(self):
        # Set recording flag
        self.recording = False
        self.thread.join()
        self.flush()
        self.writer.release()


