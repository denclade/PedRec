from queue import Queue
from threading import Thread
from typing import Union

import cv2
import numpy as np

from pedrec.models.data_structures import ImageSize
from pedrec.utils.input_providers.input_provider_base import InputProviderBase


class VideoProvider(InputProviderBase):
    def __init__(self,
                 camera: Union[int, str] = 0,
                 image_size: ImageSize = ImageSize(width=1280, height=720),
                 mirror: bool = False):
        """
        Provides frames captured from a webcam. Uses OpenCV internally.
        :param camera: The cameras id or path to the video file
        :param image_size: The image size which should be used, may be limited by camera parameters
        :param fps: The fps on which the frames should be grabbed, may be limited by camera parameters
        """
        self.first_frame = True
        self.requires_resize = isinstance(camera, str)
        self.cap = cv2.VideoCapture(camera)
        self.mirror = mirror
        self.image_size = image_size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size.height)

    def get_data(self) -> np.ndarray:
        assert self.cap.isOpened(), 'Cannot capture source'
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.first_frame:
                if frame.shape[1] == self.image_size.width and frame.shape[0] == self.image_size.height:
                    self.requires_resize = False
                self.first_frame = False
            if self.requires_resize:
                frame = cv2.resize(frame, (self.image_size.width, self.image_size.height))
            if self.mirror:
                frame = cv2.flip(frame, 1)
            yield frame
        self.cap.release()

    def stop(self):
        self.cap.release()


class WebcamProviderAsync(InputProviderBase):
    def __init__(self,
                 camera_number: int = 0,
                 image_size: ImageSize = ImageSize(width=1280, height=720),
                 fps: int = 60,
                 queue_size=128):
        """
        Provides frames captured from a webcam. Uses OpenCV internally.
        :param camera_number: The cameras id
        :param image_size: The image size which should be used, may be limited by camera parameters
        :param fps: The fps on which the frames should be grabbed, may be limited by camera parameters
        """
        self.cap = cv2.VideoCapture(camera_number)
        self.stopped = True
        self.queue = Queue(maxsize=queue_size)
        self.image_size = image_size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size.height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def start(self):
        self.stopped = False
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        assert self.cap.isOpened(), 'Cannot capture source'

        while self.cap.isOpened():
            if self.stopped:
                return None
            if not self.queue.full():
                ret, frame = self.cap.read()
                if ret:
                    self.queue.put(frame)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        self.stop()
                        return None

                else:
                    self.stop()
                    return None

        self.cap.release()

    def get_data(self) -> np.ndarray:
        assert self.cap.isOpened(), 'Cannot capture source'
        if self.stopped:
            self.start()
        while True:
            if self.stopped:
                return None
            if self.queue.qsize() > 0:
                yield self.queue.get()

    def stop(self):
        self.stopped = True
        self.cap.release()
