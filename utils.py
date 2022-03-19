import cv2 as cv
import time









class TimeTracker:
    def __init__(self) -> None:
        self.primetime = time.time()
        self.ptime2 = time.time() ### track the seconds passes
    
    def current_time(self):
        return time.time()

    def fps_calculate(self):
        fps = 1 / (self.current_time() - self.primetime)
        self.primetime = self.current_time()
        return fps
    
    def format_time(self, format: str="%y_%m_%d_%H_%M_%S"):
        return time.strftime(format, time.localtime(self.current_time()))

    def time_pass(self, time_pass_ms):
        if (self.current_time() - self.ptime2)*1000.0 > time_pass_ms:
            self.ptime2 = self.current_time()
            return True







