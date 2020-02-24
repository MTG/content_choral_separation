#! /usr/bin/env python
# encoding: utf-8
# Copied from https://github.com/mauriciovander/silence-removal/blob/master/segment.py



import numpy
import scipy.io.wavfile as wf
import sys
from synth.config import config

# class VoiceActivityDetection:

#     def __init__(self, sr, ms, channel):
#         self.__sr = sr
#         self.__channel = channel
#         self.__step = int(sr/50)
#         self.__buffer_size = int(sr/50)
#         self.__buffer = numpy.array([],dtype=numpy.int16)
#         self.__out_buffer = numpy.array([],dtype=numpy.int16)
#         self.__n = 0
#         self.__VADthd = 0.
#         self.__VADn = 0.
#         self.__silence_counter = 0
#         self.__segment_count = 0
#         self.__voice_detected = False
#         self.__silence_thd_ms = ms

#         self.out_segments = []
#         self.out_segments_back = []

#     # Voice Activity Detection
#     # Adaptive threshold
#     def vad(self, _frame):
#         frame = numpy.array(_frame) ** 2.
#         result = True
#         threshold = 0.1
#         thd = numpy.min(frame) + numpy.ptp(frame) * threshold
#         self.__VADthd = (self.__VADn * self.__VADthd + thd) / float(self.__VADn + 1.)
#         self.__VADn += 1.

#         if numpy.mean(frame) <= self.__VADthd:
#             self.__silence_counter += 1
#         else:
#             self.__silence_counter = 0

#         if self.__silence_counter > self.__silence_thd_ms*self.__sr/(1000*self.__buffer_size):
#             result = False
#         return result

#     # Push new audio samples into the buffer.
#     def add_samples(self, data):
#         self.__buffer = numpy.append(self.__buffer, data)
#         result = len(self.__buffer) >= self.__buffer_size
#         # print('__buffer size %i'%self.__buffer.size)
#         return result

#     # Pull a portion of the buffer to process
#     # (pulled samples are deleted after being
#     # processed
#     def get_frame(self):

#         window = self.__buffer[:self.__buffer_size]
#         self.__buffer = self.__buffer[self.__step:]
#         # print('__buffer size %i'%self.__buffer.size)
#         return window

#     # Adds new audio samples to the internal
#     # buffer and process them
#     def process(self, data):
#         if self.add_samples(data):
#             while len(self.__buffer) >= self.__buffer_size:
#                 # Framing
#                 window= self.get_frame()
#                 # print('window size %i'%window.size)
#                 if self.vad(window):  # speech frame
#                     # print('voiced')
#                     self.__out_buffer = numpy.append(self.__out_buffer, window)
#                     self.__voice_detected = True
#                 elif self.__voice_detected:
#                     self.__out_buffer = numpy.append(self.__out_buffer, window)
#                     self.__voice_detected = False
#                     self.__segment_count = self.__segment_count + 1
#                     self.out_segments.append(self.__out_buffer)
#                     # wf.write('%s.%i.%i.wav'%(sys.argv[2],self.__channel,self.__segment_count),sr,self.__out_buffer)
#                     self.__out_buffer = numpy.array([],dtype=numpy.int16)
#                     # print(self.__segment_count)


#         return self.out_segments

#                 # print('__out_buffer size %i'%self.__out_buffer.size)

#     def get_voice_samples(self):
#         return self.__out_buffer

class VoiceActivityDetection:

    def __init__(self, sr, ms, channel):
        self.__sr = sr
        self.__channel = channel
        self.__step = int(sr/50)
        self.__buffer_size = int(sr/50) 
        self.__buffer_back = numpy.array([],dtype=numpy.int16)
        self.__buffer = numpy.array([],dtype=numpy.int16)
        self.__out_buffer_back = numpy.array([],dtype=numpy.int16) 
        self.__out_buffer = numpy.array([],dtype=numpy.int16)
        self.__n = 0
        self.__VADthd = 0.
        self.__VADn = 0.
        self.__silence_counter = 0
        self.__segment_count = 0
        self.__voice_detected = False
        self.__silence_thd_ms = ms

        self.out_segments = []
        self.out_segments_back = []

    # Voice Activity Detection
    # Adaptive threshold
    def vad(self, _frame):
        frame = numpy.array(_frame) ** 2.
        result = True
        threshold = 0.1
        thd = numpy.min(frame) + numpy.ptp(frame) * threshold
        self.__VADthd = (self.__VADn * self.__VADthd + thd) / float(self.__VADn + 1.)
        self.__VADn += 1.

        if numpy.mean(frame) <= self.__VADthd:
            self.__silence_counter += 1
        else:
            self.__silence_counter = 0

        if self.__silence_counter > self.__silence_thd_ms*self.__sr/(1000*self.__buffer_size):
            result = False
        return result

    # Push new audio samples into the buffer.
    def add_samples(self, data, back):
        self.__buffer = numpy.append(self.__buffer, data)
        self.__buffer_back = numpy.append(self.__buffer_back, back)
        result = len(self.__buffer) >= self.__buffer_size
        # print('__buffer size %i'%self.__buffer.size)
        return result

    # Pull a portion of the buffer to process
    # (pulled samples are deleted after being
    # processed
    def get_frame(self):

        window = self.__buffer[:self.__buffer_size]
        window_back = self.__buffer_back[:self.__buffer_size]
        self.__buffer = self.__buffer[self.__step:]
        self.__buffer_back = self.__buffer_back[self.__step:]
        # print('__buffer size %i'%self.__buffer.size)
        return window, window_back

    # Adds new audio samples to the internal
    # buffer and process them
    def process(self, data):
        back = numpy.arange(0, len(data)/config.fs, 1/config.fs)
        if self.add_samples(data, back):
            while len(self.__buffer) >= self.__buffer_size:
                # Framing
                window, window_back = self.get_frame()
                # print('window size %i'%window.size)
                if self.vad(window):  # speech frame
                    # print('voiced')
                    self.__out_buffer = numpy.append(self.__out_buffer, window)
                    self.__out_buffer_back = numpy.append(self.__out_buffer_back, window_back)
                    self.__voice_detected = True
                elif self.__voice_detected:
                    self.__out_buffer = numpy.append(self.__out_buffer, window)
                    self.__out_buffer_back = numpy.append(self.__out_buffer_back, window_back)
                    self.__voice_detected = False
                    self.__segment_count = self.__segment_count + 1
                    assert len(self.__out_buffer) == len(self.__out_buffer_back)
                    self.out_segments.append(self.__out_buffer)
                    self.out_segments_back.append(self.__out_buffer_back)
                    # wf.write('%s.%i.%i.wav'%(sys.argv[2],self.__channel,self.__segment_count),sr,self.__out_buffer)
                    self.__out_buffer = numpy.array([],dtype=numpy.int16)
                    self.__out_buffer_back = numpy.array([],dtype=numpy.int16)
                    # print(self.__segment_count)


        return self.out_segments, self.out_segments_back

                # print('__out_buffer size %i'%self.__out_buffer.size)

    def get_voice_samples(self):
        return self.__out_buffer