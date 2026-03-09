# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: ZHAO Kuangshi, zhaokuangshi@neuracle.cn
# shaolichen@neuracle.cn
#
# Copyright (c) 2022 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/
import socket
import time
from enum import Enum
from threading import Thread, Lock
from struct import unpack
import numpy as np
import tempfile
from pyedflib import highlevel
import pickle


class BaseBuffer:
    def __init__(self, n_chan, n_points):
        self.n_chan = n_chan
        self.n_points = n_points
        self.buffer = np.zeros((n_chan, n_points))
        self.lastPtr = 0
        self.currentPtr = 0
        self.nUpdate = 0
        self.bufferLock = Lock()

    # reset buffer
    def resetBuffer(self):
        self.bufferLock.acquire()
        self.buffer = np.zeros((self.n_chan, self.n_points))
        self.currentPtr = 0
        self.nUpdate = 0
        self.bufferLock.release()


class DoubleBuffer(BaseBuffer):
    """
    DoubleBuffer is a buffer with two "chunks" of buffer: one acts as "front"
    buffer who receives new data and actively refresh its content, another acts
    as "back" buffer who temporarily stores a data chunk, waits to be flushed
    into disk (or a temp file) and never to be changed.

    DoubleBuffer using two buffers to balance performance and memory usage.
    """

    def __init__(self, n_chan: int, n_points: int = 300000):
        super(DoubleBuffer, self).__init__(n_chan, n_points)
        self.backBuffer = np.zeros((n_chan, n_points))
        self.backBufferLock = Lock()
        self.tempfile = []
        self.cached = True  # indicate if a back buffer has been cached
        self.backBufferRemain = False  # if newly updated data remain on
        # the back buffer, i.e., not read
        # by self.getUpdate()
        self.firstTime = True  # indicate the first cache

    def flip(self):
        # print('flip')
        # check if original back buffer has been cached
        assert self.cached
        # assert not self.backBufferRemain)
        # flip
        self.bufferLock.acquire()
        self.backBufferLock.acquire()
        self.backBuffer = self.buffer
        self.cached = False
        self.buffer = np.zeros((self.n_chan, self.n_points))
        self.backBufferRemain = True
        self.currentPtr = 0
        self.bufferLock.release()
        self.backBufferLock.release()
        # caching data
        Thread(target=self.caching).start()

    def caching(self):
        self.backBufferLock.acquire()
        assert not self.cached
        tf = tempfile.TemporaryFile()
        '''
        if self.firstTime:
            data = self.backBuffer
        else:
            self.tempfile.seek(0)
            tem = np.load(self.tempfile, allow_pickle=True)
            data = np.hstack([tem, self.backBuffer])
        self.tempfile.seek(0)
        data.dump(self.tempfile)
        '''
        self.backBuffer.dump(tf)
        self.cached = True
        self.tempfile.append(tf)
        self.backBufferLock.release()
        self.firstTime = False

    def appendBuffer(self, data):
        """
        Append buffer and update current pointer.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            New data chunk to be updated.
        """
        # CAUTION: Cannot write anything to backbuffer!
        n = data.shape[1]
        if self.currentPtr + n > self.n_points:  # full buffer
            # buffering data at the tail first
            tail = data[:, :self.n_points - self.currentPtr]
            data = data[:, self.n_points - self.currentPtr:]
            n -= self.n_points - self.currentPtr
            self.bufferLock.acquire()
            self.buffer[:, self.currentPtr:] = tail
            self.bufferLock.release()
            # then flip two buffer
            self.flip()
        # XXX: do not consider the situation that received data chunk is much
        #      longer than self.n_points. Please use a large enough buffer
        #      length.
        self.bufferLock.acquire()
        self.buffer[:, self.currentPtr:self.currentPtr + n] = data
        self.currentPtr = self.currentPtr + n
        self.nUpdate = self.nUpdate + n
        self.bufferLock.release()

    def getUpdate(self):
        self.bufferLock.acquire()
        if self.backBufferRemain:  # first to read backbuffer
            self.backBufferLock.acquire()
            data = self.backBuffer[:, self.lastPtr:]
            self.backBufferRemain = False
            self.backBufferLock.release()
            data = np.hstack([data, self.buffer[:, :self.currentPtr]])
        else:
            data = self.buffer[:, self.lastPtr:self.currentPtr]
        self.lastPtr = self.currentPtr
        self.nUpdate = 0
        self.bufferLock.release()
        return data

    def getData(self):
        self.bufferLock.acquire()
        self.backBufferLock.acquire()
        diskData = []
        for tf in self.tempfile:
            tf.seek(0)
            diskData.append(np.load(tf, allow_pickle=True))
        if self.cached:
            '''
            self.tempfile.seek(0)
            diskData = np.load(self.tempfile, allow_pickle=True)
            '''
            memData = self.buffer[:, :self.currentPtr]
        else:
            '''
            self.tempfile.seek(0)
            diskData = np.load(self.tempfile, allow_pickle=True)
            '''
            memData = np.hstack([self.backBuffer, self.buffer[:, :self.currentPtr]])
        data = np.hstack([*diskData, memData])
        self.backBufferLock.release()
        self.bufferLock.release()
        return data


class RingBuffer(BaseBuffer):
    def appendBuffer(self, data):
        """
        Append buffer and update current pointer.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            New data chunk to be updated.
        """
        n = data.shape[1]
        self.bufferLock.acquire()
        self.buffer[:, np.mod(np.arange(self.currentPtr, self.currentPtr + n), self.n_points)] = data
        self.currentPtr = np.mod(self.currentPtr + n - 1, self.n_points) + 1
        self.nUpdate = self.nUpdate + n
        self.bufferLock.release()

    def getUpdate(self):
        self.bufferLock.acquire()
        if self.nUpdate <= self.n_points:
            if self.lastPtr <= self.currentPtr:
                data = self.buffer[:, self.lastPtr:self.currentPtr]
            else:
                data = np.hstack([self.buffer[:, self.lastPtr:], self.buffer[:, :self.currentPtr]])
        else:
            data = np.hstack([self.buffer[:, self.currentPtr:], self.buffer[:, :self.currentPtr]])
        self.lastPtr = self.currentPtr
        self.nUpdate = 0
        self.bufferLock.release()
        return data

    def getData(self):
        self.bufferLock.acquire()
        data = np.hstack([self.buffer[:, self.currentPtr:], self.buffer[:, :self.currentPtr]])
        self.bufferLock.release()
        return data


def resolveMeta(raw: bytes) -> dict:
    """
    Parse the meta packet.
    :param raw: Raw bytes of the meta packet
    :return:
    """
    # Parse HeadLength
    headerLength = int.from_bytes(raw[2:6], byteorder="little", signed=False)
    head = raw[:headerLength]
    # Parse HeadToken, HeaderLength, TotalLength, Flag, ModuleCount (total length equals HeaderLength)
    # < means little-endian byte order
    # H means 1 unsigned short (2 bytes)
    # 4I means 4 unsigned ints (4 bytes each)
    # See https://docs.python.org/3/library/struct.html
    print(len(head))

    _, headerLength, totalLength, flag, moduleCount = unpack("<H4I", head)
    # _, headerLength, totalLength, flag, moduleCount = unpack("<4I", head)
    # Parse the remaining part
    body = raw[headerLength:]
    # Number of modules (probes)
    D = moduleCount
    # Starting offset for each module's data packet; unpack returns a tuple
    moduleOffsets = unpack(f"<{D}I", body[: 4 * D])
    # Raw byte data for each module
    eachModuleData = []
    for m in range(D):
        offset = moduleOffsets[m]
        if m < D - 1:
            # For the first D-1 modules, data range is from this module's offset to the next module's offset
            end = moduleOffsets[m + 1]
        else:
            # For the last module, data range is from its offset to the end of the meta packet minus TailToken
            end = totalLength - 2
        eachModuleData.append(raw[offset:end])
    # Parse meta data for each module; key is serial number, value is the parsed meta data
    modules = {}
    for m in range(D):
        module = resolveMetaEachModule(eachModuleData[m])
        key = module['serialNumber']
        modules[key] = module
    # Result
    result = {
        "flag": flag,
        "moduleCount": moduleCount,
        "modules": modules
    }
    return result


def resolveMetaEachModule(fragment: bytes) -> dict:
    """
    Parse information for each module in the meta packet.
    :param fragment: Raw bytes of each module's data
    :return:
    """
    # Parse PersonName, ModuleName, ModuleType, SerialNumber, ChannelCount
    personName, moduleName, moduleType, serialNumber, channelCount = unpack("<30s30s30s2I", fragment[:98])
    # Strip trailing null characters
    personName = personName.decode("utf8").strip("\x00")
    moduleName = moduleName.decode("utf8").strip("\x00")
    moduleType = moduleType.decode("utf8").strip("\x00")
    # Channel names
    channelNames = list(unpack("10s" * channelCount, fragment[98: 98 + 10 * channelCount]))
    channelNames = [b.decode("utf8").strip("\x00") for b in channelNames]
    # Channel types
    channelTypes = list(unpack("10s" * channelCount, fragment[98 + 10 * channelCount: 98 + 20 * channelCount]))
    channelTypes = [b.decode("utf8").strip("\x00") for b in channelTypes]
    # Sample rates
    sampleRates = list(unpack(f"<{channelCount}I", fragment[98 + 20 * channelCount: 98 + 24 * channelCount]))
    # Data count per channel
    dataCountPerChannel = list(unpack(f"<{channelCount}I", fragment[98 + 24 * channelCount: 98 + 28 * channelCount]))
    # Max digital value per channel
    maxDigital = list(unpack(f"<{channelCount}i", fragment[98 + 28 * channelCount: 98 + 32 * channelCount]))
    # Min digital value per channel
    minDigital = list(unpack(f"<{channelCount}i", fragment[98 + 32 * channelCount: 98 + 36 * channelCount]))
    # Max physical (analog) value per channel
    maxPhysical = list(unpack(f"<{channelCount}f", fragment[98 + 36 * channelCount: 98 + 40 * channelCount]))
    # Min physical (analog) value per channel
    minPhysical = list(unpack(f"<{channelCount}f", fragment[98 + 40 * channelCount: 98 + 44 * channelCount]))
    # Gain per channel
    gain = list(unpack(f"{channelCount}c", fragment[98 + 44 * channelCount: 98 + 45 * channelCount]))
    # Result
    result = {"personName": personName,
              "moduleName": moduleName,
              "moduleType": moduleType,
              "serialNumber": serialNumber,
              "channelCount": channelCount,
              "channelNames": channelNames,
              "channelTypes": channelTypes,
              "sampleRates": sampleRates,
              "dataCountPerChannel": dataCountPerChannel,
              "maxDigital": maxDigital,
              "minDigital": minDigital,
              "maxPhysical": maxPhysical,
              "minPhysical": minPhysical,
              "gain": gain}
    return result


def isChannelNotAllEEG(channelTypes: list) -> bool:
    """
    Check if all channels are non-EEG types.
    :param channelTypes: List of all channel types
    :return:
    """
    for channelType in channelTypes:
        # If any channel is EEG, return False
        if channelType == 'EEG':
            return False
    # All channels are non-EEG, return True
    return True


def resolveData(raw: bytes, meta: dict) -> dict:
    """
    Parse a data packet.
    :param raw: Raw bytes of the data packet
    :param meta: Previously parsed meta packet
    :return:
    """
    # Header length, fixed at 30
    headerLength = int.from_bytes(raw[2:6], byteorder="little", signed=False)
    # Parse HeadToken, HeaderLength, TotalLength, StartTimestamp, TimeStampLength, TriggerCount, Flag, ModuleCount
    head = raw[:headerLength]
    # HeadToken is unsigned short, the rest are unsigned ints
    _, headerLength, totalLength, startTimeStamp, timeStampLength, triggerCount, flag, moduleCount = unpack("<H7I", head)
    # Parse the remaining part
    body = raw[headerLength:]
    # D and T follow the naming convention used in the protocol
    D = moduleCount
    T = triggerCount
    # Offset of each module in the data packet; moduleOffsets is a tuple
    moduleOffsets = unpack(f"<{D}I", body[: 4 * D])
    # For the first D-1 modules, data range is from this module's offset to the next module's offset
    # For the last module, data range is from its offset to the end minus TriggerTimestamps, Triggers, and TailToken
    eachModuleData = []
    for m in range(D):
        offset = moduleOffsets[m]
        if m < D - 1:
            end = moduleOffsets[m + 1]
        else:
            # TriggerTimestamps, Triggers, TailToken are 4*T, 30*T, and 2 bytes respectively
            end = totalLength - 2 - 34 * T
        eachModuleData.append(raw[offset:end])
    # Data for each module, including serial number, Bitmask, and Datas
    modules = {}
    for m in range(D):
        module = resolveDataEachModule(eachModuleData[m], meta)
        key = module['serialNumber']
        modules[key] = module
    # Result
    result = {
        "flag": flag,
        "startTimeStamp": startTimeStamp,
        "timeStampLength": timeStampLength,
        "moduleCount": moduleCount,
        "modules": modules
    }
    return result


def resolveDataEachModule(fragment: bytes, meta: dict) -> dict:
    """
    Parse each module's data in bulk forwarding mode.
    :param fragment: Raw bytes of this module's data
    :param meta: Previously parsed meta data
    :return:
    """
    # Serial number of this module
    serialNumber = int.from_bytes(fragment[:4], byteorder="little", signed=False)
    # Number of channels
    N = meta["modules"][serialNumber]["channelCount"]
    # Number of data points per channel
    dataCountPerChannel = meta["modules"][serialNumber]["dataCountPerChannel"]
    # Whether each channel has data
    bitmask = list(unpack(f"{N}?", fragment[4:4 + N]))
    # data is a list with shape: num_channels * points_per_channel
    data = []
    raw = list(unpack(f"<{sum(dataCountPerChannel)}f", fragment[4 + N:]))
    # Split the long continuous raw data into num_channels * points_per_channel based on each channel's point count
    cursor = 0
    for count in dataCountPerChannel:
        data.append(raw[cursor: cursor + count])
        cursor += count
    # Result
    result = {
        "serialNumber": serialNumber,
        "bitmask": bitmask,
        "data": data
    }
    return result


class ConnectState(Enum):
    # the first state with nothing ready
    NOTCONNECT = 0
    # already connect the data server, demonstrating at least
    # there is an available TCP port
    CONNECTED = 1
    # successfully receive and resolve the META packet, and
    # open the data flow in order to stabilize it. (data will
    # not be stored into buffer)
    READY = 2
    # receiving data at present
    RUNNING = 3
    # Abort connection
    ABORT = 4


class DataServerThread:
    """
    Users only need to call this class.
    """

    def __init__(self, sample_rate: int = 1000, t_buffer: float = 60):
        """
        Initialize.
        :param sample_rate: Sampling rate
        :param t_buffer: Buffer length (seconds)
        """
        # Sampling rate
        self.sample_rate = sample_rate
        # Buffer length (seconds)
        self.t_buffer = t_buffer
        # Initialize as not connected
        self.state = ConnectState.NOTCONNECT
        # Transition period to stabilize data before formally receiving
        self.stabilizeCount = 0
        self.pointsForStabilize = 50
        # Start and end timestamps
        self.firstTimestamp = -1
        self.lastTimestamp = -1
        # Whether the meta packet has been received
        self.__hasMeta = False
        # Lock for the socket connection buffer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socketBuffer = bytes()
        self.sockBufLock = Lock()
        # Cache for per-module forwarded data packets
        self.single_module_data_buffer = []
        # Cache for per-module forwarded trigger packets
        self.single_module_trigger_buffer = []
        # Max cached per-module forwarded data packets
        self.max_single_packet = 20
        # Total number of received packets
        self.packet_count = 0
        # Timestamps of received packets, separated by data and trigger
        self.timeStamp = {'data': [], 'trigger': []}
        # # For real-time output of triggers in per-module forwarding, for comparison with JellyFish
        # self.trigger_count = 0

    def connect(self, hostname: str = '127.0.0.1', port: int = 8712):
        """
        Connect to JellyFish.
        :param hostname: IP of the computer running JellyFish; use 127.0.0.1 if running locally
        :param port: Port opened by JellyFish, default is 8712
        :return:
        """
        self.hostname = hostname
        self.port = port
        # Reconnection count
        reconnect_time = 10
        # Keep trying to connect until successful
        while self.state == ConnectState.NOTCONNECT:
            try:
                self.sock.connect((self.hostname, self.port))
                # Set to non-blocking mode
                self.sock.setblocking(False)
                # Set state to connected
                self.state = ConnectState.CONNECTED
                # Start the data reading thread
                self.openStream()
                print('connect success')
            except:
                # Output retry count on connection failure
                reconnect_time += 1
                print(f'connection failed, retrying for {reconnect_time} times')
                time.sleep(1)
                # Give up reconnecting if max retries exceeded
                if reconnect_time >= 1:
                    break
        # Return whether connection succeeded
        return self.state == ConnectState.NOTCONNECT

    def isReady(self):
        """
        Check whether data reading is ready (i.e., whether the meta packet has been parsed successfully).
        :return:
        """
        return self.state == ConnectState.READY

    def start(self):
        """
        Start receiving forwarded data.
        :return:
        """
        # Cannot start receiving data before meta packet is resolved
        if self.state == ConnectState.NOTCONNECT or self.state == ConnectState.CONNECTED:
            raise RuntimeError("Cannot start data recording before ready.")
        # Start once the meta packet has been resolved
        elif self.state == ConnectState.READY:
            self.state = ConnectState.RUNNING

    def stop(self):
        """
        Stop receiving data.
        :return:
        """
        self.state = ConnectState.ABORT

    def openStream(self):
        """
        Start the data reading threads.
        :return:
        """
        # Receiving and parsing must be in separate threads; otherwise 5ms per-module forwarding may lose packets
        Thread(target=self.readDataThread, daemon=True).start()
        Thread(target=self.resolveDataThread, daemon=True).start()

    def readDataThread(self):
        """
        Data receiving function executed in a separate thread.
        :return:
        """
        # Cannot receive data before connection is established
        if self.state == ConnectState.NOTCONNECT:
            raise RuntimeError("Cannot start receiving data before connect.")
        # Keep receiving data until stopped
        while self.state != ConnectState.ABORT:
            # Actual data receiving function
            self.receiveData()
            # Cannot sleep; 1ms sleep severely impacts 5ms per-module forwarding
            # time.sleep(0.001)

    def resolveDataThread(self):
        """
        Data parsing thread.
        :return:
        """
        while True:
            if len(self.socketBuffer) > 0:
                self.resolve()

    def receiveData(self):
        """
        Actual data receiving function.
        :return:
        """
        buffer = bytes()
        retry = True
        while retry:
            # In non-blocking mode, recv raises an exception when no data is available
            try:
                # The socket read buffer size is variable.
                # Packet size is actually indeterminate; connecting many probes leads to large packets.
                # Hence 4096*10 to ensure the buffer is larger than any single packet.
                # Because packet size is uncertain, "incomplete packet" or "sticky packet" issues may occur:
                # either too little data (incomplete packet) or too much (multiple packets concatenated).
                # The 4096*10 logic ensures only "sticky packets" occur; resolve() handles splitting them.
                for i in range(10):
                    buffer += self.sock.recv(4096)
                    # if len(self.socketBuffer) > 0:
                    #     self.resolve()
            except:
                if len(buffer) > 0:
                    retry = False
                else:
                    retry = True
                    # time.sleep(0.001)
        # Lock needed since resolve uses self.socketBuffer
        self.sockBufLock.acquire()
        self.socketBuffer += buffer
        self.sockBufLock.release()
        # # Actual data parsing process
        # self.resolve()

    def isSingleModule(self):
        """
        Check whether this is a single-module setup.
        :return:
        """
        # Bulk forwarding has only 1 module
        if self.meta['moduleCount'] == 1:
            return True
        else:
            # Per-module forwarding should have 2 modules, one with SN=0
            if len(self.meta['modules'].keys()) == 2 and 0 in self.meta['modules'].keys():
                return True
            else:
                # Does not meet the above conditions; not a single-person single-module setup
                return False

    def mergeMetaTriggerModule(self):
        """
        In per-module forwarding, merge the trigger module's info into the high-speed module.
        :return:
        """
        # Channel count +1
        self.n_chan += 1
        # Append trigger module info
        self.srates.extend(self.meta['modules'][0]['sampleRates'])
        self.channelNames.extend(self.meta['modules'][0]['channelNames'])
        self.channelTypes.extend(self.meta['modules'][0]['channelTypes'])
        self.maxDigital.extend(self.meta['modules'][0]['maxDigital'])
        self.minDigital.extend(self.meta['modules'][0]['minDigital'])
        self.maxPhysical.extend(self.meta['modules'][0]['maxPhysical'])
        self.minPhysical.extend(self.meta['modules'][0]['minPhysical'])
        self.gain.extend(self.meta['modules'][0]['gain'])
        self.dataCountPerChannel.extend(self.meta['modules'][0]['dataCountPerChannel'])

    def resolve(self):
        """
        Actual data parsing process. Protocol defined in <Data Forwarding Protocol - New Version.xlsx>.
        :return:
        """
        # Acquire lock since socketBuffer will be used
        self.sockBufLock.acquire()
        # Determine whether the received packet is meta or data
        # HeadToken: 0x5FF5 for meta packets, 0x5AA5 for data packets
        metaHeadToken = bytes.fromhex('5FF5')
        dataHeadToken = bytes.fromhex('5AA5')
        headToken = self.socketBuffer[0:2]
        if headToken == metaHeadToken:
            # Meta packet
            isMeta = True
        elif headToken == dataHeadToken:
            # Data packet
            isMeta = False
        else:
            # Invalid packet, raise error
            raise ValueError(f"Invalid head token \"{headToken.hex()}\"")
        # Get full packet length from TotalLength
        totalLength = int.from_bytes(self.socketBuffer[6:10], byteorder="little", signed=False)
        msg = self.socketBuffer[:totalLength]
        # Check TailToken: 0xF55F for meta packets, 0xA55A for data packets
        metaTailToken = bytes.fromhex('F55F')
        dataTailToken = bytes.fromhex('A55A')
        tailToken = msg[-2:]
        # Both HeadToken and TailToken must match for meta packets
        if isMeta and tailToken != metaTailToken:
            raise ValueError(f"Invalid tail token \"{tailToken.hex()}\"")
        # Both HeadToken and TailToken must match for data packets
        elif not isMeta and tailToken != dataTailToken:
            raise ValueError(f"Invalid tail token \"{tailToken.hex()}\"")
        # Clear the current socketBuffer
        self.socketBuffer = self.socketBuffer[totalLength:]
        # Packet length does not match TotalLength
        if totalLength != len(msg):
            raise ValueError('The message is not as long as it assigned!')
        self.sockBufLock.release()
        # Parse meta packet
        if isMeta:
            if self.__hasMeta:
                # JellyFish may still be sending meta packets before receiving our OK confirmation
                # Skip if meta packet has already been parsed
                pass
            else:
                # Parse meta packet
                self.meta = resolveMeta(msg)
                # Raise error if not a single module
                if not self.isSingleModule():
                    raise Exception('Only single-person single-module is supported!')
                # Get the parsed data
                # Find the non-zero SN; if we reach here, a non-zero SN must exist
                for sn in self.meta['modules'].keys():
                    if sn != 0:
                        self.serialNumber = sn
                # For bulk forwarding, retrieve info directly
                self.n_chan = self.meta['modules'][self.serialNumber]['channelCount']
                # Copy to prevent per-module merge from modifying self.meta, which would break data parsing
                self.srates = self.meta['modules'][self.serialNumber]['sampleRates'].copy()
                self.channelNames = self.meta['modules'][self.serialNumber]['channelNames'].copy()
                self.channelTypes = self.meta['modules'][self.serialNumber]['channelTypes'].copy()
                self.maxDigital = self.meta['modules'][self.serialNumber]['maxDigital'].copy()
                self.minDigital = self.meta['modules'][self.serialNumber]['minDigital'].copy()
                self.maxPhysical = self.meta['modules'][self.serialNumber]['maxPhysical'].copy()
                self.minPhysical = self.meta['modules'][self.serialNumber]['minPhysical'].copy()
                self.gain = self.meta['modules'][self.serialNumber]['gain'].copy()
                self.dataCountPerChannel = self.meta['modules'][self.serialNumber]['dataCountPerChannel'].copy()
                self.personName = self.meta['modules'][self.serialNumber]['personName']
                self.moduleName = self.meta['modules'][self.serialNumber]['moduleName']
                self.moduleType = self.meta['modules'][self.serialNumber]['moduleType']
                if len(self.meta['modules'].keys()) == 2:
                    # Per-module forwarding: merge trigger info into the high-speed module
                    self.mergeMetaTriggerModule()
                # Raise error if all channels are non-EEG types
                if isChannelNotAllEEG(self.channelTypes):
                    raise Exception('All channels are non-EEG types!')
                # Buffer length
                nPoints = int(np.round(self.t_buffer * self.sample_rate))
                # Number of channels
                nChans = len(self.channelNames)
                # Initialize RingBuffer
                self.buffer = RingBuffer(nChans, nPoints)
                # DoubleBuffer stores data for correctness verification
                self.save_buffer = DoubleBuffer(nChans, nPoints)
                # Send metadata received OK confirmation packet
                succ = bytes.fromhex('F55F5FF5')
                self.sock.send(succ)
                self.__hasMeta = True
        # Parse data packet
        else:
            if not self.__hasMeta:
                # Raise error if meta packet has not been resolved yet
                raise RuntimeError("Wrong program. Receive data before meta.")
            # Consider data ready for parsing once stabilized
            if self.state == ConnectState.CONNECTED and self.stabilizeCount >= self.pointsForStabilize:
                self.state = ConnectState.READY
                # No longer needed after stabilization, reset
                self.stabilizeCount = 0
            # Data not yet stable
            elif self.state == ConnectState.CONNECTED:
                self.stabilizeCount += 1
                return
            # Data packet parse result
            dataStruct = resolveData(msg, self.meta)
            self.packet_count += 1
            # Bulk forwarding
            if self.meta['flag'] % 2 == 0:
                self.isDataPacketLost(dataStruct)
                dataArr = dataStruct['modules'][self.serialNumber]['data']
                tempBuf = []
                for ch in range(self.n_chan):
                    tempBuf.append(dataArr[ch])
                # Resample trigger channel
                tempBuf = self.ResampleTrigger(tempBuf)
                tempBuf = np.array(tempBuf)
                if self.state == ConnectState.RUNNING:
                    # Append data to RingBuffer
                    self.buffer.appendBuffer(tempBuf)
                    # Also append to DoubleBuffer for verification
                    self.save_buffer.appendBuffer(tempBuf)
                    # Timestamp
                    self.timeStamp['data'].append(dataStruct['startTimeStamp'])
            # Per-module forwarding requires packet assembly
            else:
                sn = list(dataStruct['modules'].keys())[0]
                # Trigger channel has SN=0
                if sn == 0:
                    # Accumulate trigger timestamps
                    self.timeStamp['trigger'].append(dataStruct['startTimeStamp'])
                    # Cache trigger packet
                    self.single_module_trigger_buffer.append(dataStruct)
                    # Output received trigger packet info
                    # self.trigger_count += 1
                    # print('trigger count:', self.trigger_count, 'trigger value:', dataStruct['modules'][0]['data'][0], 'time stamp:',
                    #       dataStruct['startTimeStamp'])
                else:
                    self.isDataPacketLost(dataStruct)
                    # Accumulate data packet timestamps
                    self.timeStamp['data'].append(dataStruct['startTimeStamp'])
                    # Cache data packet
                    self.single_module_data_buffer.append(dataStruct)
                # Assemble packets when cache is full
                if len(self.single_module_data_buffer) == self.max_single_packet:
                    self.combineDataAndTrigger()

    def ResampleTrigger(self, temBuf):
        """
        In bulk forwarding, when sample rate is not 1000 Hz, the trigger channel and other channels
        have different point counts within one packet and need resampling.
        :param temBuf: Data of each channel within one packet
        :return:
        """
        # No resampling needed at 1000 Hz
        if self.sample_rate == 1000:
            return temBuf
        oldTriggerChannel = temBuf[-1]
        # New trigger channel
        newTriggerChannel = [0] * len(temBuf[0])
        # Resampling ratio
        rate = 1000 / self.sample_rate
        for i in range(len(oldTriggerChannel)):
            if oldTriggerChannel[i] > 0:
                newTriggerChannel[int(i / rate)] = oldTriggerChannel[i]
        temBuf[-1] = newTriggerChannel
        return temBuf

    def isDataPacketLost(self, dataStruct):
        """
        Verify whether data packets were lost by checking timestamps.
        :param dataStruct: Parsed data packet
        :return:
        """
        if self.state == ConnectState.RUNNING and self.firstTimestamp == -1:
            self.firstTimestamp = dataStruct["startTimeStamp"]
        # Verify packet loss via timestamps
        if self.lastTimestamp > 0 and self.lastTimestamp != dataStruct["startTimeStamp"]:
            raise RuntimeError(
                "Maybe a packet loss happened. Expected startTimestamp "
                f"is {self.lastTimestamp} but received "
                f"{dataStruct['startTimeStamp']}")
        self.lastTimestamp = dataStruct["startTimeStamp"] + dataStruct["timeStampLength"]

    def combineDataAndTrigger(self):
        """
        Actual packet assembly process.
        :return:
        """
        # Buffer for all channels; the last channel is trigger data
        temp_buffer = []
        for i in range(self.n_chan):
            temp_buffer.append([])
        # Place data packets into the preceding channels
        for i in range(self.max_single_packet):
            data = self.single_module_data_buffer[i]['modules'][self.serialNumber]['data']
            for ch in range(self.n_chan - 1):
                temp_buffer[ch].extend(data[ch])
        # Initialize the last channel of temp_buffer to 0
        temp_buffer[-1] = [0] * len(temp_buffer[0])
        # Calculate data points per packet: forwarding duration * sample_rate / 1000
        dataCount = int(self.single_module_data_buffer[0]['timeStampLength'] * self.sample_rate / 1000)
        # Get timestamps for this batch of data
        totalTimestamp = []
        for i in range(self.max_single_packet):
            totalTimestamp.append(self.single_module_data_buffer[i]['startTimeStamp'])
        # Iterate in reverse to safely remove trigger packets that are assembled into data
        for i in range(len(self.single_module_trigger_buffer) - 1, -1, -1):
            startTimestamp = self.single_module_trigger_buffer[i]['startTimeStamp']
            # Find the position of this trigger's timestamp among all data timestamps
            index = self.FindTriggerTimeStampIndex(totalTimestamp, startTimestamp, dataCount)
            # Trigger timestamp exceeds the max in this batch, continue
            if index == -1:
                continue
            # Trigger timestamp is less than the min in this batch, discard it
            elif index == -2:
                # Remove this trigger
                self.single_module_trigger_buffer.pop(i)
            else:
                # Place the trigger value at the corresponding position
                temp_buffer[-1][index] = self.single_module_trigger_buffer[i]['modules'][0]['data'][0][0]
                # Remove this trigger
                self.single_module_trigger_buffer.pop(i)
        temp_buffer = np.array(temp_buffer)
        # Send the assembled packet into the buffer
        if self.state == ConnectState.RUNNING:
            self.buffer.appendBuffer(temp_buffer)
            # Also send to save_buffer for correctness verification
            self.save_buffer.appendBuffer(temp_buffer)
        # Clear cached data packets
        self.single_module_data_buffer = []

    def FindTriggerTimeStampIndex(self, totalTimestamp, triggerTimeStamp, dataCount):
        """
        Find the position of a trigger's timestamp within this batch of data.
        :param totalTimestamp: All timestamps of this batch of data packets
        :param triggerTimeStamp: Timestamp of the current trigger
        :param dataCount: Data points per packet
        :return:
        """
        # Trigger timestamp exceeds the max in totalTimestamp, keep it
        if triggerTimeStamp > totalTimestamp[-1]:
            return -1
        # Trigger timestamp is less than the min in totalTimestamp, discard it
        if triggerTimeStamp < totalTimestamp[0]:
            return -2
        # Otherwise, find the nearest data packet timestamp position
        # i.e., which data packet it belongs to
        properTimeStampIndex = -1
        for timeStamp in totalTimestamp:
            if timeStamp <= triggerTimeStamp:
                properTimeStampIndex += 1
            else:
                # Found the first timestamp greater than the trigger's
                break
        # If the found timestamp exactly matches the trigger's timestamp,
        # place it at the first position of the corresponding data packet
        if totalTimestamp[properTimeStampIndex] == triggerTimeStamp:
            index = properTimeStampIndex * dataCount
        else:
            # If the found timestamp is less than the trigger's, handle by sample rate
            # For sample rate >= 1000, find the precise timestamp position
            if self.sample_rate >= 1000:
                # Calculate precise timestamp position within this packet
                subTimeStampIndex = (triggerTimeStamp - totalTimestamp[properTimeStampIndex]) * int(self.sample_rate / 1000)
                # Final position = packet start position + offset within packet
                index = properTimeStampIndex * dataCount + subTimeStampIndex
            else:
                # For sample rate < 1000, place at the first position of the corresponding packet
                index = properTimeStampIndex * dataCount
        return index

    def GetDataLenCount(self):
        """
        Get the count of newly updated data points.
        :return:
        """
        return self.buffer.nUpdate

    def ResetDataLenCount(self):
        """
        Reset the count of updated data points.
        :return:
        """
        self.buffer.nUpdate = 0

    def ResetTriggerChanofBuff(self):
        """
        Set all values in the trigger channel to 0.
        """
        self.buffer.buffer[-1, :] = np.zeros((1, self.buffer.buffer.shape[-1]))

    def GetBufferData(self):
        """
        Get data from the buffer.
        :return:
        """
        return self.buffer.getData()

    def getSaveDataBuffer(self):
        temBuf = self.save_buffer.getData()
        return temBuf

    def process_trig(self):
        trig = self.getSaveDataBuffer()[-1]
        assert trig.ndim == 1
        rst = []
        ids = np.where(trig > 0)[0]  # indices of triggers in
        # sample points
        for i in ids:
            trg = str(int(trig[i]))
            rst.append((i / self.sample_rate, 0, trg))  # (onset duration,desc)
        return rst

    def save(self, fpath: str):
        signal_headers = []
        for ich in range(self.n_chan - 1):
            signal_headers.append(
                highlevel.make_signal_header(
                    self.channelNames[ich],
                    dimension='uV',
                    sample_rate=self.srates[ich],
                    physical_min=self.minPhysical[ich],
                    physical_max=self.maxPhysical[ich],
                    digital_min=self.minDigital[ich],
                    digital_max=self.maxDigital[ich]))
        header = highlevel.make_header(patientname='s', gender='Unknown')
        header['annotations'] = self.process_trig()
        # [file type]
        # EDF = 0, EDF+ = 1, BDF = 2, BDF+ = 3, automatic from extension = -1
        highlevel.write_edf(fpath, self.getSaveDataBuffer()[:-1], signal_headers, header, file_type=3)

    def save_timeStamp(self):
        """
        Save timestamps to file.
        :return:
        """
        with open('./test_time_stamp.pickle', 'wb') as f:
            pickle.dump(self.timeStamp, f)


#################################################################################
# This main method is written to test the data saving functionality.
# It receives data forwarded by JellyFish and saves it as a BDF file (test.bdf).
#    Trigger and data are stored in the same file.
#    BDF file storage has 1-second time resolution, so the saveable trigger count
#    equals the number of seconds of stored data.
#    Even if more triggers are received, only as many as the data duration can be saved.
# 1) t_buffer is the buffer length in seconds, also the max saveable data duration.
# 2) The sample rate when constructing DataServerThread must match the JellyFish EEG device sample rate.
# 3) To compare the test.bdf file saved here with the file recorded by JellyFish,
#    use test_data_validation.py for post-processing. See that file for details.
#################################################################################
if __name__ == "__main__":
    # sample_rate must match the JellyFish device sampling rate
    w4_data = DataServerThread(sample_rate=1000, t_buffer=60)

    # Connect to JellyFish data forwarding
    notconnect = w4_data.connect()
    if notconnect:
        raise TypeError("Can't connect JellyFish, Please open the hostport ")
    # Wait until meta packet is resolved
    while not w4_data.isReady():
        continue

    # Start
    w4_data.start()
    print('start')
    # Save data for N seconds
    # The seconds in time.sleep should not exceed the t_buffer value
    time.sleep(30)
    w4_data.stop()
    print('packet_count', w4_data.packet_count)
    # print(w4_data.timeStamp['trigger'])
    w4_data.save('./test.bdf')
    print('end')
