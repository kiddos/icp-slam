from __future__ import print_function
import serial
import time
import numpy as np
import cv2
import math
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def to_binary(char):
    binary = list()
    value = ord(char)
    while value > 0:
        binary = [value % 2] + binary
        value /= 2
    return binary


def to_decimal(binary):
    value = 0
    for i in range(len(binary)):
        value += binary[i] * pow(2, len(binary) - 1 - i)
    return value


def decode(char):
    new_char = chr(ord(char) - ord('0'))
    return to_binary(new_char)


def decode_to_value(string):
    binary = list()
    for i in range(len(string)):
        s = string[i]
        new_char = chr(ord(s) - ord('0'))
        b = to_binary(new_char)
        if len(b) < 6:
            for k in range(6 - len(b)):
                b = [0] + b
        binary += b
    return to_decimal(binary)


# open
def bm_command(lidar):
    message = 'BM;%s\x0A' % ('open_lidar')
    lidar.write(message)

    read_bytes = 5 + len(message)
    data = lidar.read(read_bytes).replace('\x0A', ' | ')
    print('bm command respond: ', end='')
    print(data)
    #  for c in data[3 + len(message):]:
    #      print(ord(c))
    status = data.split(' | ')[1]
    return status[:2]


# close
def qt_command(lidar):
    message = 'QT;%s\x0A' % ('close_lidar')
    lidar.write(message)

    read_bytes = 5 + len(message)
    data = lidar.read(read_bytes).replace('\x0A', ' | ')
    print('qt command respond: ', end='')
    print(data[:len(message)])


def time_stamp_value(time_stamp):
    binary = list()
    for i in range(len(time_stamp)):
        b = decode(time_stamp[i])
        if len(b) < 6:
            for k in range(6 - len(b)):
                b = [0] + b
        binary += b
    return to_decimal(binary)


def mdms_command(lidar, ctype='S', start='0044', end='0725', cluster='01',
        interval='0', scan_count='01', message='lidar_data', output=True):
    input_message = 'M%s%s%s%s%s%s;%s\x0A' % (ctype, start, end,
        cluster, interval, scan_count, message)
    lidar.write(input_message)

    read_bytes = len(input_message)
    response = lidar.read(read_bytes + 5).replace('\x0A', ' | ')
    if output:
        print('mdms command 1st response: %s' % (response))

    data = list()
    status = response.split(' | ')[1][:2]
    if status not in ['00', '99']:
        return status, data


    # get data
    unit_size = 2 if ctype == 'S' else 3
    data_size = (unit_size) * (int(end) - int(start) + 1)
    data_size += 2 * (data_size / 64 + 1) + 1
    data_size = read_bytes + 10 + data_size
    print('computed datasize: %s' % data_size)
    response = lidar.read(data_size).split('\x0A')
    #  # get time stamp
    time_stamp = time_stamp_value(response[2][:4])
    print('time stamp: %s' % (time_stamp))

    total_length = 0
    for i in '\x0A'.join(response):
        total_length += 1
    print('total length: %s' % total_length)

    all_data = list()
    for r in response[3:]:
        all_data += r[:-1]
    print(len(all_data))
    for i in range(0, len(all_data), 2):
        data.append(decode_to_value(all_data[i:i+2]))

    #  return status[:2], data
    return status, data


def gdgs_command(lidar, ctype='S', start='0044', end='0726',
        cluster='03', message='lidar_data'):
    input_message = 'G%s%s%s%s%s' % (ctype, start, end,
        cluster, message) + chr(10)
    print('input message lenth: %d' % len(input_message))
    lidar.write(input_message)

    read_bytes = len(input_message)
    response = lidar.read(read_bytes).replace(chr(10), 'LF')
    data = list()
    print('gdgs command: %s' % (response))

    status = lidar.read(5)
    if status[:2] not in ['00', '99']:
        return status[:2], data

    return status[:2], data


def main():
    lidar = serial.Serial(port='/dev/ttyACM0', baudrate=750000,
        parity=serial.PARITY_ODD, stopbits=serial.STOPBITS_TWO,
        bytesize=serial.SEVENBITS, timeout=3)

    #  open lidar
    status = bm_command(lidar)
    print('open status: %s' % status)

    center = (300, 300)
    start_theta = - math.pi * 2 / 3
    images = list()

    fig = plt.figure()
    for i in range(1000):
        image = np.zeros(shape=[600, 600, 3])
        begin = datetime.datetime.now()
        response, data = mdms_command(lidar)
        end = datetime.datetime.now()
        passed = end - begin
        print('timepassed: %s' % (passed.seconds + 1e-6 * passed.microseconds))
        #  print(response)
        if response not in ['00', '99']:
            break

        print('data size: %s' % (len(data)))
        delta = math.pi * 4 / 3 / len(data)
        for i, theta in enumerate(
                [start_theta + delta * i for i in range(len(data))]):
            val = data[i]
            p = (int(center[0] + val * math.cos(theta)),
                 int(center[1] + val * math.sin(theta)))
            cv2.line(image, center, p, (255, 0, 255), 2)

        cv2.imshow('Image', image)
        images.append((plt.imshow(image[:, :, 0],),))
        key = cv2.waitKey(10)
        if key in [10, 27]:
            break

    plt.colorbar()
    anim = animation.ArtistAnimation(fig, images, interval=200, repeat_delay=3000,
        blit=True)
    anim.repeat = False
    plt.show()
    anim.save('measurement noise.mp4')

    # close lidar
    qt_command(lidar)
    lidar.close()


if __name__ == '__main__':
    main()
