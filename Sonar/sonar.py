import time
import serial
import threading

'''
    Simple Moving Average.
    Used to smooth the measurements as this helps avoid the noise
    in the environment to which Ultrasonic sensor is susceptible to.
    We keep it to two measurements so we can get readings faster 
    and avoid hitting obstacle.
'''
class SMA:
    def __init__(self, window):
        self.window = window
        self.values = []

    def append(self, value):
        if len(self.values) > self.window - 1: self.values.pop(0)
        self.values.append(value)

    def calculate(self):
        return sum(self.values)/self.window

    def __str__(self):
        return self.calculate()


def read_sensor_package(bytes_serial):
    """
    Read a sensor from serial bytes. Expected format is 5 bytes:
        1, 2 : The first two bytes indicate 'YY' to tell start of message.
        3 : This byte is an unsigned integer for sensor index.
        4, 5 : unsigned integers for reading distance.
    :return:
        sensor_index, reading
    """

    if bytes_serial[0] == 0x59 and bytes_serial[1] == 0x59:  # check for 'YY'
        # print(bytes_serial)
        sensor_index = bytes_serial[2]  # sensor index
        reading = bytes_serial[4] + bytes_serial[3] * 256  # 2 bytes for reading
        return sensor_index, reading
    else:
        return -1, None

'''
    Getting readings from sonars through this function which is assigned
    a thread and storing them corresponding to the sensor they were
    taken from.
'''
def read_serial(serial, sensors):
    while True:

        # Read by bytes
        counter = serial.in_waiting  # count the number of bytes of the serial port
        bytes_to_read = 5
        if counter > bytes_to_read - 1:
            bytes_serial = serial.read(bytes_to_read)
            # ser.reset_input_buffer()  # reset buffer

            sensor_index, sensor_reading = read_sensor_package(bytes_serial)

            if sensor_index >= 0:
                if sensor_index not in sensors:
                    sensors[sensor_index] = SMA(2)
                if sensor_reading > 0:
                    sensors[sensor_index].append(sensor_reading)


'''
    Main driver function for Ultrasonic sensor.
    We select the port at which Arduino is connected, assign thread to
    function which takes readings from sensors and stores them and
    calculate distance. Based on current distance to object, feedback
    is given by a beep whose frequency varies depending on distance i.e. 
    more frequent beeps if closer and less frequent if further away from
    user.
'''

if __name__ == "__main__":

    _serial = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    _serial.flushInput()

    sensors = {}

    serial_thread = threading.Thread(target=read_serial, args=(_serial, sensors), daemon=True)
    serial_thread.start()

    while True:

        for k, v in sensors.items():
            print(f"Sonar {k}: {v.calculate()} cm ", end="")
            dist = v.calculate()
            if dist > 50 and dist < 100:
                print('\a')
                time.sleep(0.7)
            elif dist < 50:
                print('\a')
                time.sleep(0.25)
            

        time.sleep(0.2)