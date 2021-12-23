import math
import pyttsx3
import numpy as np

def getGradient(p1,p2):
    numerator = p2[1] - p1[1]
    denominator = p2[0] - p1[0]
    gradient = numerator / denominator
    return gradient

def calculateAngle(coordinates):
    # p1 = center
    # p2 = leftmost corner
    p1 = (320,480)
    p2 = (0,480)
    p3,p4 = coordinates[-2:] # Coordinates of bounding box
    p5 = ((p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2) # Mid point of bounding box
    
    # Calculating gradients of lines
    m1 = getGradient(p1,p2)
    m2 = getGradient(p1,p3)
    m3 = getGradient(p1,p4)
    m4 = getGradient(p1,p5)

    # Calculating Angles between lines
    angleInRadiansLeftEdge = math.atan((m2-m1)/(1+(m2*m1)))
    angleInRadiansRightEdge = math.atan((m3-m1)/(1+(m3*m1)))
    angleInRadiansMidPoint = math.atan((m4-m1)/(1+(m4*m1)))

    # Converting angles to degrees
    angleInDegreesLeftEdge = round(math.degrees(angleInRadiansLeftEdge))
    angleInDegreesRightEdge = round(math.degrees(angleInRadiansRightEdge))
    angleInDegreesMidPoint = round(math.degrees(angleInRadiansMidPoint))
    
    if angleInDegreesLeftEdge < 0:
        angleInDegreesLeftEdge = 180 + angleInDegreesLeftEdge
    if angleInDegreesRightEdge < 0:
        angleInDegreesRightEdge = 180 + angleInDegreesRightEdge
    if angleInDegreesMidPoint < 0:
        angleInDegreesMidPoint = 180 + angleInDegreesMidPoint

    return float(angleInDegreesMidPoint)

# Setting voice engine properties
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', 140)
voices = engine.getProperty('voices')
engine.setProperty('voice', "english")

# Main TTS function speaking object name, location and distance
def tts_object_location(boundingBox, distance):
    distance = np.round_((distance / 100))
    c1 = list(boundingBox[:2])
    c2 = list(boundingBox[2:4])
    coordinates = [c1,c2]
    Object = boundingBox[-1]
    angle = calculateAngle(coordinates) # coordinates[leftx,lefty,rightx,righty]
    engine.say(f'{Object}, approximately {distance} meters away, at {angle} degrees')
    engine.runAndWait()
