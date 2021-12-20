import math
import pyttsx3

def getGradient(p1,p2):
    numerator = p2[1] - p1[1]
    denominator = p2[0] - p1[0]
    gradient = numerator / denominator
    return gradient

def calculateAngle(coordinates):
    # p1 = [(320,480)]
    # p2 = [(0,480)]
    # p3 = coordinates[-1:]
    # p1 = center
    # p2 = leftmost corner
    p1 = (320,480)
    p2 = (0,480)
    p3,p4 = coordinates[-2:]
    p5 = ((p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2)
    # print("P5: ", p5)
    m1 = getGradient(p1,p2)
    m2 = getGradient(p1,p3)
    m3 = getGradient(p1,p4)
    m4 = getGradient(p1,p5)

    angleInRadiansLeftEdge = math.atan((m2-m1)/(1+(m2*m1)))
    angleInRadiansRightEdge = math.atan((m3-m1)/(1+(m3*m1)))
    angleInRadiansMidPoint = math.atan((m4-m1)/(1+(m4*m1)))

    angleInDegreesLeftEdge = round(math.degrees(angleInRadiansLeftEdge))
    angleInDegreesRightEdge = round(math.degrees(angleInRadiansRightEdge))
    angleInDegreesMidPoint = round(math.degrees(angleInRadiansMidPoint))
    
    if angleInDegreesLeftEdge < 0:
        angleInDegreesLeftEdge = 180 + angleInDegreesLeftEdge
    if angleInDegreesRightEdge < 0:
        angleInDegreesRightEdge = 180 + angleInDegreesRightEdge
    if angleInDegreesMidPoint < 0:
        angleInDegreesMidPoint = 180 + angleInDegreesMidPoint

    # print(type(angleInDegreesLeftEdge) , " " , type(angleInDegreesRightEdge))
    # return float(angleInDegreesLeftEdge),float(angleInDegreesRightEdge), float(angleInDegreesMidPoint)
    return float(angleInDegreesMidPoint)

# [[248, 153], [370, 155]]
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', 165)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def tts_object_location(boundingBox, distance):
    c1 = list(boundingBox[:2])
    c2 = list(boundingBox[2:4])
    coordinates = [c1,c2]
    Object = boundingBox[-1]
    angle = calculateAngle(coordinates) # coordinates[leftx,lefty,rightx,righty]
    engine.say(f'{Object}, {distance} centimeters away, at {angle} degrees')
    engine.runAndWait()

#param = (248, 153, 370, 155, "Chair")
#tts_object_location(param)

# engine.say("I will speak this text")
