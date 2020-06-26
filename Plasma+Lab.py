
# coding: utf-8

# In[742]:


import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from scipy.stats import linregress
import random as random


# In[743]:


def ifZero(val):
    if val == 0:
        if random.random() < .5:
            return -1*random.random()
        else:
            return random.random()
    else:
        return val
def sortByPosition(ions):
    ions.sort(key=lambda x:x.position, reverse=False)
    return ions
def getTable(ions):
    #ions = sortByPosition(ions)
    print("Charge\tIon\t\tPosition\t\tVelocity\t\tAcceleration")
    print("_________________________________________________________________________________")
    for x in ions:
        if x.charge == -1 and x.velocity < 0:
            print(x.charge,"\t",x.ion,"\t",x.position,"\t",x.velocity,"\t\t",x.acceleration)
        elif x.charge == -1 and x.velocity >= 0:
            print(x.charge,"\t",x.ion,"\t",x.position,"\t","",x.velocity,"\t\t",x.acceleration)
        elif x.charge == 1 and x.velocity < 0:
            print("",x.charge,"\t",x.ion,"\t",x.position,"\t",x.velocity,"\t\t",x.acceleration)
        elif x.charge == 1 and x.velocity >= 0:
            print("",x.charge,"\t",x.ion,"\t",x.position,"\t","",x.velocity,"\t\t",x.acceleration)
        else:
            print("",x.charge,"\t",x.ion,"\t",x.position,"\t",x.velocity,"\t\t",x.acceleration)
class ion:
    def __init__(self, val, count, pos):
        self.ion =  int(val)
        if count % 2 == 0:
            self.charge = 1
        else:
            self.charge = -1
        self.position = pos
        self.velocity = random.random()*ifZero(random.randrange(-1,1))
        self.acceleration = None
        self.timeLeft = None
        self.timeRight = None
def Acceleration(ions, element, get):
    ions = sortByPosition(ions)
    if element > len(ions):
        return print("Element",element,"does not exist")
    positive_left= 0
    positive_right = 0
    for x in range(len(ions)):
        if x >= element:
            continue
        if ions[x].charge == 1:
            positive_left += 1
    for x in range(len(ions)):
        if x <= element:
            continue
        if ions[x].charge == 1:
            positive_right += 1
    negative_left = element - positive_left
    negative_right = len(ions)-1-element-positive_right
    if (ions[element].charge == 1):
        # acc = positive_right - positive_left - negative_left + negative_right
        negativec = -1*negative_left + negative_right
        positivec = -1*positive_right + positive_left
        acc = (positivec + negativec)
    else:
        negativec = negative_left - negative_right
        positivec = positive_right - positive_left
        acc = (positivec + negativec)
    if get == False:
        ions[element].acceleration = acc
        return ions
    elif get == True:
        return print("Ion:",ions[element].ion,"\nPositive ions on left:",positive_left,"\nNegative ions on left",negative_left,"\nPositive ions on right:",positive_right,"\nNegative ions on right",negative_right)
def calculateAccelerations(ions, sort = True):
    if sort:
        sortByPosition(ions)
    for x in range(len(ions)):
        Acceleration(ions, x, False)
temp_ions = np.linspace(1, 100, 100)
position = []
for x in range(len(temp_ions)):
    position.append(random.random())
ions = []
for x in range(len(temp_ions)):
    ions.append(ion(temp_ions[x], x, position[x]))


# In[744]:


getTable(ions)


# In[745]:


Acceleration(ions, 3, True)
#this prints the information for acceleration for the ion that is in the 3rd smallest position


# In[746]:


#setting accelerations based on how many to the left and right
for x in range(len(ions)):
    ions = sortByPosition(ions)
    Acceleration(ions, x, False)
getTable(ions)
# i think the math for calculating acceleration is wrong. This is what I currently need help with


# In[747]:


def solveQuadraticGreaterThanZero(a, b, c,): #quadratic formula
    if a == 0:
        if -c / b > 0:
            return -c / b
        return None
    d = b * b - 4 * a * c
    if d < 0:
        return None
    x1 = (-b + np.sqrt(d)) / (2 * a)
    x2 = (-b - np.sqrt(d)) / (2 * a)
    if x1 > 0:
        if x2 > 0:
            return min(x1, x2)
        else:
            return x1
    elif x2 > 0:
        return x2
    return None
def calculateCollisionTime(ions, debug = False):
    ions = sortByPosition(ions)

    # "collision events"
    events = []

    # first/last ion with wall:
    # wallPos = a/2 * t^2 + v * t + pos
    leftWallTime = solveQuadraticGreaterThanZero(ions[0].acceleration / 2, ions[0].velocity, ions[0].position)
    events.append([leftWallTime, 0, 'wall'])
    rightWallTime = solveQuadraticGreaterThanZero(ions[-1].acceleration / 2, ions[-1].velocity, ions[-1].position - 1)
    events.append([rightWallTime, -1, 'wall'])

    for i in range(1, len(ions)):
        # adjacent ion pairs:
        # a1/2 * t^2 + v1 * t + pos1 = a2/2 * t^2 + v2 * t + pos2
        pairTime = solveQuadraticGreaterThanZero(
            ions[i].acceleration / 2 - ions[i - 1].acceleration / 2,
            ions[i].velocity - ions[i - 1].velocity,
            ions[i].position - ions[i - 1].position
        )
        events.append([pairTime, i, i - 1])
    # exclude events that will not happen
    events = [e for e in events if e[0] is not None]
    #for x in events:
      # print(x, sep='') 
    if len(events) == 0:
        return [None, None, None]

    # find the first event
    return min(events, key = lambda e: e[0])
calculateCollisionTime(ions)


# In[748]:


def evolve(ions, time, element):
    #new position = s + v*t + (a*t^2/2)
    s = ions[element].position + ions[element].velocity*time + (ions[element].acceleration*time**2)/2
    #changed by vwb
    v = ions[element].velocity + ions[element].acceleration*time
    #Comment by vwb
    #there is something messed up about this.  If an ion hit's the wall, it should trigger a 'wall' least time event, 
    #and it should never really go past 0 or 1, it should just reverse velocity
    if s < 0:
        s *= -1
        v *= -1
    elif s >= 1:
        #modified by vwb
        s = 2 - s
        v *= -1
    ions[element].position = s
    #new velocity = v + a*t
    ions[element].velocity = v


# In[749]:


def evolveAmount(ions, amount):
    pos = []
    vels = []
    times = []
    for y in range(amount):
        leastTime = calculateCollisionTime(ions)
        for x in range(len(ions)):
            evolve(ions, leastTime[0], x)
            pos.append(ions[x].position)
            vels.append(ions[x].velocity)
    pos.sort()
    vels.sort()
    calculateAccelerations(ions, False)
    return [pos, vels]
#evo = evolveAmount(ions, 100)
#plt.plot(evo[1], evo[0])
#plt.show()
#print (min(evo[0]))


# In[750]:


def evolveAmountTable(ions, amount):
    initial_pos = []
    final_pos = []
    times = []
    for y in range(amount):
        message = ''
        leastTime = calculateCollisionTime(ions)
        times.append(leastTime)
        for x in range(len(ions)):
            evolve(ions, leastTime[0], x)
        message += "Table for evolution number "+str(y+1)
        if leastTime[2] is 'wall':
            message += '\nThis was a wall collision, therefore no ions switched positions'
        else:
            ions[leastTime[1]], ions[leastTime[2]] = ions[leastTime[2]], ions[leastTime[1]]
            message += "\nThis was evoled for "+str(leastTime[0])+" seconds\nIon "+str(ions[leastTime[1]].ion)+" collided with ion "+str(ions[leastTime[2]].ion)
        calculateAccelerations(ions, False)
        print(message)
        getTable(ions)
        print('\n\n')
#evolveAmountTable(ions, 20)


# In[751]:


def getHistogram(ions, amount, figure):
    vels = []
    times = []
    ion_positions = []
    leastTime = calculateCollisionTime(ions)
    for y in range(amount):
        for x in range(len(ions)):
            evolve(ions, leastTime[0], x)
            ion_positions.append(ions[x].position)
            vels.append(ions[x].velocity)
    calculateAccelerations(ions, False)
    if figure is 'positions':
        plt.hist(ion_positions, bins=20)
    else:
        plt.hist(vels, bins=20)
    plt.title('Figure 2 (evolved '+str(amount)+' times)')
    plt.show()
getHistogram(ions, 1000, 'vels')


# In[752]:


getTable(ions)

