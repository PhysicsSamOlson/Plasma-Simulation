
# coding: utf-8

# In[2248]:


import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from scipy.stats import linregress
import random as random


# In[2249]:


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


# In[2250]:


getTable(ions)


# In[2251]:


Acceleration(ions, 3, True)
#this prints the information for acceleration for the ion that is in the 3rd smallest position


# In[2252]:


#setting accelerations based on how many to the left and right
for x in range(len(ions)):
    ions = sortByPosition(ions)
    Acceleration(ions, x, False)
getTable(ions)
# i think the math for calculating acceleration is wrong. This is what I currently need help with


# In[2253]:


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


# In[2254]:


def evolve(ions, time, element):
    #new position = s + v*t + (a*t^2/2)
    s = ions[element].position + ions[element].velocity*time + (ions[element].acceleration*time**2)/2
    #changed by vwb
    v = ions[element].velocity + ions[element].acceleration*time
    #Comment by vwb
    #there is something messed up about this.  If an ion hit's the wall, it should trigger a 'wall' least time event, 
    #and it should never really go past 0 or 1, it should just reverse velocity
    if s <= 0:
        s *= -1
        v *= -1
    elif s >= 1:
        #modified by vwb
        s = 2-s
        v *= -1
    ions[element].position = s
    #new velocity = v + a*t
    ions[element].velocity = v
    #Acceleration(ions, element, False)


# In[2255]:


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


# In[2256]:


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


# In[2257]:


times = []
energies = []
P_pos = []
P_neg = []
def getHistogram(ions, amount, figure):
    vels = []
    ion_positions = []
    for y in range(amount):
        leastTime = calculateCollisionTime(ions)
        if y is not 0:
            times.append(leastTime[0]+ times[y-1])
        else:
            times.append(leastTime[0])
        energy = 0
        for x in range(len(ions)):
            evolve(ions, leastTime[0], x)
            #Accerleration(ions, x, False)
            ion_positions.append(ions[x].position)
            vels.append(ions[x].velocity)
            energy += (ions[x].velocity**2) * 0.5
            if x is 0:
                continue;
            elif ions[x].charge is 1 and ions[x-1].charge is 1:
                P_pos.append((ions[x].position - ions[x-1].position))
            elif ions[x].charge is -1 and ions[x-1].charge is -1:
                P_pos.append((ions[x].position - ions[x-1].position))
            else:
                P_neg.append((ions[x].position - ions[x-1].position))
        energies.append(energy)
        #calculateAccelerations(ions, False)
        for x in range(len(ions)):
            Acceleration(ions, x, False)
    if figure is 'positions':
        plt.hist(ion_positions, bins=20)
    else:
        plt.hist(vels, bins=20)
    plt.title('Figure 2 (evolved '+str(amount)+' times)')
    plt.show()
    #print(*vels, sep = '\n')
getHistogram(ions, 1000, 'vels')


# In[2258]:


plt.plot(times, energies)
plt.xlabel('t')
plt.ylabel('Energy')
plt.title('Figure 1')
plt.show()


# In[2221]:


P_pos = []
P_neg = []
probability_neg = []
probability_pos = []
for x in range(len(ions)):
    for y in range(len(ions)):
        if x is y:
            continue;
        if ions[x].charge is ions[y].charge:
            P_pos.append(np.abs(ions[x].position - ions[y].position))
        elif ions[x].charge is not ions[y].charge:
            P_neg.append(np.abs(ions[x].position - ions[y].position))
P_neg.sort()
P_pos.sort()
for y in range(len(P_pos)):
    probability_pos.append(P_pos[y]/(P_pos[y]+P_neg[y]))
    probability_neg.append(P_neg[y]/(P_pos[y]+P_neg[y]))
slicer = slice(0, 4900)
slicer2 = slice(50, 4900)
P_neg = P_neg[slicer]
P_neg = P_neg[slicer2]
P_pos = P_pos[slicer2]
probability_neg = probability_neg[slicer2]
probability_pos = probability_pos[slicer2]


# In[2222]:


plt.plot(P_pos, probability_pos, label='P-', color='#34b521')
plt.plot(P_neg, probability_neg, label='P+', color='#0377fc')
plt.title('Figure 3')
plt.xlabel('Δx')
plt.ylabel('probability')
plt.legend(loc="upper right")
plt.show()


# In[2223]:


log_prob = np.log(probability_neg)
print(np.polyfit(P_neg, log_prob, 1))


# In[2283]:


print(*P_neg, sep = '\n')
#print(*probability_neg, sep = '\n')


# In[2144]:


getTable(ions)


# In[2145]:


positive_less = 0
negative_less = 0
negative_more = 0
positive_more = 0
for x in range(len(ions)):
    if x < 50 and ions[x].charge is 1:
        positive_less+=1
    elif x < 50 and ions[x].charge is -1:
        negative_less+=1
    elif x >= 50 and ions[x].charge is 1:
        positive_more+=1
    else:
        negative_more+=1
print(positive_less, negative_less, positive_more, negative_more)
Acceleration(ions, 49, True)
print("END CODING")
print("_______________________________________________________________________________________________________")


# # Post lab writeup
# What does figure 1 mean?
# - Figure 1 shows kinetic energy vs time of the system. When t is small, kinetic energy oscillates rapidly. This is because equilibrium is far from being reached; opposite charged particles are next to each other at the beginning and before movement they hold just potential energy. As the system evolves, the potential energy is converted to kinetic as particles are pushed away from each other. The velocity increases, giving a local maximum in kinetic energy, and then as the distance increases from the original opposing particle, the distance to others is smaller. The charge from these other particles slows down incoming, high velocity particles, and as the opposing charge slows down the particle, kinetic energy drops again and potential energy is high.
# 
# What does figure 2 mean?
# - Figure 2 tells us that the distribution of velocity follows a Maxwell distribution. This is important because it shows that our code was correct in respect to calculating velocities due to applied force. It proves that over a range from -1 to 1, most particles have a velocity towards the center, and very few have the max velocities. An equal amount of particles have opposite charge, forcing particles to bounce back and forth between oppositely charged particles. If you take the average of these oscillations, they will be near zero, which is why the graph shows a maximum number of particles around v = 0.
# 
# What does figure 3 mean?
# - Figure 3 represents the probability that particles of either opposite or same charge will be a certain distance from each other. This is important to the study because it is a different way of telling us that opposite charges repel each other. If you look at the particles that are directly next to each other, 60% of them are opposite charges and 40% are the same. The graph also shows that the probability reaches 50% at a distance of around .15. This could either mean that opposing charges are no longer forceful at a distance of .15, or that in our system, when opposing particles are that far apart, other particles present are close enough to apply a force that cancels out the original repelling force.
# 
# ## Plasma Simulation Lab by Sam Olson and Jess Bosch
# 
# *Note: The following code is for 1000 ions and then 10 ions*

# In[2146]:


temp_ions = np.linspace(1, 1000, 1000)
new_position = []
for x in range(len(temp_ions)):
    new_position.append(random.random())
new_ions = []
for x in range(len(temp_ions)):
    new_ions.append(ion(temp_ions[x], x, new_position[x]))
for x in range(len(new_ions)):
    new_ions = sortByPosition(new_ions)
    Acceleration(new_ions, x, False)
getTable(new_ions)


# In[2147]:


new_times = []
new_energies = []
new_P_pos = []
new_P_neg = []
new_probability_neg = []
new_probability_pos = []
def getHistogram(new_ions, amount, figure):
    vels = []
    ion_positions = []
    for y in range(amount):
        leastTime = calculateCollisionTime(new_ions)
        if y is not 0:
            new_times.append(leastTime[0]+ new_times[y-1])
        else:
            new_times.append(leastTime[0])
        energy = 0
        for x in range(len(new_ions)):
            evolve(new_ions, leastTime[0], x)
            #Accerleration(ions, x, False)
            ion_positions.append(new_ions[x].position)
            vels.append(new_ions[x].velocity)
            energy += (new_ions[x].velocity**2) * 0.5
            if x is 0:
                continue;
            elif new_ions[x].charge is 1 and new_ions[x-1].charge is 1:
                new_P_pos.append((new_ions[x].position - new_ions[x-1].position))
            elif new_ions[x].charge is -1 and new_ions[x-1].charge is -1:
                new_P_pos.append((new_ions[x].position - new_ions[x-1].position))
            else:
                new_P_neg.append((new_ions[x].position - new_ions[x-1].position))
        new_energies.append(energy)
        #calculateAccelerations(ions, False)
        for x in range(len(new_ions)):
            Acceleration(new_ions, x, False)
   # if figure is 'positions':
      #  plt.hist(ion_positions, bins=20)
    #else:
        #plt.hist(vels, bins=20)
    plt.title('Figure 2 (evolved '+str(amount)+' times)')
    #plt.show()
getHistogram(new_ions, 100, 'vels')


# In[2149]:


for x in range(len(new_ions)):
    for y in range(len(new_ions)):
        if x is y:
            continue;
        if new_ions[x].charge is new_ions[y].charge:
            new_P_pos.append(np.abs(new_ions[x].position - new_ions[y].position))
        elif new_ions[x].charge is not new_ions[y].charge:
            new_P_neg.append(np.abs(new_ions[x].position - new_ions[y].position))
new_P_neg.sort()
new_P_pos.sort()
for y in range(len(new_P_neg)):
    new_probability_pos.append(new_P_pos[y]/(new_P_pos[y]+new_P_neg[y]))
    new_probability_neg.append(new_P_neg[y]/(new_P_pos[y]+new_P_neg[y]))
slicer2 = slice(50, 150000)
new_P_neg = new_P_neg[slicer2]
new_P_pos = new_P_pos[slicer2]
new_probability_neg = new_probability_neg[slicer2]
new_probability_pos = new_probability_pos[slicer2]
plt.plot(new_P_pos, new_probability_pos, label='P+', color='#34b521')
plt.plot(new_P_neg, new_probability_neg, label='P-', color='#0377fc')
plt.title('Figure 3')
plt.xlabel('Δx')
plt.ylabel('probability')
plt.legend(loc="upper right")
plt.show()


# In[2227]:


#print(*new_P_neg, sep = '\n')
print(*new_probability_neg, sep = '\n')
#print(new_P_neg, new_probability_neg)


# In[2155]:


temp_ions = np.linspace(1, 10, 10)
new_position = []
for x in range(len(temp_ions)):
    new_position.append(random.random())
new_ions = []
for x in range(len(temp_ions)):
    new_ions.append(ion(temp_ions[x], x, new_position[x]))
for x in range(len(new_ions)):
    new_ions = sortByPosition(new_ions)
    Acceleration(new_ions, x, False)
getTable(new_ions)


# In[2156]:


new_times = []
new_energies = []
new_P_pos = []
new_P_neg = []
new_probability_neg = []
new_probability_pos = []
def getHistogram(new_ions, amount, figure):
    vels = []
    ion_positions = []
    for y in range(amount):
        leastTime = calculateCollisionTime(new_ions)
        if y is not 0:
            new_times.append(leastTime[0]+ new_times[y-1])
        else:
            new_times.append(leastTime[0])
        energy = 0
        for x in range(len(new_ions)):
            evolve(new_ions, leastTime[0], x)
            #Accerleration(ions, x, False)
            ion_positions.append(new_ions[x].position)
            vels.append(new_ions[x].velocity)
            energy += (new_ions[x].velocity**2) * 0.5
            if x is 0:
                continue;
            elif new_ions[x].charge is 1 and new_ions[x-1].charge is 1:
                new_P_pos.append((new_ions[x].position - new_ions[x-1].position))
            elif new_ions[x].charge is -1 and new_ions[x-1].charge is -1:
                new_P_pos.append((new_ions[x].position - new_ions[x-1].position))
            else:
                new_P_neg.append((new_ions[x].position - new_ions[x-1].position))
        new_energies.append(energy)
        #calculateAccelerations(ions, False)
        for x in range(len(new_ions)):
            Acceleration(new_ions, x, False)
    #if figure is 'positions':
     #   plt.hist(ion_positions, bins=20)
    #else:
        #plt.hist(vels, bins=20)
    plt.title('Figure 2 (evolved '+str(amount)+' times)')
    #plt.show()
getHistogram(new_ions, 1000, 'vels')


# In[2157]:


for x in range(len(new_ions)):
    for y in range(len(new_ions)):
        if x is y:
            continue;
        if new_ions[x].charge is new_ions[y].charge:
            new_P_pos.append(np.abs(new_ions[x].position - new_ions[y].position))
        elif new_ions[x].charge is not new_ions[y].charge:
            new_P_neg.append(np.abs(new_ions[x].position - new_ions[y].position))
new_P_neg.sort()
new_P_pos.sort()
for y in range(len(new_P_pos)):
    new_probability_pos.append(new_P_pos[y]/(new_P_pos[y]+new_P_neg[y]))
    new_probability_neg.append(new_P_neg[y]/(new_P_pos[y]+new_P_neg[y]))
slicer2 = slice(500, 3000)
new_P_neg = new_P_neg[slicer2]
new_P_pos = new_P_pos[slicer2]
new_probability_neg = new_probability_neg[slicer2]
new_probability_pos = new_probability_pos[slicer2]


# In[2167]:


plt.plot(new_P_pos, new_probability_pos, label='P+', color='#34b521')
plt.plot(new_P_neg, new_probability_neg, label='P-', color='#0377fc')
plt.title('Figure 3')
plt.xlabel('Δx')
plt.ylabel('probability')
plt.legend(loc="upper right")
plt.show()


# In[2171]:


log_new_prob = np.log(new_probability_pos)
vals = np.polyfit(new_P_neg, log_new_prob, 1)
old_y = []
for i in range(len(new_P_pos)):
    old_y.append(vals[0] * new_P_pos[i])
y = np.exp(vals[1]) * np.exp(old_y)
plt.plot(new_P_pos, y, label = 'P+')
plt.legend(loc="upper right")
plt.show()
print(np.polyfit(new_P_neg, log_new_prob, 1))


# In[2170]:


log_new_prob = []
vals = []
for i in range(len(new_probability_neg)):
    if new_probability_neg[i] is not 0:
        log_new_prob.append(np.log(new_probability_neg[i]))
vals = np.polyfit(new_P_neg, log_new_prob, 1)
old_y = []
for i in range(len(new_P_neg)):
    old_y.append(vals[0] * new_P_neg[i])
y = np.exp(vals[1]) * np.exp(old_y)
plt.plot(new_P_neg, y, label = 'P-')
plt.legend(loc="upper right")
plt.show()

