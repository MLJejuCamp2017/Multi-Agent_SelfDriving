#%%
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw
import numpy as np
import random
from random import shuffle

# Pygame init
width = 1500
height = 300

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Merge Scenario")
clock = pygame.time.Clock()
# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows thins downs.
show_sensors = False
draw_screen = True

pi_unit = np.pi / 180


class GameState():

    def __init__(self, numCars = 12, draw_screen=True, test=False):
        self.crashed = False
        self.test = test
        self.w = width
        self.h = height
        # physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        self.draw_screen = draw_screen
        self.numCar = numCars

        # Lane Stuff
        self.y_m = int(self.h * (1. / 30))
        self.lane_width = int((self.h - self.y_m * 2) * (1. / 2))
        self.m_stPoint = int(self.w * (1. / 4))
        self.m_laneLength = int(self.w - (self.m_stPoint * 2))
        self.l_lane1Start = (0, 0)
        self.l_lane1Stop = (0, self.lane_width)
        self.l_lane2Start = (0, self.lane_width + self.y_m * 2)
        self.l_lane2Stop = (0, self.h)
        self.m_llaneStart = (self.m_stPoint, self.y_m)
        self.m_llaneJoint = (self.m_stPoint, self.y_m + self.lane_width)
        self.m_llaneStop = (self.m_stPoint, self.y_m + self.lane_width * 2)
        mlaneStop = self.m_stPoint + self.m_laneLength
        self.m_rlaneStart = (mlaneStop, self.y_m)
        self.m_rlaneJoint = (mlaneStop, self.y_m + self.lane_width)
        self.m_rlaneStop = (mlaneStop, self.y_m + self.lane_width * 2)
        self.e_lane1Start = (self.w, 0)
        self.e_lane1Stop = (self.w, self.lane_width)
        self.e_lane2Start = (self.w, self.lane_width + 2 * self.y_m)
        self.e_lane2Stop = (self.w, self.h)

        # Multi Agent Cars
        carLane1X = self.l_lane1Stop[0] + self.y_m
        carLane1Y = self.l_lane1Stop[1]
        # carLaneWidth = self.m_llaneStart[0] - self.y_m
        # carLaneHeigth = self.l_lane1Stop[1] - self.y_m
        carLaneWidth = self.m_rlaneStart[0] - self.y_m
        carLaneHeigth = self.l_lane1Stop[1] - self.y_m
        carLane2X = self.l_lane2Stop[0] + self.y_m
        carLane2Y = self.l_lane2Stop[1] - self.y_m
        lane1 = [carLane1X, carLane1Y, carLaneWidth, carLaneHeigth]
        lane2 = [carLane2X, carLane2Y, carLaneWidth, carLaneHeigth]

        self.rewardRed, self.rewardGreen = self._set_rewardPoints()
        self.endRed, self.endGreen = self._set_lastRewardPoints()
        self.multiAgent = MultiAgentCar(lane1, lane2, self.space, self.w, self.h, 
            goal=[self.rewardRed, self.rewardGreen], goal2=[self.endRed, self.endGreen], numCar=self.numCar)

        # Record steps.
        self.num_steps = 0
        statics = self._getLanes()
        self._setBackPoints()

        self.space.add(statics)
        # self.create_obstacle()
        # self._setBackground()

    def _getLanes(self):
        # about lane
        static = [
            pymunk.Segment(self.space.static_body,
                           self.l_lane1Start, self.l_lane1Stop, 1),
            pymunk.Segment(self.space.static_body,
                           self.l_lane1Stop, self.m_llaneJoint, 1),
            pymunk.Segment(self.space.static_body,
                           self.m_llaneJoint, self.l_lane2Start, 1),
            pymunk.Segment(self.space.static_body,
                           self.l_lane2Start, self.l_lane2Stop, 1),
            pymunk.Segment(self.space.static_body,
                           self.l_lane2Stop, self.m_llaneStop, 1),
            pymunk.Segment(self.space.static_body,
                           self.m_llaneStop, self.m_rlaneStop, 1),
            pymunk.Segment(self.space.static_body,
                           self.m_rlaneStop, self.e_lane2Stop, 1),
            pymunk.Segment(self.space.static_body,
                           self.e_lane2Stop, self.e_lane2Start, 1),
            pymunk.Segment(self.space.static_body,
                           self.e_lane2Start, self.m_rlaneJoint, 1),
            pymunk.Segment(self.space.static_body,
                           self.m_rlaneJoint, self.e_lane1Stop, 1),
            pymunk.Segment(self.space.static_body,
                           self.e_lane1Stop, self.e_lane1Start, 1),
            pymunk.Segment(self.space.static_body,
                           self.e_lane1Start, self.m_rlaneStart, 1),
            pymunk.Segment(self.space.static_body,
                           self.m_rlaneStart, self.m_llaneStart, 1),
            pymunk.Segment(self.space.static_body,
                           self.m_llaneStart, self.l_lane1Start, 1)
        ]

        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['black']
        return static

    def _setBackPoints(self):
        self.point_list_left = [self.l_lane1Stop,
                                self.m_llaneJoint, self.l_lane2Start]
        self.point_list_right = [self.e_lane1Stop,
                                 self.m_rlaneJoint, self.e_lane2Start]
        self.point_list_up = [
            self.l_lane2Stop, self.m_llaneStop, self.m_rlaneStop, self.e_lane2Stop]
        self.point_list_down = [
            self.l_lane1Start, self.e_lane1Start, self.m_rlaneStart, self.m_llaneStart]

    def _draw_blackBackground(self):
        color = (0, 0, 0)
        pygame.draw.polygon(screen, color, self.point_list_left)
        pygame.draw.polygon(screen, color, self.point_list_right)
        pygame.draw.polygon(screen, color, self.point_list_up)
        pygame.draw.polygon(screen, color, self.point_list_down)

    def frame_step(self, action):
        # if action == 0:  # Turn left.
        #     self.car_body.angle -= .2
        # elif action == 1:  # Turn right.
        #     self.car_body.angle += .2

        # driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        # self.car_body.velocity = 1 * driving_direction
        self.multiAgent.moveMultiCars(action)

        screen.fill(THECOLORS["white"])
        self._draw_blackBackground()
        # self._drawRewarod_point()
        draw(screen, self.space)
        self.space.step(1. / 10)
        if self.draw_screen:
            pygame.display.flip()
        clock.tick()

        readings = self.multiAgent.getMultiState()
        states = np.array(readings)

        rewards, dones = self.multiAgent.checkRewards()
        # Include spped of car in states?
        # print(states)
        # print(rewards)
        # print(dones)
        # val = raw_input("Stop here : ")
        # print(states.shape) # (12,10)
        # val = raw_input("Stop here : ")
        self.multiAgent.getBack(test=self.test)

        return states, np.array(rewards), dones

    def _set_rewardPoints(self):
        x, y = self.m_rlaneJoint
        smallWidth = self.lane_width-4*self.y_m
        bigWidth = self.w - x
        bigHeight = self.e_lane2Start[1] - y
        # reward x position
        reward_x1 = x+smallWidth

        # upper lane y reward position
        l_ystep = self.lane_width/4
        y_step = int((smallWidth*bigHeight)/bigWidth)
        reward_y1 = y+y_step+l_ystep
        reward_y2 = y+y_step+3*l_ystep

        # lower lane y reward psotion
        reward_y3 = y-(y_step+l_ystep)
        reward_y4 = y-(y_step+3*l_ystep)
        reward_for_red = [(reward_x1, reward_y1), (reward_x1, reward_y2)]
        reward_for_green = [(reward_x1, reward_y3), (reward_x1, reward_y4)]
        return reward_for_red, reward_for_green

    def _set_lastRewardPoints(self):
        l_ystep = self.lane_width/4
        x = self.w+500
        red_y1 = self.e_lane2Start[1]+l_ystep
        red_y2 = self.e_lane2Start[1]+3*l_ystep
        green_y1 = self.e_lane1Start[1]+l_ystep
        green_y2 = self.e_lane1Start[1]+3*l_ystep
        e_re_red = [(x, red_y1), (x, red_y2)]
        e_re_green = [(x, green_y1), (x, green_y2)]
        return e_re_red, e_re_green

    def _drawRewarod_point(self):
        red1, red2 = self.rewardRed
        green1, green2 = self.rewardGreen
        e_r1, e_r2 = self.endRed
        e_g1, e_g2 = self.endGreen
        draw_reward_point = True
        if draw_reward_point:
            pygame.draw.circle(screen, (255, 0, 0), red1, 10)
            pygame.draw.circle(screen, (255, 0, 0), red2, 10)
            pygame.draw.circle(screen, (0, 255, 0), green1, 10)
            pygame.draw.circle(screen, (0, 255, 0), green2, 10)
            pygame.draw.circle(screen, (255, 0, 0), e_r1, 10)
            pygame.draw.circle(screen, (255, 0, 0), e_r2, 10)
            pygame.draw.circle(screen, (0, 255, 0), e_g1, 10)
            pygame.draw.circle(screen, (0, 255, 0), e_g2, 10)

##############################################################################
##############################################################################
##############################################################################


class MultiAgentCar():
    def __init__(self, lane1, lane2, space, width, height, goal, goal2, numCar=12):
        self.maxCar = 48
        self.car_radius = 25
        self.width = width
        self.height = height
        self.numCar = numCar
        if self.maxCar < self.numCar:
            print("You can only make {} cars".format(self.max_cars))
            raise SystemExit(0)
        self.space = space
        self.goal= goal
        self.goal2 = goal2
        self.lanePos1 = self.startPosition(lane1)
        self.lanePos2 = self.startPosition(lane2)
        self.Cars = self.makeCars()
        self.newY = [self.lanePos1[0][1], self.lanePos1[6][1], self.lanePos2[0][1], self.lanePos2[6][1]] 
        self.col_Counter = 0
        self.backPoint = self.width-self.car_radius*6 # x-axis

    def makeCars(self):
        index1= list(range(len(self.lanePos1)))
        index2 = list(range(len(self.lanePos2)))
        shuffle(index1)
        shuffle(index2)
        color_pro = np.random.sample(self.numCar) < 0.5
        lane_pro = np.random.sample(self.numCar) < 0.5

        car = []
        for i in xrange(0, self.numCar):
            color = 1
            if color_pro[i]:
                color = 0
            if lane_pro[i]: ## make use pop method.
                idx = index1.pop()
                lane = self.lanePos1[idx]
            else:
                idx = index2.pop()
                lane = self.lanePos2[idx]
            car.append(Car(0, lane[0], lane[1], self.space, self.width, self.height, goal = self.goal, 
                goal2 = self.goal2, car_radius=self.car_radius, agentId=i, color=color))
        return car

    def startPosition(self, laneInfo): 
        cols = int(self.maxCar / 2)
        rows = 2
        startX, startY = laneInfo[0] + self.car_radius, laneInfo[1] - self.car_radius*1.5
        endX, endY = startX + laneInfo[2]+ self.car_radius*2, startY - laneInfo[3]
        x_step = (endX-startX) /cols
        y_step = (startY - endY) /rows
        lanePos = []
        for i in range(rows):
            for j in range(cols):
                lanePos.append([startX + j * x_step, startY - i * y_step])
        return lanePos

    def moveMultiCars(self, action):
        for i in xrange(self.numCar):
            self.Cars[i].moveCar(action[i])

    def getMultiState(self):
        states = []
        for i in xrange(self.numCar):
            states.append(self.Cars[i].get_position_sonar_readings())
        return states

    def checkRewards(self):
        rewards = []
        dones = []
        for i in xrange(self.numCar):
            done = self.Cars[i].check_getback(self.backPoint)
            dones.append(done)
            reward = self.Cars[i].get_reward(done)
            rewards.append(reward)
            
        return rewards, dones


    def getBack(self, test=False):
        checkingList = []
        if test:
            backPoint = self.backPoint-300
        else:
            backPoint = self.backPoint
        for i in xrange(self.numCar):
            if self.Cars[i].check_getback(backPoint):
                checkingList.append(i)
        nCar = len(checkingList)
        if nCar != 0:
            choise = self.newY
            shuffle(choise)
            color_pro = np.random.sample(nCar) < 0.5
            for i, index in enumerate(checkingList):
                if color_pro[i]: color = 0 
                else : color = 1
                pos_y = choise[i%len(choise)]
                self.Cars[index].resetCar(color, 0, pos_y)





class Car():
    def __init__(self, r, x, y, space, width, height, goal, goal2, car_radius=25, agentId=0, color=0):
        self.color = color # 0:green, 1:red
        self.agentId = agentId
        self.car_radius = car_radius
        self.space = space
        self.width = width
        self.height = height
        self.vmax = 200
        self.redGoal = goal[0]
        self.greenGoal = goal[1]
        self.eRedGoal = goal2[0]
        self.eGreenGoal = goal2[1]

        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, 14, (0, 0))
        self.car_body = pymunk.Body(mass, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, car_radius)
        if self.color == 0:
            self.car_shape.color = THECOLORS["green"]
        else:
            self.car_shape.color = THECOLORS["red"]
        self.space.add(self.car_body, self.car_shape)
        

    def make_sonar_arm(self):
        spread = 2  # Default spread.
        distance = 25  # Gap before first sensor.
        arm_points = []
        x, y = self.car_body.position
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 50):
            arm_points.append((distance + x + (spread * i), y))
        return arm_points

    def get_position_sonar_readings(self):
        color = float(self.color)
        x, y = self.car_body.position
        angle = self.car_body.angle
        vel = self.direct/self.vmax
        car_info = [color, x, y, vel, angle]
        arm = self.make_sonar_arm()
        readings = []
        # Rotate them and get readings.
        for i in range(-90, 91, 30):
            readings.append(self.get_arm_distance(arm, pi_unit*i))
        if show_sensors:
            pygame.display.update()
        self.readings = np.array(readings)
        return car_info+readings

    def get_arm_distance(self, arm, offset):
        # Used to count the distance.
        x, y = self.car_body.position
        angle = self.car_body.angle
        i = 0
        # Look at each point and see if we've hit something
        for point in arm:
            i += 1
            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= self.width or rotated_p[1] >= self.height:
                return i
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i
            if show_sensors:
                pygame.draw.circle(screen, (0, 0, 255), (rotated_p), 1)
        return i

    def get_track_or_not(self, reading):
        if reading == THECOLORS['white']:
            return 0
        else:
            return 1
    ################################################################
    ### I have to focus on this part
    ################################################################
    def get_reward(self, done):
        base_reward = 200
        my_color = self.color
        my_angle = self.car_body.angle
        my_x, my_y = self.car_body.position
        goalx = self.redGoal[0][0]
        ySep = int((self.redGoal[0][1]+self.greenGoal[0][1])/2.)
        # ySep = int((self.redGoal[0][1]+self.greenGoal[0][1])/2.)
        if done:
            if my_color == 0:
                if my_y > ySep:
                    reward = base_reward*6
                else:
                    reward = base_reward*-6
            else:
                if my_y < ySep:
                    reward = base_reward*6
                else:
                    reward = base_reward*-6
            return reward
        else:
            if self.car_is_crashed():
                reward = -3*base_reward
                return reward

            if my_x >= goalx-self.car_radius:
                if my_color == 0:
                    angle1 = self._get_angle(self.eRedGoal[0])
                    angle2 = self._get_angle(self.eRedGoal[1])
                    reward = 2*base_reward*(self.direct/(self.vmax))*max(math.cos(my_angle-angle1), math.cos(my_angle-angle2))
                else:
                    angle1 = self._get_angle(self.eGreenGoal[0])
                    angle2 = self._get_angle(self.eGreenGoal[1])
                    reward = 2*base_reward*(self.direct/(self.vmax))*max(math.cos(my_angle-angle1), math.cos(my_angle-angle2))
                return reward
            
            if my_color == 0:
                angle1 = self._get_angle(self.redGoal[0])
                angle2 = self._get_angle(self.redGoal[1])
                reward = (1./2)*base_reward*(self.direct/(self.vmax))*max(math.cos(my_angle-angle1), math.cos(my_angle-angle2))
            else:
                angle1 = self._get_angle(self.greenGoal[0])
                angle2 = self._get_angle(self.greenGoal[1])
                reward = (1./2)*base_reward*(self.direct/(self.vmax))*max(math.cos(my_angle-angle1), math.cos(my_angle-angle2))
            return reward
    
    # def _euclidean_dist(self, target):
    #     x_ , y_ = self.car_body.position
    #     x = np.array([x_, y_])
    #     t = np.array(target)
    #     dist = np.sqrt(np.sum((t-x)**2))
    #     return dist

    def _get_angle(self, target):
        x, y = self.car_body.position
        t_x, t_y = target
        rad = math.atan2(t_y-y, t_x-x)
        return rad

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        x = x_2-x_1
        y = y_2-y_1
        x_change = x*math.cos(radians) - y*math.sin(radians)
        y_change = x*math.sin(radians) + y*math.cos(radians)
        new_x = x_change + x_1
        new_y = self.height - (y_change + y_1)
        return int(new_x), int(new_y)

    def moveCar(self, action):
        steering = float(action[0])
        accel = float(action[1])
        brake = float(action[2])
        # self.car_body.angle = self.testFindGoal()
        # The below one is original
        self.car_body.angle = steering*45.*pi_unit # steering [-1, 1] -> [-45, 45]
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.direct = (self.vmax*(accel-brake+1))*1./2
        self.car_body.velocity = self.direct*driving_direction
        # print(self.direct, type(self.direct)) # float
        # print(driving_direction, type(driving_direction)) # Vec2d 
        # print(self.car_body.velocity, type(self.car_body.velocity)) # Vec2D
        # val = raw_input("Fucking Stop:")

    def car_is_crashed(self): # check out the collision function in pymunk
        if (self.readings == 1).any():
            return True
        else:
            return False
        # if self.readins == 1:

    def resetCar(self, color, x, y):
        self.color = color
        self.car_body.position = x, y
        self.car_body.angle = 0.
        if self.color == 0:
            self.car_shape.color = THECOLORS["green"]
        else:
            self.car_shape.color = THECOLORS["red"]

    def check_dest(self):
        """
            I think I do not this function.
        """
        my_color = self.color
        my_angle = self.car_body.angle
        my_velocity = self.car_body.velocity

        if my_color == 0:

            angle1 = self._get_angle(self.redGoal[0])
            angle2 = self._get_angle(self.redGoal[1])
            reward = max(my_angle*math.cos(angle1), my_angle*math.cos(angle2))
        else:
            angle1 = self._get_angle(self.greenGoal[0])
            angle2 = self._get_angle(self.greenGoal[1])
            reward = max(my_angle*math.cos(angle1), my_angle*math.cos(angle2))
        return reward

    def check_getback(self, x_point):
        '''
            Check car have to be back.
            x_point : back point
        '''
        x, y = self.car_body.position
        if x >= x_point: return True # x-axis: return True
        else: return False

    # def testFindGoal(self):
    #     if self.color == 0:
    #         dist1 = self._euclidean_dist(self.redGoal[0])
    #         dist2 = self._euclidean_dist(self.redGoal[1])
    #         if dist1 < dist2:
    #             angle = self._get_angle(self.redGoal[0])
    #         else:
    #             angle = self._get_angle(self.redGoal[1])
    #         # angle = self._get_angle(self.redGoal[0])
    #     else:
    #         # angle = self._get_angle(self.greenGoal[0])
    #         dist1 = self._euclidean_dist(self.greenGoal[0])
    #         dist2 = self._euclidean_dist(self.greenGoal[1])
    #         if dist1 < dist2:
    #             angle = self._get_angle(self.greenGoal[0])
    #         else:
    #             angle = self._get_angle(self.greenGoal[1])
    #     return angle


if __name__ == "__main__":
#    Settings
    steering = 0.
    accel = 0.5
    brake = 0.
    acDim = 3
    defaultAction = np.array([[steering, accel, brake]])
    numCars = 1
    for i in xrange(numCars):
        if i == 0:
            actions = np.zeros((1, acDim))
            actions = defaultAction
        else:
            actions = np.vstack((actions, defaultAction))
    game_state = GameState(numCars=numCars)

    while True:
        game_state.frame_step(actions)
    pygame.quit()









