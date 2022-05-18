import numpy as np
import cv2
import random
import scipy.stats as ss
import pandas as pd
from gym import Env, spaces


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class Geosteering(Env):
    def __init__(self,render_,eval,fault=4):
        super(Geosteering, self).__init__()

        self.pixel_x = 1280
        self.pixel_y = 720
        self.min_y = 1110
        self.max_y = 1150
        self.observation_shape = (self.pixel_y,self.pixel_x,3)
        self.observation_space = spaces.Box(low = np.array([-3,-3,0,0.25,1,-10,-10,-5.0,0.5,0], dtype=np.float32),
                                            high = np.array([3,3,1,0.25,5,10,-10,5.0,1.0,1], dtype=np.float32),
                                            shape = (10,))

        self.action_space = spaces.Discrete(6,)

        self.y_min = 0
        self.x_min = 0
        self.y_max = int(self.observation_shape[0])
        self.x_max = self.observation_shape[1]

        self.elements = []

        #=============================================================================================#

        self.n_data = 30 # no. of decision points
        self.decdist = 30 # distance per decision [m]
        self.value = 2 # value of reservoir at each decision point [10^6 $]
        self.h = 3 # thickness [m]
        self.bd_step = 2 # no. of options: 2*self.bd_step + 2 e.g., [-2,-1,0,1,2,sidetrack]
        self.bd_int = 0.25 # change in bitdepth per step [m] e.g., -2 step = -0.5 change in m
        self.landing_bd = 1131.2
        self.rho = 0.5 
        self.UB_prior_SD = 0.5
        self.look_ahead = 1
        self.max_st = 10
        
        self.fault_mean = None
        self.fault_SD = None
        self.fault_i = None
        self.fault_f = None
        self.fault_pos = None
        self.fault_dist = None
        self.fault_num = None
        self.f_idx = None
        self.prob = None
        self.possible_UB = None
        self.res_cont = None
        self.res_value = None
        self.total_cost = None
        self.bd = None
        self.st = None
        self.exit = None
        
        self.render_ = render_
        self.eval = eval
        self.given_faults = fault
        self.run = 0
        self.train_run = 0
        
        self.UB_sensor = np.zeros(self.n_data)
        self.LB_sensor = np.zeros(self.n_data)
        self.UB_prior = np.zeros(self.n_data)
        self.UB_post = np.zeros(self.n_data)
        self.LB_post = np.zeros(self.n_data)
        self.z_sensor = np.zeros(self.n_data)
        self.UB_post_SD = np.ones(self.n_data)*self.UB_prior_SD
        self.BD_matrix = np.empty([2*self.bd_step*(self.n_data-1) + 1, self.n_data])
        self.BD_matrix[:] = np.NaN
        self.BD_matrix[0,0] = self.landing_bd
        self.bdd = np.arange(1,self.bd_step+1)
        self.bitdepth = np.empty([self.n_data,self.max_st])
        self.bitdepth[:] = np.NaN 

        self.rig_rent = 0.5 # 10^6 $/day
        self.ROP = 20 #m/hr
        self.op_cost = self.rig_rent/24 #$/hr
        self.cost_per_m = self.op_cost/self.ROP #$/m
        self.drill_cost = 2*self.cost_per_m*self.decdist
        self.ST_time = 123/24 #day
        self.sidetrack = self.rig_rent * self.ST_time

    def draw_elements_on_canvas(self,action):
        self.canvas = np.ones(self.observation_shape, dtype=np.uint8) * 1

        num_y = int((self.max_y-self.min_y)/5)
        for idx in range(num_y+1):
            text = str(self.min_y+idx*5)
            location = (0,int(self.pixel_y/num_y*idx))
            self.canvas = cv2.putText(self.canvas, text, location, font, 0.6, (255,255,255), 1, cv2.LINE_AA)

        for idx in range(0,10):
            text = str(100*idx)
            location = (int(self.pixel_x/9*idx),self.pixel_y-20)
            self.canvas = cv2.putText(self.canvas, text, location, font, 0.6, (255,255,255), 1, cv2.LINE_AA)

        
        text2 = r'Cost: ${:.2f}'.format(self.total_cost)
        text3 = r'EMV: ${:.2f}'.format(self.res_value-self.total_cost)

        self.canvas = cv2.putText(self.canvas, text2, (self.pixel_x-150,20), font, 0.8,
                    (255,255,255), 1, cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, text3, (self.pixel_x-150,40), font, 0.8,
                    (255,255,255), 1, cv2.LINE_AA)

        if not self.eval:
            text1 = r'Episode no. = {}'.format(self.train_run)
            self.canvas = cv2.putText(self.canvas, text1, (int(self.pixel_x/2)-50,50), font, 0.8,
                    (255,255,255), 1, cv2.LINE_AA)

        color = (16,100,255)
        if self.bd == 0:
            self.canvas = cv2.circle(self.canvas, (self.DrillBit.x, self.DrillBit.y), radius=6, color=(0,0,100), thickness=2)
        else:
            st = 0
            bd = 0
            while bd < self.bd:
                x1 = self.real_to_gym_x((bd)*self.decdist)
                x2 = self.real_to_gym_x((bd+1)*self.decdist)
                if ~np.isnan(self.bitdepth[bd+1,st]):
                    pt1 = (x1,self.real_to_gym_y(self.bitdepth[bd,st]))
                    pt2 = (x2,self.real_to_gym_y(self.bitdepth[bd+1,st]))
                    self.canvas = cv2.line(self.canvas, pt1, pt2, color, thickness=5)
                    bd += 1
                else:
                    if st in [0,5]:
                        color = (0,255,0)
                    elif st in [1,6]:
                        color = (255,0,0)
                    elif st in [2,7]:
                        color = (0,255,255)
                    elif st in [3,8]:
                        color = (255,0,255)
                    elif st in [4,9]:
                        color = (100,160,255)
                    x1 = self.real_to_gym_x((bd-1)*self.decdist)
                    x2 = self.real_to_gym_x((bd)*self.decdist)
                    pt1 = (x1,self.real_to_gym_y(self.bitdepth[bd-1,st]))
                    pt2 = (x2,self.real_to_gym_y(self.bitdepth[bd,st+1]))
                    self.canvas = cv2.line(self.canvas, pt1, pt2, color, thickness=5)
                    st += 1
            self.canvas = cv2.circle(self.canvas, pt2, radius=6, color=color, thickness=2)
        
        for bd in range(self.n_data-1):
            x1 = self.real_to_gym_x((bd)*self.decdist)
            x2 = self.real_to_gym_x((bd+1)*self.decdist)
            pt1 = (x1,self.real_to_gym_y(self.UB_sensor[bd]))
            pt2 = (x2,self.real_to_gym_y(self.UB_sensor[bd+1]))
            self.canvas = cv2.line(self.canvas, pt1, pt2, (255,255,255))
            pt1 = (x1,self.real_to_gym_y(self.UB_sensor[bd]+self.h))
            pt2 = (x2,self.real_to_gym_y(self.UB_sensor[bd+1]+self.h))
            self.canvas = cv2.line(self.canvas, pt1, pt2, (255,255,255))

        if self.bd < self.n_data-1:
            x1 = self.real_to_gym_x((self.bd)*self.decdist)
            x2 = self.real_to_gym_x((self.bd+1)*self.decdist)
            pt1 = (x1,self.real_to_gym_y(self.UB_post[self.bd]))
            pt2 = (x2,self.real_to_gym_y(self.possible_UB))
            self.canvas = cv2.line(self.canvas, pt1, pt2, (0,0,255), thickness=4)
            pt1 = (x1,self.real_to_gym_y(self.UB_post[self.bd]+self.h))
            pt2 = (x2,self.real_to_gym_y(self.possible_UB+self.h))
            self.canvas = cv2.line(self.canvas, pt1, pt2, (0,0,255), thickness=4)

        for bd in range(self.bd+1,self.n_data-1):
            x1 = self.real_to_gym_x((bd)*self.decdist)
            x2 = self.real_to_gym_x((bd+1)*self.decdist)
            pt1 = (x1,self.real_to_gym_y(self.UB_post[bd]))
            pt2 = (x2,self.real_to_gym_y(self.UB_post[bd+1]))
            self.canvas = cv2.line(self.canvas, pt1, pt2, (255,255,255), thickness=4)
            pt1 = (x1,self.real_to_gym_y(self.UB_post[bd]+self.h))
            pt2 = (x2,self.real_to_gym_y(self.UB_post[bd+1]+self.h))
            self.canvas = cv2.line(self.canvas, pt1, pt2, (255,255,255), thickness=4)
 
    def render(self, mode='human'):
        assert mode in ['human','rgb_array'], 'Invalid mode, must be either \'human\' or \'rgb_array\''

        if mode == 'human':
            cv2.imshow('Game', self.canvas)
            cv2.waitKey(50)

        elif mode == 'rgb_array':
            return self.canvas

    def gym_to_real_y(self, y):
        return ((y)*(self.max_y-self.min_y)/self.pixel_y+self.min_y)

    def real_to_gym_y(self, y):
        return int((y-self.min_y)/(self.max_y-self.min_y)*self.pixel_y)

    def gym_to_real_x(self, x):
        return ((x)*(self.decdist*self.n_data)/self.pixel_x)

    def real_to_gym_x(self, x):
        return int(x/(self.decdist*self.n_data)*self.pixel_x)

    def close(self):
        cv2.destroyAllWindows()

    #====================================================================================================#

    def reset(self):
        self.UB_prior = np.ones(self.n_data) * 1130
        self.UB_sensor[0] = self.UB_prior[0]
        self.UB_post_SD = np.ones(self.n_data)*self.UB_prior_SD
        
        if self.eval == False:
            #Generating fault information from continuous distribution
            self.fault_i = np.array([6,11,16,21])+random.choice((-1,0,1))
            self.fault_f = np.array([9,14,19,24])+random.choice((-1,0,1))
            self.fault_num = random.choice((3,4,5))
            self.fault_mean = np.empty(self.fault_num)
            self.fault_SD = np.empty(self.fault_num)
            for idx in range(self.fault_num):
                self.fault_mean[idx] = random.choice((-4,-3.5,-3,-2.5,2.5,3,3.5,4))
                self.fault_SD[idx] = np.random.uniform(0.5,1.0)
            
            if self.fault_num == 3:
                idx = random.choice((0,1,2,3))
                self.fault_i = np.delete(self.fault_i, idx)
                self.fault_f = np.delete(self.fault_f, idx)
            elif self.fault_num == 5:
                self.fault_i = np.append(self.fault_i, 26)
                self.fault_f = np.append(self.fault_f, 28)

            self.fault_pos = np.round(ss.uniform.rvs(loc=self.fault_i, scale=self.fault_f-self.fault_i, size=self.fault_num))
            self.fault_dist = ss.norm.rvs(loc=self.fault_mean, scale=self.fault_SD, size=self.fault_num)

            self.train_run += 1
        else:
            np.random.seed(10)
            if self.run == 0:
                self.fault_mean_data = pd.read_excel('test cases/Fault_%s.xlsx'%self.given_faults, sheet_name='fault_mean').to_numpy()
                self.fault_SD_data = pd.read_excel('test cases/Fault_%s.xlsx'%self.given_faults, sheet_name='fault_SD').to_numpy()
                self.fault_i_data = pd.read_excel('test cases/Fault_%s.xlsx'%self.given_faults, sheet_name='fault_i').to_numpy().astype(int)
                self.fault_f_data = pd.read_excel('test cases/Fault_%s.xlsx'%self.given_faults, sheet_name='fault_f').to_numpy().astype(int)
                self.fault_pos_data = pd.read_excel('test cases/Fault_%s.xlsx'%self.given_faults, sheet_name='fault_pos').to_numpy().astype(int)
                self.fault_dist_data = pd.read_excel('test cases/Fault_%s.xlsx'%self.given_faults, sheet_name='fault_dist').to_numpy()
                
            self.fault_num = len(self.fault_mean_data[self.run,~np.isnan(self.fault_mean_data[self.run,:])])

            self.fault_mean = self.fault_mean_data[self.run,:self.fault_num]
            self.fault_SD = self.fault_SD_data[self.run,:self.fault_num]
            self.fault_i = self.fault_i_data[self.run,:self.fault_num]    
            self.fault_f = self.fault_f_data[self.run,:self.fault_num]
            self.fault_pos = self.fault_pos_data[self.run,:self.fault_num]
            self.fault_dist = self.fault_dist_data[self.run,:self.fault_num]
        
            self.run += 1
        
        #Generating prior boundaries, sensor reading, and bitdepth matrix (for sidetrack)
        self.UB_prior_func()
        self.UB_post_SD_func()
        self.BD_matrix_func()
        self.z_sensor_func()
        self.UB_sensor_func()
        self.UB_post = np.copy(self.UB_prior)
        
        self.bd = 0
        self.st = 0
        self.f_idx = 0
        self.total_cost = 0
        self.res_value = 0
        self.exit = 0
        self.prob = 0
        self.bitdepth = np.empty([self.n_data,self.max_st])
        self.bitdepth[:] = np.NaN
        self.bitdepth[self.bd,self.st] = self.landing_bd

        #=============================================================================================#
        if self.render_:
            self.ep_return = 0
            x = 0
            y = self.real_to_gym_y(self.bitdepth[self.bd,self.st])

            self.DrillBit = DrillBit('robot', self.x_max, self.x_min, self.y_max, self.y_min)
            self.DrillBit.set_position(x,y)

            self.elements = [self.DrillBit]

        return self.first_step()

    def UB_prior_func(self):
        for idx in range(self.fault_num):
            if idx != self.fault_num - 1:
                end = self.fault_f[idx+1]
            else:
                end = self.n_data
            self.UB_prior[self.fault_f[idx]:end] = self.UB_prior[self.fault_f[idx]-1] + self.fault_mean[idx]

    def UB_post_SD_func(self):
        for idx in range(self.fault_num):
            self.UB_post_SD[self.fault_i[idx]:self.fault_f[idx]] = \
            np.sqrt(self.UB_post_SD[self.fault_i[idx]:self.fault_f[idx]]**2 + self.fault_SD[idx]**2)
    
    def z_sensor_func(self):
        for idx in range(1,self.n_data):
            self.z_sensor[idx] = self.rho*self.z_sensor[idx-1] + np.sqrt(1-self.rho**2)*ss.norm.rvs()
    
    def UB_sensor_func(self):
        UB_store = np.copy(self.UB_prior)
        for idx in range(1,self.n_data):
            if idx in self.fault_pos:
                x = np.where(self.fault_pos==idx)[0][0]
                self.UB_sensor[idx] = self.UB_sensor[idx-1] + self.fault_dist[x]
                if x != self.fault_num - 1:
                    end = self.fault_f[x+1]
                else:
                    end = self.n_data
                UB_store[idx:end] = self.UB_sensor[idx]
            else:
                self.UB_sensor[idx] = self.UB_prior_SD*self.z_sensor[idx] + UB_store[idx]
        self.LB_sensor = np.copy(self.UB_sensor) + self.h
        
    def UB_post_func(self):
        if self.bd in self.fault_pos:
            x = np.where(self.fault_pos == self.bd)[0][0]
            if x != self.fault_num - 1:
                end = self.fault_f[x+1]
            else:
                end = self.n_data
            self.UB_post[self.bd:end] = self.UB_sensor[self.bd]
            self.UB_post_SD[self.bd:end] = self.UB_prior_SD
        
        self.UB_post[self.bd+1] += self.rho*(self.UB_post_SD[self.bd+1]/self.UB_post_SD[self.bd]) \
        * (self.UB_sensor[self.bd] - self.UB_post[self.bd])
        
        self.UB_post_SD[self.bd+1] = np.sqrt(self.UB_post_SD[self.bd+1]**2*(1-self.rho**2))
        self.UB_post[self.bd] = self.UB_sensor[self.bd]
        self.UB_post_SD[self.bd] = 0
        self.LB_post = np.copy(self.UB_post) + self.h

    def BD_matrix_func(self):
        for idx in range(1,self.n_data):
            self.BD_matrix[:2*self.bd_step*idx+1,idx] = np.concatenate((self.BD_matrix[0,idx-1]-self.bdd[::-1]*self.bd_int,
                                                                           self.BD_matrix[:2*self.bd_step*(idx-1)+1,idx-1],
                                                                           self.BD_matrix[2*self.bd_step*(idx-1),idx-1]\
                                                                            +self.bdd*self.bd_int))  

    def first_step(self):
        self.UB_post_func()
        state = []
        for idx in range(1,self.look_ahead+1):
            self.possible_UB = self.UB_post[self.bd+idx]
            state.append((self.bitdepth[self.bd,self.st]-self.possible_UB)/self.h)
            state.append((self.possible_UB+self.h-self.bitdepth[self.bd,self.st])/self.h)
        state.append(self.exit)
        state.append(self.bd_int)
        state.append(self.fault_num)
        state.append(self.fault_i[self.f_idx]-self.bd)
        state.append(self.fault_f[self.f_idx]-self.bd)
        state.append(self.fault_mean[self.f_idx])
        state.append(self.fault_SD[self.f_idx])
        state.append(self.prob)

        if self.render_:
            self.draw_elements_on_canvas(self.bd_step)

        return np.array(state, dtype=np.float32) 

    def step(self, action):
        reward = 0
        if self.exit == 1 and action == self.bd_step*2+1 and self.max_st-self.st>1:
            state, reward, done, _  = self.st_step()
            return state, reward, done, _

        else:
            self.bitdepth[self.bd+1,self.st] = self.bitdepth[self.bd,self.st] + (action - self.bd_step) * self.bd_int
            if self.render_:
                self.DrillBit.x += self.real_to_gym_x(self.decdist)
                self.DrillBit.y = self.real_to_gym_y(self.bitdepth[self.bd+1,self.st])
            self.bd += 1

            norm_DTUB = (self.bitdepth[self.bd,self.st] - self.UB_sensor[self.bd])/self.h
            norm_DTLB = (self.LB_sensor[self.bd] - self.bitdepth[self.bd,self.st])/self.h
            alpha = min(norm_DTUB, norm_DTLB)

            if 0 <= alpha <= 1.0:
                reward += self.value - self.drill_cost
                self.res_value += self.value
            else:
                reward -= self.drill_cost
            self.total_cost += self.drill_cost
            
            done = False
            if self.bd == self.n_data-1:
                if self.render_:
                    self.draw_elements_on_canvas(action)
                done = True
                state = []
                x = []
                for bd in range(1,self.n_data):
                    if self.UB_sensor[bd] < self.bitdepth[bd,self.st] < self.LB_sensor[bd]:
                        x.append(1)
                self.res_cont = sum(x)/(self.n_data-1)
                for idx in range(8):
                    state.append(0)
                for idx in range(1,self.look_ahead+1):
                    state.append(0)
                    state.append(0)
                return np.array(state, dtype=np.float32), reward, done, {}

            if self.UB_sensor[self.bd] < self.bitdepth[self.bd,self.st] < self.LB_sensor[self.bd]:
                self.exit = 0
            else: self.exit = 1

            self.UB_post_func()
            
            state = []
            for idx in range(1,self.look_ahead+1):
                if self.bd+idx >= self.n_data:
                    state.append((self.bitdepth[self.bd,self.st]-self.UB_post[self.n_data-1])/self.h)
                    state.append((self.UB_post[self.n_data-1]+self.h-self.bitdepth[self.bd,self.st])/self.h)
                else:
                    if self.f_idx < self.fault_num and self.bd >= self.fault_i[self.f_idx] - 1 and self.bd < self.fault_pos[self.f_idx]:
                        self.possible_UB = self.UB_post[self.bd+idx] + ss.norm.rvs(loc=self.fault_mean[self.f_idx], scale=self.fault_SD[self.f_idx], size=1)[0] * self.prob
                        state.append((self.bitdepth[self.bd,self.st]-self.possible_UB)/self.h)
                        state.append((self.possible_UB+self.h-self.bitdepth[self.bd,self.st])/self.h)
                        if self.bd == self.fault_pos[self.f_idx]-1:
                            self.prob = 0
                            self.f_idx = self.f_idx+1
                        else:
                            self.prob = 1/(self.fault_f[self.f_idx] - self.bd)
                    else:
                        self.possible_UB = self.UB_post[self.bd+idx]
                        state.append((self.bitdepth[self.bd,self.st]-self.possible_UB)/self.h)
                        state.append((self.possible_UB+self.h-self.bitdepth[self.bd,self.st])/self.h)
            state.append(self.exit)
            state.append(self.bd_int)
            if self.f_idx == self.fault_num:
                for idx in range(6):
                    state.append(0)
            else:
                state.append(self.fault_num)
                state.append(self.fault_i[self.f_idx]-self.bd)
                state.append(self.fault_f[self.f_idx]-self.bd)
                state.append(self.fault_mean[self.f_idx])
                state.append(self.fault_SD[self.f_idx])
                state.append(self.prob)
            if self.render_:
                self.draw_elements_on_canvas(action)
            return np.array(state, dtype=np.float32), reward, done, {} 

    def st_step(self):
        reward = 0
        
        bdlength = len(self.BD_matrix[~np.isnan(self.BD_matrix[:,self.bd]),self.bd])
        alpha = np.minimum((self.BD_matrix[:bdlength,self.bd]-self.UB_sensor[self.bd])/self.h,
                           (self.UB_sensor[self.bd]+self.h-self.BD_matrix[:bdlength,self.bd])/self.h)
        midres = np.where(alpha==max(alpha))[0]
        self.bitdepth[:self.bd,self.st+1] = self.bitdepth[:self.bd,self.st]
        self.bitdepth[self.bd,self.st+1] = self.BD_matrix[random.choice(midres),self.bd]
        if self.render_:
            self.DrillBit.y = self.real_to_gym_y(self.bitdepth[self.bd,self.st+1])
            self.draw_elements_on_canvas(self.bd_step)
        
        self.st += 1

        norm_DTUB = (self.bitdepth[self.bd,self.st] - self.UB_sensor[self.bd])/self.h
        norm_DTLB = (self.UB_sensor[self.bd] + self.h - self.bitdepth[self.bd,self.st])/self.h
        alpha = min(norm_DTUB, norm_DTLB)

        if 0 < alpha < 1.0:
            reward += self.value - self.sidetrack
            self.res_value += self.value
        else:
            reward -= self.sidetrack
        self.total_cost+=self.sidetrack
            
        if self.UB_sensor[self.bd] < self.bitdepth[self.bd,self.st] < self.LB_sensor[self.bd]:
            self.exit = 0
        else: self.exit = 1
            
        done = False
        state = []
        
        for idx in range(1,self.look_ahead+1):
            if self.bd+idx >= self.n_data:
                state.append((self.bitdepth[self.bd,self.st]-self.UB_post[self.n_data-1])/self.h)
                state.append((self.UB_post[self.n_data-1]+self.h-self.bitdepth[self.bd,self.st])/self.h)
            else:
                if self.f_idx < self.fault_num and self.bd >= self.fault_i[self.f_idx] - 1 and self.bd < self.fault_pos[self.f_idx]:
                    self.possible_UB = self.UB_post[self.bd+idx] + ss.norm.rvs(loc=self.fault_mean[self.f_idx], scale=self.fault_SD[self.f_idx], size=1)[0]*self.prob
                    state.append((self.bitdepth[self.bd,self.st]-self.possible_UB)/self.h)
                    state.append((self.possible_UB+self.h-self.bitdepth[self.bd,self.st])/self.h)
                    if self.bd == self.fault_pos[self.f_idx]-1:
                        self.prob = 0
                        self.f_idx = self.f_idx+1
                    else:
                        self.prob = 1/(self.fault_f[self.f_idx] - self.bd)
                else:
                    self.possible_UB = self.UB_post[self.bd+idx]
                    state.append((self.bitdepth[self.bd,self.st]-self.possible_UB)/self.h)
                    state.append((self.possible_UB+self.h-self.bitdepth[self.bd,self.st])/self.h)
        state.append(self.exit)
        state.append(self.bd_int)
        if self.f_idx == self.fault_num:
            for idx in range(6):
                state.append(0)
        else:
            state.append(self.fault_num)
            state.append(self.fault_i[self.f_idx]-self.bd)
            state.append(self.fault_f[self.f_idx]-self.bd)
            state.append(self.fault_mean[self.f_idx])
            state.append(self.fault_SD[self.f_idx])
            state.append(self.prob)
        return np.array(state, dtype=np.float32), reward, done, {} 

class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.name = name

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.set_position(self.x, self.y)

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def clamp(self, n, minn, maxn):
        return max(min(maxn,n), minn)


class DrillBit(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(DrillBit, self).__init__(name, x_max, x_min, y_max, y_min)


