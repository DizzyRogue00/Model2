import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pd

import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sb

import os
import math
import copy
from functools import reduce
from itertools import *
import random
import pickle
import logging
import logging.config
import yaml

np.random.seed(20)

sb.set()

with open('logging.yaml','r',encoding='utf-8') as f:
    #config=yaml.load(f,Loader=yaml.FullLoader)
    config = yaml.safe_load(f)
    #print(config)
    logging.config.dictConfig(config)
logger=logging.getLogger('mylogger')

def load_pickle(path):
    with open(path,'rb') as f:
        while 1:
            try:
                y=pickle.load(f)
                yield y
            except EOFError:
                logger.error('Error', exc_info=True)
                break

class FOT(object):
    def __init__(self, routeNo, distance, average_distance, speed, demand, peak_point_demand):
        self._gammar=25 #$/veh.hr
        self._beta=0.25 #$/veh.seat.hr

        self._v_w=10    #time value of waiting time $/pax.hr
        self._v_v=6     #time value of in-vehicle time $/pax.hr

        self._t_u=1.0/180   #hr/parcel
        self._alpha=1.2 #[0.6,0.8,1,1.2,1.5]

        self._c=16000   #$/veh
        self._e=2400    #$/veh
        self._recovery=0.1359

        self._m_j=np.random.randint(2,4,routeNo)

        self._eta=0.125
        self._v_p=1500#the weight of one parcel is 0.5 tons.

        self._demand=demand
        self._peak_point_demand=peak_point_demand

        self._distance=distance
        self._average_distance=average_distance

        self._speed=speed
        self._period = len(speed[0])
        self._routeNo=routeNo

        self._d_i, self._d_j = self.generate_freight_demand()
        print(self._m_j,self._d_j)

        self._path='data.pickle'

    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self,value):
        self._beta=value

    @property
    def gammar(self):
        return self._gammar
    @gammar.setter
    def gammar(self,value):
        self._gammar=value

    @property
    def v_w(self):
        return self._v_w
    @v_w.setter
    def v_w(self,value):
        self._v_w=value

    @property
    def v_v(self):
        return self._v_v
    @v_v.setter
    def v_v(self,value):
        self._v_v=value

    @property
    def t_u(self):
        return self._t_u
    @t_u.setter
    def t_u(self,value):
        self._t_u=value

    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self,value):
        self._alpha=value

    @property
    def c(self):
        return self._c
    @c.setter
    def c(self,value):
        self._c=value

    @property
    def e(self):
        return self._e
    @e.setter
    def e(self,value):
        self._e=value

    @property
    def recovery(self):
        return self._recovery
    @recovery.setter
    def recovery(self,value):
        self._recovery=value

    @property
    def eta(self):
        return self._eta
    @eta.setter
    def eta(self,value):
        self._eta=value

    # def __call__(self, *args, **kwargs):
    #     result=self.Optimal()
    #     return result

    def generate_freight_demand(self):
        value=np.max(self._m_j)
        d_i=np.zeros((self._routeNo,value))
        for row,item in enumerate(self._m_j):
            d_i[row, :item] = np.random.randint(40, 100, item)
        d_j=d_i.sum(axis=1)
        return d_i,d_j

    def SubProblem(self,y):
        '''
        :param y:
            N_hat:N_j_t
            N_bar:N_s
            q:q_j_t
            X:X_j_t
            delta:delta_j_t
            xi:xi_j_t
            zeta:zeta_j_t
        :return:
        '''
        m1=gp.Model('subproblem')
        m1.setParam('nonconvex', 2)
        m1.Params.timeLimit = 200

        index_line_period = gp.tuplelist([(line, time) for line in range(1, self._routeNo + 1) for time in range(1, self._period + 1)])

        S=m1.addVars(range(1,3),lb=1,ub=200,name='S')
        S_inverse=m1.addVars(range(1,3),name='S_inverse')
        #h_2=m1.addVars(index_line_period,lb=0.05,name='h_2')
        H=m1.addVars(index_line_period,lb=0.05,name='H')
        H_H=m1.addVars(index_line_period,name='H_H')
        #N_hat=m1.addVars(index_line_period,lb=1,ub=100,name='N_hat')
        #N_bar=m1.addVars(range(1,3),name='N_bar')
        u_0 = m1.addVars(index_line_period, name='u_0')
        u_1 = m1.addVars(index_line_period, name='u_1')
        u_2=m1.addVars(index_line_period,name='u_2')
        u_5=m1.addVars(index_line_period,name='u_5')
        u_6=m1.addVars(index_line_period,name='u_6')
        #u_u=m1.addVars(index_line_period,lb=-GRB.INFINITY,name='u_u')
        #h_h=m1.addVars(index_line_period,name='h_h')
        #u_3 = m1.addVar(name='u_3')
        #u_4 = m1.addVar(name='u_4')
        #u_0=m1.addVars(index_line_period,name='u_0')
        #u_1=m1.addVars(index_line_period,name='u_1')
        #u_2=m1.addVars(index_line_period,name='u_2')
        #u_2=m1.addVars(index_line_period,name='u_2')
        #u_3=m1.addVar(name='u_3')

        #2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
        #2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
        #y['delta'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]
        #(1-y['delta'][j, t])/ self._peak_point_demand[j - 1][t - 1] * S[2]
        #self._v_w*self._demand[j-1][t-1]+2*self._alpha*self._v_v*self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._peak_point_demand[j-1][t-1]*self._average_distance[j-1]*y['X'][j,t]*y['delta'][j,t]/(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._speed[j-1][t-1])
        # self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])
        # self._beta*y['delta'][j,t]*(1-(1-self._alpha)*y['X'][j,t])*S[1]
        # self._beta*(1-y['delta'][j,t])*S[2]
        #(2*self._distance[j-1]*(self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])+self._beta*y['delta'][j,t]*(1-(1-self._alpha)*y['X'][j,t])*S[1]+self._beta*(1-y['delta'][j,t])*S[2]))/self._speed[j-1][t-1]

        '''original
        m1.addConstrs((S[s]*S_inverse[s]==1 for s in range(1,3)),name='aux_0')
        m1.addConstrs(((self._v_w*self._demand[j-1][t-1]+2*self._alpha*self._v_v*self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._peak_point_demand[j-1][t-1]*self._average_distance[j-1]*y['X'][j,t]*y['delta'][j,t]/(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._speed[j-1][t-1]))*h_2[j,t]*h_2[j,t]-(2*self._distance[j-1]*(self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])+self._beta*y['delta'][j,t]*(1-(1-self._alpha)*y['X'][j,t])*S[1]+self._beta*(1-y['delta'][j,t])*S[2]))/self._speed[j-1][t-1]==0 for j,t in index_line_period),name='aux_1')
        m1.addConstrs((y['q'][j, t] * S[1] - self._eta * (S[2] - S[1]) * self._peak_point_demand[j-1][t-1] <= 0 for j,t in index_line_period),name='sub_0')
        m1.addConstrs((2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1])
                       -y['delta'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]*y['N_hat'][j,t]
                       -(1-y['delta'][j, t])/ self._peak_point_demand[j - 1][t - 1] * S[2]*y['N_hat'][j,t]<=0 for j,t in index_line_period),name='sub_1')
        m1.addConstrs((2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1])
                       -h_2[j,t]*y['N_hat'][j,t]<=0 for j,t in index_line_period),name='sub_2')
        m1.addConstr(self._eta*(S[1] - S[2]) + 1 <= 0,name='sub_3')
        m1.addConstr(gp.quicksum(
            u_0[j,t]*(y['q'][j, t] * S[1] - self._eta * (S[2] - S[1]) * self._peak_point_demand[j-1][t-1])+
            u_1[j,t]*(2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1])
                       -y['delta'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]*y['N_hat'][j,t]
                       -(1-y['delta'][j, t])/ self._peak_point_demand[j - 1][t - 1] * S[2]*y['N_hat'][j,t])+
            u_2[j,t]*(2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1])
                       -h_2[j,t]*y['N_hat'][j,t]) for j,t in index_line_period
        )+u_3*(self._eta*(S[1]-S[2])+1)>=-1e-4,name='sub_4')
        '''
        m1.addConstrs((S[s] * S_inverse[s] == 1 for s in range(1, 3)), name='aux_0')
        m1.addConstrs((H_H[j,t]==H[j,t]*H[j,t] for j,t in index_line_period),name='aux_1')
        # m1.addConstrs(((self._v_w*self._demand[j-1][t-1]+2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t])*h_2[j,t]*h_2[j,t]
        #                ==2*self._distance[j-1]/self._speed[j-1][t-1]*(
        #     self._gammar
        #     -self._gammar*(1-self._alpha)*y['X'][j,t]*y['delta'][j,t]
        #     +self._beta*y['delta'][j,t]*S[1]
        #     -self._beta*(1-self._alpha)*y['X'][j,t]*y['delta'][j,t]*S[1]
        #     +self._beta*(1-y['delta'][j,t])*S[2]
        #                )
        #     for j,t in index_line_period),name='aux_1'
        # )
        # m1.addVars((h_h[j,t]==h_2[j,t]*h_2[j,t] for j,t in index_line_period),name='aux_2')

        m1.addConstrs(
            (y['q'][j, t] * H[j,t] - self._eta * (S[2] - S[1]) <= 0
             for j, t in index_line_period), name='sub_0')
        m1.addConstrs((H[j, t]
                       - S[1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
                       - S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t]) <= 0
                       for j, t in index_line_period), name='sub_1')
        m1.addConstrs(
            (
                (self._v_w * self._demand[j - 1][t - 1]
                 + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] *
                 self._average_distance[j - 1] / self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] * y['delta'][
                     j, t]) * H[j, t] * H[j, t]
                - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                        self._gammar
                        - self._gammar * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]
                        + self._beta * y['delta'][j, t] * S[1]
                        - self._beta * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t] * S[1]
                        + self._beta * (1 - y['delta'][j, t]) * S[2]
                ) <= 0
                for j, t in index_line_period
            ), name='sub_2'
        )

        m1.addConstr(self._eta * (S[1] - S[2]) + 1 <= 0, name='sub_3')
        m1.addConstr(self._eta * (S[2] - S[1]) - 6 <= 0, name='sub_4')
        #m1.addConstr(S[1]<=80,name='sub_s')

        # m1.addConstrs((
        #     y['N_hat'][j, t] * H[j, t]
        #     - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1]*y['delta'][j,t]*y['X'][j,t]
        #     - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * S[1] * y['q'][j, t] * y['X'][j, t] * y['delta'][
        #         j, t]
        #     - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t]) <= 0
        #     for j, t in index_line_period), name='sub_5')
        m1.addConstrs((
            y['N_hat'][j, t] * H[j, t]
            - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['delta'][j, t] * y['X'][j, t]
            - 2 * self._t_u  * H[j,t] * y['q'][j, t] * y['X'][j, t] * y['delta'][
                j, t]
            - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])-1e-3 <= 0
            for j, t in index_line_period), name='sub_5')
        m1.addConstrs((
            -1e-3
            -y['N_hat'][j, t] * H[j, t]
            + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['delta'][j, t] * y['X'][j, t]
            + 2 * self._t_u * H[j, t] * y['q'][j, t] * y['X'][j, t] * y['delta'][
                j, t]
            + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t]) <= 0
            for j, t in index_line_period), name='sub_6')

        #m1.addConstrs((y['q'][j,t]*S[1]-self._eta*(S[2]-S[1])*self._peak_point_demand[j-1][t-1]<=0 for j,t in index_line_period),name='sub_0')
        # m1.addConstrs((-y['N_hat'][j, t] * H[j, t]
        #                + 2 * self._distance[j - 1] * y['X'][j, t] * y['delta'][j, t] * (self._alpha * self._distance[j - 1]*self._peak_point_demand[j-1][t-1] + self._t_u * y['q'][j, t] * self._speed[j-1][t-1]*S[1]) / (self._speed[j - 1][t - 1] * self._distance[j - 1]*self._peak_point_demand[j-1][t-1])
        #                + 2 * self._distance[j - 1] * (1 - y['X'][j, t] * y['delta'][j, t]) / self._speed[j - 1][t - 1] <= 0
        #                for j, t in index_line_period), name='sub_2')
        # m1.addConstrs((
        #     2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *y['delta'][j, t]
        #     +2*self._t_u*y['X'][j,t]*y['delta'][j,t]*y['q'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]
        #     +2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])
        #     -y['N_hat'][j,t]*S[1]*y['delta'][j,t]/self._peak_point_demand[j-1][t-1]
        #     -y['N_hat'][j,t]*S[2]*(1-y['delta'][j,t])/self._peak_point_demand[j-1][t-1]
        #     <=0
        #     for j,t in index_line_period),name='sub_1')
        # m1.addConstrs((
        #     2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][j, t]
        #     + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] / self._peak_point_demand[j - 1][t - 1] *
        #     S[1]
        #     + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
        #     - y['N_hat'][j, t]*h_2[j,t]
        #     <= 0
        #     for j, t in index_line_period), name='sub_2')

        #m1.addConstr(S[1]<=20,name='sub_6')
        # m1.addConstrs((-S[1]/self._peak_point_demand[j-1][t-1]+0.05<=0 for j,t in index_line_period),name='sub_5_1')
        # m1.addConstrs((-S[2] / self._peak_point_demand[j - 1][t - 1] + 0.05 <= 0 for j, t in index_line_period),name='sub_5_2')
        # m1.addConstrs((
        #         -S[1]*y['delta'][j,t]/self._peak_point_demand[j-1][t-1]
        #         -S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])+0.05<=0
        #         for j,t in index_line_period),name='sub_5')

        # m1.addConstrs((-N_bar[1]+N_hat.prod(y['delta'],'*',t)<=0 for t in range(1,self._period+1)),name='sub_6')
        # m1.addConstrs((-N_bar[2]
        #                +N_hat.sum('*',t)
        #                -N_hat.prod(y['delta'], '*', t) <= 0 for t in range(1, self._period + 1)),
        #               name='sub_7')
        # m1.addConstr(
        #     gp.quicksum(
        #         u_0[j,t]*(y['q'][j,t]*H[j,t]-self._eta*(S[2]-S[1]))
        #         +u_2[j,t]*(
        #                 -y['N_hat'][j, t] * H[j, t]
        #                 +2 * self._distance[j - 1] * y['X'][j, t] * y['delta'][j, t] * (self._alpha * self._distance[j - 1]*self._peak_point_demand[j-1][t-1] + self._t_u * y['q'][j, t] * self._speed[j-1][t-1]*S[1]) / (self._speed[j - 1][t - 1] * self._distance[j - 1]*self._peak_point_demand[j-1][t-1])
        #                 +2 * self._distance[j - 1] * (1 - y['X'][j, t] * y['delta'][j, t]) / self._speed[j - 1][t - 1]
        #         )
        #         for j,t in index_line_period
        #     )
        #     +u_3*(self._eta*(S[1]-S[2])+1)
        #     +u_4*(self._eta*(S[2]-S[1])-6)>=-1e-3
        # )
        # m1.addConstr(
        #     gp.quicksum(
        #         u_0[j, t] * (y['q'][j,t]*S[1]-self._eta*(S[2]-S[1])*self._peak_point_demand[j-1][t-1])
        #         + u_1[j, t] * (
        #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][j, t]
        #                 +2 * self._t_u * y['X'][j, t] * y['delta'][j,t] * y['q'][j, t] / self._peak_point_demand[j - 1][t - 1] * S[1]
        #                 +2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
        #                 -y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]*y['delta'][j,t]
        #                 -y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*S[2]
        #                 +y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*S[2]*y['delta'][j,t]
        #         )
        #         +u_2[j,t]*(
        #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][j, t]
        #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /self._peak_point_demand[j - 1][t - 1] * S[1]
        #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
        #                 -y['N_hat'][j,t]*h_2[j,t]
        #         )
        #         +u_5[j,t]*(
        #             -S[1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
        #             -S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])
        #             +0.05
        #         )
        #         +u_u[j,t]*(
        #             (self._v_w*self._demand[j-1][t-1]
        #              +2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t])*h_h[j,t]
        #             -2*self._distance[j-1]/self._speed[j-1][t-1]*(self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t]))
        #             -2*self._distance[j-1]/self._speed[j-1][t-1]*(self._beta*y['delta'][j,t]*S[1])
        #             +2*self._distance[j-1]/self._speed[j-1][t-1]*(self._beta*(1-self._alpha)*y['X'][j,t]*y['delta'][j,t]*S[1])
        #             -2*self._distance[j-1]/self._speed[j-1][t-1]*self._beta*(1-y['delta'][j,t])*S[2]
        #         )
        #         for j, t in index_line_period
        #     )>= -1e-3
        # )
        m1.addConstr(
            gp.quicksum(
                u_0[j, t] * (y['q'][j, t] * H[j,t] - self._eta * (S[2] - S[1]))
                + u_1[j, t] * (
                    H[j,t]
                    -S[1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
                    -S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t]))
                + u_2[j, t] * (
                    H_H[j,t]*(
                        self._v_w*self._demand[j-1][t-1]
                        +2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
                    )
                    -2*self._distance[j-1]/self._speed[j-1][t-1]*(
                        self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])
                        +self._beta*(y['delta'][j,t]-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])*S[1]
                        +self._beta*(1-y['delta'][j,t])*S[2]
                    )
                )
                # + u_5[j, t] * (
                #     y['N_hat'][j,t]*H[j,t]
                #     -2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*y['delta'][j,t]*y['X'][j,t]
                #     -2*self._t_u/self._peak_point_demand[j-1][t-1]*S[1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
                #     -2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])
                # )
                + u_5[j, t] * (
                        y['N_hat'][j, t] * H[j, t]
                        - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['delta'][j, t] *
                        y['X'][j, t]
                        - 2 * self._t_u * H[j,t] * y['q'][j, t] * y['X'][j, t] *
                        y['delta'][j, t]
                        - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])-1e-3
                )
                + u_6[j, t] * (
                        -1e-3
                        -y['N_hat'][j, t] * H[j, t]
                        + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['delta'][j, t] *
                        y['X'][j, t]
                        + 2 * self._t_u * H[j, t] * y['q'][j, t] * y['X'][j, t] *
                        y['delta'][j, t]
                        + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
                )
                for j, t in index_line_period
            ) >= -1e-3
        )

        #2*self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._gammar/self._speed[j-1][t-1]*S_inverse[1]+2*self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._beta/self._speed[j-1][t-1]+2*self._gammar*self._t_u*y['q'][j,t]+2*self._beta*self._t_u*y['q'][j,t]*S[1]
        #self._v_w*self._demand[j-1][t-1]/self._peak_point_demand[j-1][t-1]*S[1]
        #2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
        obj=gp.quicksum((2 * self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][
                            t - 1] * self._gammar / self._speed[j - 1][t - 1] * S_inverse[1] + 2 * self._alpha *
                                self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] * self._beta /
                                self._speed[j - 1][t - 1] + 2 * self._gammar * self._t_u * y['q'][
                                    j, t] + 2 * self._beta * self._t_u * y['q'][j, t] * S[1]+
                         self._v_w * self._demand[j - 1][t - 1] / self._peak_point_demand[j - 1][t - 1] * S[1]+
                         2 * self._v_v * self._demand[j - 1][t - 1] * self._average_distance[j - 1] * (
                                     self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][
                                 t - 1] + self._t_u * y['q'][j, t] * self._speed[j - 1][t - 1] * S[1]) / (
                                     self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][
                                 t - 1])
        )*y['X'][j,t]*y['delta'][j,t] for j,t in index_line_period
        )
        #2*self._distance[j-1]*(self._gammar+self._beta*S[1])*self._peak_point_demand[j-1][t-1]*S_inverse[1]/self._speed[j-1][t-1]
        #self._v_w * self._demand[j - 1][t - 1] / self._peak_point_demand[j - 1][t - 1] * S[1]
        #2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1]
        obj=obj+gp.quicksum(
            (
                    2 * self._distance[j - 1] * (self._gammar + self._beta * S[1]) * self._peak_point_demand[j - 1][
                t - 1] * S_inverse[1] / self._speed[j - 1][t - 1]
                    +self._v_w * self._demand[j - 1][t - 1] / self._peak_point_demand[j - 1][t - 1] * S[1]
                    +2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1]
            )*(1-y['X'][j,t])*y['delta'][j,t] for j,t in index_line_period
        )
        #2*self._distance[j-1]*(self._gammar+self._beta*S[2])*self._peak_point_demand[j-1][t-1]*S_inverse[2]/self._speed[j-1][t-1]
        #self._v_w * self._demand[j - 1][t - 1] / self._peak_point_demand[j - 1][t - 1] * S[2]
        #2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1]
        obj=obj+gp.quicksum(
            (
                    2 * self._distance[j - 1] * (self._gammar + self._beta * S[2]) * self._peak_point_demand[j - 1][
                t - 1] * S_inverse[2] / self._speed[j - 1][t - 1]
                    +self._v_w * self._demand[j - 1][t - 1] / self._peak_point_demand[j - 1][t - 1] * S[2]
                    +2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1]
            )*(1-y['delta'][j,t]) for j,t in index_line_period
        )
        obj=obj+gp.quicksum((self._c+self._e*S[s])*self._recovery/365*y['N_bar'][s] for s in range(1,3))
        obj=obj+(gp.quicksum(self._d_j)-gp.quicksum(y['q'][j,t] for j,t in index_line_period))*self._v_p
        # obj=obj+gp.quicksum(
        #     u_0[j,t]*(y['q'][j,t]*H[j,t]-self._eta*(S[2]-S[1]))
        #     +u_2[j,t]*(
        #         -y['N_hat'][j,t]*H[j,t]
        #         +2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
        #         +2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])
        #     )
        #
        #     for j,t in index_line_period
        # )+u_3*(self._eta*(S[1]-S[2])+1)+u_4*(self._eta*(S[2]-S[1])-6)
        obj=obj+gp.quicksum(
                u_0[j, t] * (y['q'][j, t] * H[j,t] - self._eta * (S[2] - S[1]))
                + u_1[j, t] * (
                    H[j,t]
                    -S[1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
                    -S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t]))
                + u_2[j, t] * (
                    H_H[j,t]*(
                        self._v_w*self._demand[j-1][t-1]
                        +2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
                    )
                    -2*self._distance[j-1]/self._speed[j-1][t-1]*(
                        self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])
                        +self._beta*(y['delta'][j,t]-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])*S[1]
                        +self._beta*(1-y['delta'][j,t])*S[2]
                    )
                )
                # + u_5[j, t] * (
                #     y['N_hat'][j,t]*H[j,t]
                #     -2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*y['X'][j,t]*y['delta'][j,t]
                #     -2*self._t_u/self._peak_point_demand[j-1][t-1]*S[1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
                #     -2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])
                # )
                + u_5[j, t] * (
                        y['N_hat'][j, t] * H[j, t]
                        - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                        y['delta'][j, t]
                        - 2 * self._t_u  * H[j,t] * y['q'][j, t] * y['X'][j, t] *
                        y['delta'][j, t]
                        - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
                        -1e-3
                )
                + u_6[j, t] * (
                        -1e-3
                        -y['N_hat'][j, t] * H[j, t]
                        + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                        y['delta'][j, t]
                        + 2 * self._t_u * H[j, t] * y['q'][j, t] * y['X'][j, t] *
                        y['delta'][j, t]
                        + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
                )
                for j, t in index_line_period
            )
        # obj = obj + gp.quicksum(
        #     u_0[j, t] * (y['q'][j, t] * S[1] - self._eta * (S[2] - S[1]) * self._peak_point_demand[j - 1][t - 1])
        #     + u_1[j, t] * (
        #             2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
        #         j, t]
        #             + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] / self._peak_point_demand[j - 1][
        #                 t - 1] * S[1]
        #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
        #             - y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]*S[1]
        #             -y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*S[2]
        #             +y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]*S[2]
        #     )
        #     + u_2[j, t] * (
        #             2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
        #         j, t]
        #             + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] / self._peak_point_demand[j - 1][
        #                 t - 1] * S[1]
        #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
        #             - y['N_hat'][j,t]*h_2[j,t]
        #     )
        #     + u_5[j, t] * (
        #             -S[1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
        #             - S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
        #             + 0.05
        #     )
        #     + u_u[j, t] * (
        #             (self._v_w * self._demand[j - 1][t - 1]
        #              + 2 * self._v_v * self._t_u  * self._demand[j - 1][t - 1] *
        #              self._average_distance[j - 1] / self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] * y['delta'][
        #                  j, t]) * h_h[j, t]
        #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
        #                         self._gammar * (1 - (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]))
        #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (self._beta * y['delta'][j, t] * S[1])
        #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
        #                         self._beta * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t] * S[1])
        #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * (1 - y['delta'][j, t]) * S[2]
        #     )
        #     for j, t in index_line_period
        # )

        m1.setObjective(obj,gp.GRB.MINIMIZE)
        m1.update()
        m1.write('out.lp')
        m1.optimize()

        print(m1.status)
        logger.info('Sub Problem status: {}'.format(m1.status))
        result_dict = {}
        if m1.status==GRB.OPTIMAL:
            # result_dict['S'] = dict(m1.getAttr('x', S))
            # print(result_dict['S'])
            # result_dict['h_2'] = dict(m1.getAttr('x', h_2))
            # result_dict['u_0'] = dict(m1.getAttr('x', u_0))
            # result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            # result_dict['u_2'] = dict(m1.getAttr('x', u_2))
            # result_dict['u_5'] = dict(m1.getAttr('x', u_5))
            # check_feasible=0
            # for j,t in index_line_period:
            #     if y['q'][j,t]*result_dict['S'][1]-self._eta*(result_dict['S'][2]-result_dict['S'][1])*self._peak_point_demand[j-1][t-1]>0:
            #         check_feasible+=1
            #     if (2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
            #     j, t]
            #         + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] / self._peak_point_demand[j - 1][
            #             t - 1] * result_dict['S'][1]
            #         + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #         - y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]*result_dict['S'][1]
            #         -y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*result_dict['S'][2]
            #         +y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]*result_dict['S'][2])>0:
            #         check_feasible+=1
            #     if (2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
            #     j, t]
            #         + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] / self._peak_point_demand[j - 1][
            #             t - 1] * result_dict['S'][1]
            #         + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #         - y['N_hat'][j,t]*result_dict['h_2'][j,t])>0:
            #         check_feasible+=1
            #     if (-result_dict['S'][1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
            #         - result_dict['S'][2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
            #         + 0.05)>0:
            #         check_feasible+=1
            # print(check_feasible)
            # if check_feasible==0:
            #     logger.info('The solution of subproblem is feasible, u is generated multiplier.')
            #     logger.info('The objective value of subproblem (feasible) is %s' % (m1.objVal))
            #     result_dict['objval'] = m1.objVal
            #     result_dict['u_0'] = dict(m1.getAttr('x', u_0))
            #     result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            #     result_dict['u_2'] = dict(m1.getAttr('x', u_2))
            #     result_dict['u_5'] = dict(m1.getAttr('x', u_5))
            #     result_dict['v_hat'] = {
            #         (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #             t - 1] / (
            #                         self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #                     t - 1] + self._t_u * y['q'][j, t] *
            #                         self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            #     result_dict['h_1'] = {
            #         (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
            #                 result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for
            #         j, t in index_line_period}
            #     result_dict['headway'] = {key: np.min((result_dict['h_1'][key], result_dict['h_2'][key])) for key in
            #                               index_line_period}
            #     result_dict['status'] = 1
            # else:
            #     logger.info('The solution of subproblem is infeasible, generate lambda.')
            #     m2 = gp.Model('infeasibleSubProblem')
            #     m2.setParam('nonconvex', 2)
            #     m2.Params.timeLimit = 200
            #     lambda_0 = m2.addVars(index_line_period, name='lambda_0')
            #     lambda_1 = m2.addVars(index_line_period, name='lambda_1')
            #     lambda_2 = m2.addVars(index_line_period, name='lambda_2')
            #     lambda_5=m2.addVars(index_line_period,name='lambda_5')
            #
            #     m2.addConstr(
            #         gp.quicksum(
            #             lambda_0[j,t]+lambda_1[j,t]+lambda_2[j,t]+lambda_5[j,t]
            #             for j,t in index_line_period
            #         )==1,name='scale_lambda'
            #     )
            #     m2_obj = gp.quicksum(
            #         lambda_0[j, t] * (
            #                 y['q'][j, t] * result_dict['S'][1] - self._eta * (result_dict['S'][2] - result_dict['S'][1]) * self._peak_point_demand[j - 1][
            #             t - 1])
            #         for j, t in index_line_period
            #     )
            #     m2_obj = m2_obj + gp.quicksum(
            #         lambda_1[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                             1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*result_dict['S'][1]*y['delta'][j,t]
            #                 -y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*result_dict['S'][2]
            #                 +y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*result_dict['S'][2]*y['delta'][j,t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     m2_obj = m2_obj + gp.quicksum(
            #         lambda_2[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][
            #                     j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                             1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j,t]*result_dict['h_2'][j,t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     m2_obj=m2_obj+gp.quicksum(
            #         lambda_5[j,t]*(
            #             -result_dict['S'][1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
            #             -result_dict['S'][2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])
            #             +0.05
            #         )
            #         for j,t in index_line_period
            #     )
            #     m2.addConstr(
            #         gp.quicksum(
            #             lambda_0[j, t] * (
            #                     y['q'][j, t] * result_dict['S'][1] - self._eta * (
            #                         result_dict['S'][2] - result_dict['S'][1]) * self._peak_point_demand[j - 1][
            #                         t - 1])
            #             for j, t in index_line_period
            #         )
            #         +gp.quicksum(
            #         lambda_1[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                             1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*result_dict['S'][1]*y['delta'][j,t]
            #                 -y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*result_dict['S'][2]
            #                 +y['N_hat'][j,t]/self._peak_point_demand[j-1][t-1]*result_dict['S'][2]*y['delta'][j,t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     +gp.quicksum(
            #         lambda_2[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][
            #                     j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                             1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j,t]*result_dict['h_2'][j,t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     +gp.quicksum(
            #         lambda_5[j,t]*(
            #             -result_dict['S'][1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
            #             -result_dict['S'][2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])
            #             +0.05
            #         )
            #         for j,t in index_line_period
            #     )>=1e-4
            #     )
            #
            #     m2.setObjective(m2_obj, gp.GRB.MINIMIZE)
            #     m2.update()
            #     m2.optimize()
            #
            #     print(m2.status)
            #     result_dict['objval'] = float('inf')
            #     result_dict['lambda_0'] = dict(m2.getAttr('x', lambda_0))
            #     result_dict['lambda_1'] = dict(m2.getAttr('x', lambda_1))
            #     result_dict['lambda_2'] = dict(m2.getAttr('x', lambda_2))
            #     result_dict['lambda_5'] = dict(m2.getAttr('x', lambda_5))
            #
            #     result_dict['v_hat'] = {
            #         (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #             t - 1] / (
            #                         self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #                     t - 1] + self._t_u *
            #                         y['q'][j, t] *
            #                         self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            #     result_dict['h_1'] = {
            #         (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
            #                 result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1]
            #         for j, t in index_line_period}
            #     # print(result_dict['S'])
            #     result_dict['headway'] = {key: np.min((result_dict['h_1'][key], result_dict['h_2'][key])) for key in
            #                               index_line_period}
            #
            #     result_dict['status'] = 2
            #
            #     logger.info('Sub Problem status (infeasible): {}'.format(m2.status))
            #     logger.info('The objective value of subproblem is %s' % (m2.objVal))
            logger.info('The objective value of subproblem is %s'%(m1.objVal))
            result_dict['objval']=m1.objVal
            result_dict['S']=dict(m1.getAttr('x',S))
            result_dict['headway'] = dict(m1.getAttr('x', H))
            #result_dict['h_2']=dict(m1.getAttr('x',h_2))
            #result_dict['headway']={(j,t):result_dict['S'][1]*y['delta'][j,t]/self._peak_point_demand[j-1][t-1]+result_dict['S'][2]*(1-y['delta'][j,t])/self._peak_point_demand[j-1][t-1] for j,t in index_line_period}
            result_dict['u_0']=dict(m1.getAttr('x',u_0))
            #result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            result_dict['u_2']=dict(m1.getAttr('x',u_2))
            result_dict['u_5'] = dict(m1.getAttr('x', u_5))
            result_dict['u_6'] = dict(m1.getAttr('x', u_6))
            #result_dict['u_u'] = dict(m1.getAttr('x', u_u))
            #result_dict['u_3'] = m1.getAttr('x', [u_3])[0]
            #result_dict['u_4'] = m1.getAttr('x', [u_4])[0]
            #logger.info(result_dict['u_3'])
            result_dict['v_hat']={(j,t):self._speed[j-1][t-1] * self._distance[j-1] * self._peak_point_demand[j-1][t-1] / (
                        self._alpha * self._distance[j-1] * self._peak_point_demand[j-1][t-1] + self._t_u * y['q'][j, t] *
                        self._speed[j-1][t-1] * result_dict['S'][1]) for j,t in index_line_period}
            result_dict['h_1']={(j,t):result_dict['S'][1]*y['delta'][j,t]/self._peak_point_demand[j-1][t-1]+result_dict['S'][2]*(1-y['delta'][j,t])/self._peak_point_demand[j-1][t-1] for j,t in index_line_period}
            #result_dict['headway']={key:np.min((result_dict['h_1'][key],result_dict['h_2'][key])) for key in index_line_period}
            # h2_hat={(j,t):np.sqrt(1/((self._v_w*self._demand[j-1][t-1]+2*self._alpha*self._v_v*self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._peak_point_demand[j-1][t-1]*self._average_distance[j-1]*y['X'][j,t]*y['delta'][j,t]/(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._speed[j-1][t-1]))*self._speed[j-1][t-1]/(2*self._distance[j-1]*(self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])+self._beta*y['delta'][j,t]*result_dict['S'][1]*(1-(1-self._alpha)*y['X'][j,t])+self._beta*(1-y['delta'][j,t])*result_dict['S'][2]))))
            #     for j,t in index_line_period
            #         }
            # result_dict['headway']={key:np.min((h1_hat[key],h2_hat[key])) for key in index_line_period}
            # result_dict['headway_1'] = h1_hat
            # result_dict['headway_2'] = h2_hat
            result_dict['status']=1
        elif m1.status == GRB.TIME_LIMIT:
            # result_dict['S'] = dict(m1.getAttr('x', S))
            # result_dict['h_2'] = dict(m1.getAttr('x', h_2))
            # check_feasible = 0
            # for j, t in index_line_period:
            #     if y['q'][j, t] * result_dict['S'][1] - self._eta * (result_dict['S'][2] - result_dict['S'][1]) * \
            #             self._peak_point_demand[j - 1][t - 1] > 0:
            #         check_feasible += 1
            #     if (2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
            #         j, t]
            #         + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] / self._peak_point_demand[j - 1][
            #             t - 1] * result_dict['S'][1]
            #         + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #         - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t] * result_dict['S'][1]
            #         - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * result_dict['S'][2]
            #         + y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t] * result_dict['S'][
            #             2]) > 0:
            #         check_feasible += 1
            #     if (2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
            #         j, t]
            #         + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] / self._peak_point_demand[j - 1][
            #             t - 1] * result_dict['S'][1]
            #         + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #         - y['N_hat'][j, t] * result_dict['h_2'][j, t]) > 0:
            #         check_feasible += 1
            #     if (-result_dict['S'][1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
            #         - result_dict['S'][2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
            #         + 0.05) > 0:
            #         check_feasible += 1
            # if check_feasible == 0:
            #     logger.info('The solution of subproblem is feasible (time limit), u is generated multiplier.')
            #     logger.info('The objective value of subproblem (feasible time limit) is %s' % (m1.objVal))
            #     result_dict['objval'] = m1.objVal
            #     result_dict['u_0'] = dict(m1.getAttr('x', u_0))
            #     result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            #     result_dict['u_2'] = dict(m1.getAttr('x', u_2))
            #     result_dict['u_5'] = dict(m1.getAttr('x', u_5))
            #     result_dict['v_hat'] = {
            #         (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #             t - 1] / (
            #                         self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #                     t - 1] + self._t_u * y['q'][j, t] *
            #                         self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            #     result_dict['h_1'] = {
            #         (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
            #                 result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for
            #         j, t in index_line_period}
            #     result_dict['headway'] = {key: np.min((result_dict['h_1'][key], result_dict['h_2'][key])) for key in
            #                               index_line_period}
            #     result_dict['status'] = 1
            # else:
            #     logger.info('The solution of subproblem is infeasible, generate lambda.')
            #     m2 = gp.Model('infeasibleSubProblem')
            #     m2.setParam('nonconvex', 2)
            #     m2.Params.timeLimit = 200
            #     lambda_0 = m2.addVars(index_line_period, name='lambda_0')
            #     lambda_1 = m2.addVars(index_line_period, name='lambda_1')
            #     lambda_2 = m2.addVars(index_line_period, name='lambda_2')
            #     lambda_5 = m2.addVars(index_line_period, name='lambda_5')
            #
            #     m2.addConstr(
            #         gp.quicksum(
            #             lambda_0[j, t] + lambda_1[j, t] + lambda_2[j, t] + lambda_5[j, t]
            #             for j, t in index_line_period
            #         ) == 1, name='scale_lambda'
            #     )
            #     m2_obj = gp.quicksum(
            #         lambda_0[j, t] * (
            #                 y['q'][j, t] * result_dict['S'][1] - self._eta * (
            #                     result_dict['S'][2] - result_dict['S'][1]) * self._peak_point_demand[j - 1][
            #                     t - 1])
            #         for j, t in index_line_period
            #     )
            #     m2_obj = m2_obj + gp.quicksum(
            #         lambda_1[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                         1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1] *
            #                 y['delta'][j, t]
            #                 - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * result_dict['S'][2]
            #                 + y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * result_dict['S'][2] *
            #                 y['delta'][j, t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     m2_obj = m2_obj + gp.quicksum(
            #         lambda_2[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][
            #                     j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                         1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j, t] * result_dict['h_2'][j, t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     m2_obj = m2_obj + gp.quicksum(
            #         lambda_5[j, t] * (
            #                 -result_dict['S'][1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
            #                 - result_dict['S'][2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
            #                 + 0.05
            #         )
            #         for j, t in index_line_period
            #     )
            #     m2.addConstr(
            #         gp.quicksum(
            #             lambda_0[j, t] * (
            #                     y['q'][j, t] * result_dict['S'][1] - self._eta * (
            #                     result_dict['S'][2] - result_dict['S'][1]) * self._peak_point_demand[j - 1][
            #                         t - 1])
            #             for j, t in index_line_period
            #         )
            #         + gp.quicksum(
            #             lambda_1[j, t] * (
            #                     2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                     y['delta'][j, t]
            #                     + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                     self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                     + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                             1 - y['X'][j, t] * y['delta'][j, t])
            #                     - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1] *
            #                     y['delta'][j, t]
            #                     - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * result_dict['S'][2]
            #                     + y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * result_dict['S'][2] *
            #                     y['delta'][j, t]
            #             )
            #             for j, t in index_line_period
            #         )
            #         + gp.quicksum(
            #             lambda_2[j, t] * (
            #                     2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                     y['delta'][
            #                         j, t]
            #                     + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                     self._peak_point_demand[j - 1][t - 1] * result_dict['S'][1]
            #                     + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                             1 - y['X'][j, t] * y['delta'][j, t])
            #                     - y['N_hat'][j, t] * result_dict['h_2'][j, t]
            #             )
            #             for j, t in index_line_period
            #         )
            #         + gp.quicksum(
            #             lambda_5[j, t] * (
            #                     -result_dict['S'][1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
            #                     - result_dict['S'][2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
            #                     + 0.05
            #             )
            #             for j, t in index_line_period
            #         ) >= 1e-4
            #     )
            #
            #     m2.setObjective(m2_obj, gp.GRB.MINIMIZE)
            #     m2.update()
            #     m2.optimize()
            #
            #     print(m2.status)
            #     result_dict['objval'] = float('inf')
            #     result_dict['lambda_0'] = dict(m2.getAttr('x', lambda_0))
            #     result_dict['lambda_1'] = dict(m2.getAttr('x', lambda_1))
            #     result_dict['lambda_2'] = dict(m2.getAttr('x', lambda_2))
            #     result_dict['lambda_5'] = dict(m2.getAttr('x', lambda_5))
            #
            #     result_dict['v_hat'] = {
            #         (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #             t - 1] / (
            #                         self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #                     t - 1] + self._t_u *
            #                         y['q'][j, t] *
            #                         self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            #     result_dict['h_1'] = {
            #         (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
            #                 result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1]
            #         for j, t in index_line_period}
            #     # print(result_dict['S'])
            #     result_dict['headway'] = {key: np.min((result_dict['h_1'][key], result_dict['h_2'][key])) for key in
            #                               index_line_period}
            #
            #     result_dict['status'] = 2
            #
            #     logger.info('Sub Problem status (infeasible): {}'.format(m2.status))
            #     logger.info('The objective value of subproblem (time limit) is %s' % (m2.objVal))

            result_dict['objval'] = m1.objVal
            result_dict['S'] = dict(m1.getAttr('x', S))
            result_dict['headway'] = dict(m1.getAttr('x', H))
            #result_dict['h_2'] = dict(m1.getAttr('x', h_2))
            #result_dict['N_hat']=dict(m1.getAttr('x',N_hat))
            #result_dict['N_bar'] = dict(m1.getAttr('x', N_bar))
            # result_dict['headway'] = {
            #     (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
            #             result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for j, t in
            #     index_line_period}
            result_dict['u_0'] = dict(m1.getAttr('x', u_0))
            # result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            result_dict['u_2']=dict(m1.getAttr('x',u_2))
            result_dict['u_5'] = dict(m1.getAttr('x', u_5))
            result_dict['u_6'] = dict(m1.getAttr('x', u_6))
            #result_dict['u_u'] = dict(m1.getAttr('x', u_u))
            #result_dict['u_6'] = dict(m1.getAttr('x', u_6))
            #result_dict['u_7'] = dict(m1.getAttr('x', u_7))
            #result_dict['u_3'] = m1.getAttr('x', [u_3])[0]
            #result_dict['u_4'] = m1.getAttr('x', [u_4])[0]
            # logger.info(result_dict['u_3'])
            result_dict['v_hat'] = {
                (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] / (
                        self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] + self._t_u *
                        y['q'][j, t] *
                        self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            result_dict['h_1'] = {
                (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
                        result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for j, t in
                index_line_period}
            #result_dict['headway'] = {key: np.min((result_dict['h_1'][key], result_dict['h_2'][key])) for key in index_line_period}
            '''
            result_dict['objval'] = m1.objVal
            result_dict['S'] = dict(m1.getAttr('x', S))
            result_dict['u_0'] = dict(m1.getAttr('x', u_0))
            result_dict['u_1'] = dict(m1.getAttr('x', u_1))
            result_dict['u_2'] = dict(m1.getAttr('x', u_2))
            result_dict['u_3'] = m1.getAttr('x', [u_3])[0]
            result_dict['v_hat'] = {
                (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] / (
                        self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] + self._t_u *
                        y['q'][j, t] *
                        self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            h1_hat = {(j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
                              result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for
                      j, t in index_line_period}
            '''
            # h2_hat = {(j, t): np.sqrt(1 / ((self._v_w * self._demand[j - 1][
            #     t - 1] + 2 * self._alpha * self._v_v * self._t_u * y['q'][j, t] * self._speed[j - 1][t - 1] *
            #                                 self._demand[j - 1][t - 1] * self._peak_point_demand[j - 1][t - 1] *
            #                                 self._average_distance[j - 1] * y['X'][j, t] * y['delta'][j, t] / (
            #                                             self._alpha * self._distance[j - 1] *
            #                                             self._peak_point_demand[j - 1][t - 1] * self._speed[j - 1][
            #                                                 t - 1])) * self._speed[j - 1][t - 1] / (
            #                                            2 * self._distance[j - 1] * (self._gammar * (
            #                                                1 - (1 - self._alpha) * y['X'][j, t] * y['delta'][
            #                                            j, t]) + self._beta * y['delta'][j, t] * result_dict['S'][1] * (
            #                                                                                     1 - (1 - self._alpha) *
            #                                                                                     y['X'][
            #                                                                                         j, t]) + self._beta * (
            #                                                                                     1 - y['delta'][j, t]) *
            #                                                                         result_dict['S'][2]))))
            #           for j, t in index_line_period
            #           }
            '''
            h2_hat=m1.getAttr('x',h_2)
            result_dict['headway'] = {key: np.min((h1_hat[key], h2_hat[key])) for key in index_line_period}
            result_dict['headway_1']=h1_hat
            result_dict['headway_2']=h2_hat
            '''
            result_dict['status']=1
        else:
            logger.info('The solution of subproblem is infeasible, generate lambda.')
            m2=gp.Model('infeasibleSubProblem')
            m2.setParam('nonconvex', 2)
            m2.Params.timeLimit = 200

            m2_S = m2.addVars(range(1, 3), lb=1,ub=200, name='m2_S')
            m2_H = m2.addVars(index_line_period, lb=0.05, name='m2_H')
            m2_H_H = m2.addVars(index_line_period, name='m2_H_H')
            lambda_0 = m2.addVars(index_line_period, name='lambda_0')
            lambda_1 = m2.addVars(index_line_period, name='lambda_1')
            lambda_2 = m2.addVars(index_line_period, name='lambda_2')
            lambda_5 = m2.addVars(index_line_period, name='lambda_5')
            lambda_6 = m2.addVars(index_line_period, name='lambda_6')
            m2_inf=m2.addVar(lb=1e-6,name='m2_inf')

            m2.addConstr(
                gp.quicksum(
                    lambda_0[j, t] + lambda_1[j, t] + lambda_2[j, t]+lambda_5[j,t]+lambda_6[j,t] for j, t in index_line_period
                ) == 1, name='scale_lambda')
            m2.addConstrs((m2_H_H[j, t] == m2_H[j, t] * m2_H[j, t] for j, t in index_line_period), name='in_aux_0')
            m2.addConstr(self._eta * (m2_S[1] - m2_S[2]) + 1 <= 0, name='in_sub_3')
            m2.addConstr(self._eta * (m2_S[2] - m2_S[1]) - 6 <= 0, name='in_sub_4')
            m2.addConstrs((m2_H[j,t]<=0.2 for j,t in index_line_period),name='in_sub_6')
            # m2.addConstrs(
            #     (
            #         y['N_hat'][j,t]*m2_H[j,t]
            #         -2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*y['X'][j,t]*y['delta'][j,t]
            #         -2*self._t_u/self._peak_point_demand[j-1][t-1]*m2_S[1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
            #         -2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])
            #         <=0
            #         for j,t in index_line_period
            #     ),name='in_sub_5'
            # )
            m2.addConstrs(
                (
                    y['N_hat'][j, t] * m2_H[j, t]
                    - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
                        j, t]
                    - 2 * self._t_u * m2_H[j,t] * y['q'][j, t] * y['X'][j, t] *
                    y['delta'][j, t]
                    - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
                    -1e-3
                    <= 0
                    for j, t in index_line_period
                ), name='in_sub_5'
            )
            m2.addConstrs(
                (
                    -1e-3
                    -y['N_hat'][j, t] * m2_H[j, t]
                    + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
                        j, t]
                    + 2 * self._t_u * m2_H[j, t] * y['q'][j, t] * y['X'][j, t] *
                    y['delta'][j, t]
                    + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
                    <= 0
                    for j, t in index_line_period
                ), name='in_sub_6'
            )
            m2_obj = gp.quicksum(
                lambda_0[j, t] * (
                        y['q'][j, t] * m2_H[j, t] - self._eta * (m2_S[2] - m2_S[1]))
                for j, t in index_line_period
            )
            m2_obj = m2_obj + gp.quicksum(
                lambda_1[j, t] * (
                        m2_H[j, t]
                        - m2_S[1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
                        - m2_S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
                )
                for j, t in index_line_period
            )
            m2_obj = m2_obj + gp.quicksum(
                lambda_2[j, t] * (
                        m2_H_H[j, t] * (
                        self._v_w * self._demand[j - 1][t - 1]
                        + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                        self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] * y['delta'][j, t]
                )
                        - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                self._gammar
                                - self._gammar * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]
                                + self._beta * (
                                            y['delta'][j, t] - (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]) *
                                m2_S[1]
                                + self._beta * (1 - y['delta'][j, t]) * m2_S[2]
                        )
                )
                for j, t in index_line_period
            )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_5[j, t] * (
            #             y['N_hat'][j, t] * m2_H[j, t]
            #             - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #             y['delta'][j, t]
            #             - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * m2_S[1] * y['q'][j, t] * y['X'][
            #                 j, t] * y['delta'][j, t]
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #     )
            #     for j, t in index_line_period
            # )
            m2_obj = m2_obj + gp.quicksum(
                lambda_5[j, t] * (
                        y['N_hat'][j, t] * m2_H[j, t]
                        - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                        y['delta'][j, t]
                        - 2 * self._t_u  * m2_H[j,t] * y['q'][j, t] * y['X'][
                            j, t] * y['delta'][j, t]
                        - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
                        -1e-3
                )
                for j, t in index_line_period
            )
            m2_obj = m2_obj + gp.quicksum(
                lambda_6[j, t] * (
                        -1e-3
                        -y['N_hat'][j, t] * m2_H[j, t]
                        + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                        y['delta'][j, t]
                        + 2 * self._t_u * m2_H[j, t] * y['q'][j, t] * y['X'][
                            j, t] * y['delta'][j, t]
                        + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
                )
                for j, t in index_line_period
            )
            m2.addConstr(
                gp.quicksum(
                    lambda_0[j, t] * (
                            y['q'][j, t] * m2_H[j, t] - self._eta * (m2_S[2] - m2_S[1]))
                    for j, t in index_line_period
                )
                + gp.quicksum(
                    lambda_1[j, t] * (
                            m2_H[j, t]
                            - m2_S[1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
                            - m2_S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
                    )
                    for j, t in index_line_period
                )
                + gp.quicksum(
                    lambda_2[j, t] * (
                            m2_H_H[j, t] * (
                            self._v_w * self._demand[j - 1][t - 1]
                            + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                            self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] * y['delta'][j, t]
                    )
                            - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                    self._gammar
                                    - self._gammar * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]
                                    + self._beta * (y['delta'][j, t] - (1 - self._alpha) * y['X'][j, t] * y['delta'][
                                j, t]) * m2_S[1]
                                    + self._beta * (1 - y['delta'][j, t]) * m2_S[2]
                            )
                    )
                    for j, t in index_line_period
                )
                # + gp.quicksum(
                #     lambda_5[j, t] * (
                #             y['N_hat'][j, t] * m2_H[j, t]
                #             - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                #             y['delta'][j, t]
                #             - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * m2_S[1] * y['q'][j, t] * y['X'][
                #                 j, t] * y['delta'][j, t]
                #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                #                         1 - y['X'][j, t] * y['delta'][j, t])
                #     )
                #     for j, t in index_line_period
                # )
                + gp.quicksum(
                    lambda_5[j, t] * (
                            y['N_hat'][j, t] * m2_H[j, t]
                            - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                            y['delta'][j, t]
                            - 2 * self._t_u  * m2_H[j,t] * y['q'][j, t] * y['X'][
                                j, t] * y['delta'][j, t]
                            - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                    1 - y['X'][j, t] * y['delta'][j, t])
                            -1e-3
                    )
                    for j, t in index_line_period
                )
                + gp.quicksum(
                    lambda_6[j, t] * (
                            -1e-3
                            -y['N_hat'][j, t] * m2_H[j, t]
                            + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                            y['delta'][j, t]
                            + 2 * self._t_u * m2_H[j, t] * y['q'][j, t] * y['X'][
                                j, t] * y['delta'][j, t]
                            + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                    1 - y['X'][j, t] * y['delta'][j, t])
                    )
                    for j, t in index_line_period
                )
                -m2_inf
                ==0
            )
            '''
            m2_S=y['S']
            m2_H=y['H']
            lambda_0 = m2.addVars(index_line_period, name='lambda_0')
            lambda_1 = m2.addVars(index_line_period, name='lambda_1')
            lambda_2 = m2.addVars(index_line_period, name='lambda_2')
            #lambda_5 = m2.addVars(index_line_period, lb=-GRB.INFINITY, name='lambda_5')
            m2.addConstr(
                gp.quicksum(
                    lambda_0[j, t] + lambda_1[j, t] + lambda_2[j, t] for j, t in index_line_period
                ) == 1, name='scale_lambda')
            m2_obj = gp.quicksum(
                lambda_0[j, t] * (
                        y['q'][j, t] * m2_H[j, t] - self._eta * (m2_S[2] - m2_S[1]))
                for j, t in index_line_period
            )
            m2_obj = m2_obj + gp.quicksum(
                lambda_1[j, t] * (
                        m2_H[j, t]
                        - m2_S[1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
                        - m2_S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
                )
                for j, t in index_line_period
            )
            m2_obj = m2_obj + gp.quicksum(
                lambda_2[j, t] * (
                        m2_H[j,t]*m2_H[j, t] * (
                        self._v_w * self._demand[j - 1][t - 1]
                        + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                        self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] * y['delta'][j, t]
                )
                        - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                self._gammar
                                - self._gammar * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]
                                + self._beta * (
                                        y['delta'][j, t] - (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]) *
                                m2_S[1]
                                + self._beta * (1 - y['delta'][j, t]) * m2_S[2]
                        )
                )
                for j, t in index_line_period
            )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_5[j, t] * (
            #             y['N_hat'][j, t] * m2_H[j, t]
            #             - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #             y['delta'][j, t]
            #             - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * m2_S[1] * y['q'][j, t] * y['X'][
            #                 j, t] * y['delta'][j, t]
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #     )
            #     for j, t in index_line_period
            # )
            m2.addConstr(
                gp.quicksum(
                    lambda_0[j, t] * (
                            y['q'][j, t] * m2_H[j, t] - self._eta * (m2_S[2] - m2_S[1]))
                    for j, t in index_line_period
                )
                + gp.quicksum(
                    lambda_1[j, t] * (
                            m2_H[j, t]
                            - m2_S[1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
                            - m2_S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
                    )
                    for j, t in index_line_period
                )
                + gp.quicksum(
                    lambda_2[j, t] * (
                            m2_H[j,t]*m2_H[j, t] * (
                            self._v_w * self._demand[j - 1][t - 1]
                            + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                            self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] * y['delta'][j, t]
                    )
                            - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                    self._gammar
                                    - self._gammar * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]
                                    + self._beta * (y['delta'][j, t] - (1 - self._alpha) * y['X'][j, t] * y['delta'][
                                j, t]) * m2_S[1]
                                    + self._beta * (1 - y['delta'][j, t]) * m2_S[2]
                            )
                    )
                    for j, t in index_line_period
                )
                # + gp.quicksum(
                #     lambda_5[j, t] * (
                #             y['N_hat'][j, t] * m2_H[j, t]
                #             - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
                #             y['delta'][j, t]
                #             - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * m2_S[1] * y['q'][j, t] * y['X'][
                #                 j, t] * y['delta'][j, t]
                #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                #                         1 - y['X'][j, t] * y['delta'][j, t])
                #     )
                #     for j, t in index_line_period
                # )
                >= 1e-4
            )
            '''
            #m2_obj=m2.addVar(lb=10,name='m2_obj')
            #m2_H=m2.addVars(index_line_period,lb=0.05,ub=0.9,name='m2_H')
            #m2_h_2 = m2.addVars(index_line_period,lb=0.05, name='m2_h_2')
            # m2_H = m2.addVars(index_line_period, lb=0.05, name='m2_H')
            # m2_H_H = m2.addVars(index_line_period,name='m2_H_H')
            #m2_h_h=m2.addVars(index_line_period,name='m2_h_h')
            #m2_N_hat=m2.addVars(index_line_period,lb=1,ub=100,name='m2_N_hat')
            #m2_N_bar=m2.addVars(range(1,3),name='m2_N_bar')
            #m2_aux_N_1=m2.addVars(index_line_period,name='m2_aux_N_1')
            #m2_aux_N_2=m2.addVars(index_line_period,name='m2_aux_N_2')
            # lambda_0 = m2.addVars(index_line_period, name='lambda_0')
            # #lambda_1 = m2.addVars(index_line_period, name='lambda_1')
            # lambda_1 = m2.addVars(index_line_period, name='lambda_1')
            # lambda_2=m2.addVars(index_line_period,name='lambda_2')
            # lambda_5=m2.addVars(index_line_period,lb=-GRB.INFINITY,name='lambda_5')
            #lambda_lambda=m2.addVars(index_line_period,lb=-GRB.INFINITY,name='lambda_lambda')
            #lambda_6 = m2.addVars(range(1,self._period+1), name='lambda_6')
            #lambda_7 = m2.addVars(range(1,self._period+1), name='lambda_7')
            #lambda_3 = m2.addVar(name='lambda_3')
            #lambda_4 = m2.addVar(name='lambda_4')

            # m2.addConstrs(
            #     (
            #         m2_h_h[j,t]==m2_h_2[j,t]*m2_h_2[j,t]
            #         for j,t in index_line_period
            #     ),name='in_aux_0'
            # )
            # m2.addConstr(
            #     gp.quicksum(
            #         lambda_0[j, t]+lambda_1[j,t]+ lambda_2[j, t] for j, t in index_line_period
            #     )== 1,name='scale_lambda')
            # m2.addConstrs(((self._v_w * self._demand[j - 1][t - 1] + 2 * self._v_v * self._t_u * self._demand[j - 1][
            #     t - 1] * self._average_distance[j - 1] / self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] *
            #                 y['delta'][j, t]) * m2_h_2[j, t] * m2_h_2[j, t]
            #                == 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                        self._gammar
            #                        - self._gammar * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]
            #                        + self._beta * y['delta'][j, t] * m2_S[1]
            #                        - self._beta * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t] * m2_S[1]
            #                        + self._beta * (1 - y['delta'][j, t]) * m2_S[2]
            #                )
            #                for j, t in index_line_period), name='m2_aux_0'
            #               )
            # m2.addConstrs(
            #     (   m2_aux_N_1[j,t]==
            #         m2_N_hat[j, t] * m2_S[1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1]
            #         + m2_N_hat[j, t] * m2_S[2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1]
            #         for j,t in index_line_period
            #     ),name='c_m2_aux_N_1'
            # )
            # m2.addConstrs(
            #     (
            #         m2_aux_N_2[j,t]==
            #         m2_N_hat[j,t]*m2_h_2[j,t]
            #         for j,t in index_line_period
            #     ),name='c_m2_aux_N_2'
            # )
            # m2.addConstrs(
            #     (y['q'][j, t] * m2_H[j, t] - self._eta * (m2_S[2] - m2_S[1]) <= 0
            #      for j, t in index_line_period), name='in_sub_0')
            # m2.addConstrs(
            #     (y['q'][j, t] * m2_S[1]- self._eta * (m2_S[2] - m2_S[1])*self._peak_point_demand[j-1][t-1] <= 0
            #      for j, t in index_line_period), name='in_sub_0')
            # m2.addConstrs((-y['N_hat'][j, t] * m2_H[j, t]
            #                + 2 * self._distance[j - 1] * y['X'][j, t] * y['delta'][j, t] * (
            #                            self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][
            #                        t - 1] + self._t_u * y['q'][j, t] * self._speed[j - 1][t - 1] * m2_S[1]) / (
            #                            self._speed[j - 1][t - 1] * self._distance[j - 1] *
            #                            self._peak_point_demand[j - 1][t - 1])
            #                + 2 * self._distance[j - 1] * (1 - y['X'][j, t] * y['delta'][j, t]) / self._speed[j - 1][
            #                    t - 1] <= 0
            #                for j, t in index_line_period), name='in_sub_2')
            #m2.addConstrs((m2_h_h[j,t]==m2_h_2[j,t]*m2_h_2[j,t] for j,t in index_line_period),name='in_aux_0')
            # m2.addConstrs(
            #     (
            #         y['q'][j,t]*m2_S[1]-self._eta*(m2_S[2]-m2_S[1])*self._peak_point_demand[j-1][t-1]<=0
            #         for j,t in index_line_period
            #     ),name='in_sub_0'
            # )
            # m2.addConstrs((m2_H_H[j,t]==m2_H[j,t]*m2_H[j,t] for j,t in index_line_period),name='in_aux_0')
            # m2.addConstr(self._eta * (m2_S[1] - m2_S[2]) + 1 <= 0, name='in_sub_3')
            # m2.addConstr(self._eta * (m2_S[2] - m2_S[1]) - 6 <= 0, name='in_sub_4')
            #m2.addConstr(m2_S[1]<=80,name='in_sub_s')
            # m2.addConstrs(
            #     (
            #         y['N_hat'][j,t]*m2_H[j,t]
            #         -2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*y['delta'][j,t]*y['X'][j,t]
            #         -2*self._t_u/self._peak_point_demand[j-1][t-1]*m2_S[1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
            #         -2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])==0
            #         for j,t in index_line_period
            #     ),name='in_sub_5'
            # )
            # m2.addConstrs(
            #     (
            #         -m2_S[1]*y['delta'][j,t]/self._peak_point_demand[j-1][t-1]
            #         -m2_S[2]*(1-y['delta'][j,t])/self._peak_point_demand[j-1][t-1]
            #         +0.05<=0
            #         for j,t in index_line_period
            #     ),name='in_sub_5'
            # )
            # m2.addConstrs(
            #     (
            #         (self._v_w * self._demand[j - 1][t - 1]
            #         + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] *
            #         self._average_distance[j - 1] / self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] *
            #         y['delta'][j, t]) * m2_h_2[j, t]*m2_h_2[j,t]
            #         - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (self._gammar * (1 - (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]))
            #         - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (self._beta * y['delta'][j, t] * m2_S[1])
            #         + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (self._beta * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t] * m2_S[1])
            #         - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * (1 - y['delta'][j, t]) *m2_S[2]==0
            #         for j,t in index_line_period
            #     ),name='in_aux_0'
            # )
            # m2.addConstrs((-m2_S[1]/self._peak_point_demand[j-1][t-1]+0.05<=0 for j,t in index_line_period),name='in_sub_5_1')
            # m2.addConstrs((-m2_S[2] / self._peak_point_demand[j - 1][t - 1] + 0.05 <= 0 for j, t in index_line_period),
            #               name='in_sub_5_2')
            # m2.addConstrs(
            #     (
            #         -m2_S[1]/self._peak_point_demand[j-1][t-1]
            #         -m2_S[2]/self._peak_point_demand[j-1][t-1]
            #         +0.05<=0
            #         for j,t in index_line_period
            #     ),name='in_sub_5'
            # )
            # m2.addConstrs((
            #     -m2_N_bar[1]
            #     +m2_N_hat.prod(y['delta'],"*",t)<=0
            #     for t in range(1,self._period+1)
            # ),name='in_sub_6')
            # m2.addConstrs(
            #     (
            #         -m2_N_bar[2]
            #         +sum(m2_N_hat[item1,item2] for item1,item2 in m2_N_hat.keys() if item2==t)
            #         -m2_N_hat.prod(y['delta'],"*",t)<=0
            #         for t in range(1,self._period+1)
            #     ),name='in_sub_7'
            # )
            #
            # m2_obj = gp.quicksum(
            #     lambda_0[j, t] * (
            #             y['q'][j, t] * m2_S[1] - self._eta * (m2_S[2] - m2_S[1])*self._peak_point_demand[j-1][t-1])
            #     for j, t in index_line_period
            # )
            # m2_obj = gp.quicksum(
            #     lambda_0[j, t] * (
            #             y['q'][j, t] * m2_H[j,t] - self._eta * (m2_S[2] - m2_S[1]))
            #     for j, t in index_line_period
            # )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_1[j, t] * (
            #             2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][j, t]
            #             +2 * self._t_u * y['X'][j, t] * y['delta'][j,t] * y['q'][j, t] / self._peak_point_demand[j - 1][t - 1] * m2_S[1]
            #             +2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #             -y['N_hat'][j,t]*m2_S[1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
            #             -y['N_hat'][j,t]*m2_S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])
            #             #-m2_aux_N_1[j,t]
            #     )
            #     for j, t in index_line_period
            # )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_1[j, t] * (
            #         m2_H[j,t]
            #         -m2_S[1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
            #         -m2_S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])
            #     )
            #     for j, t in index_line_period
            # )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_2[j, t] * (
            #             2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] * y['delta'][
            #         j, t]
            #             + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #             self._peak_point_demand[j - 1][t - 1] * m2_S[1]
            #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - y['X'][j, t] * y['delta'][j, t])
            #             - y['N_hat'][j,t]*m2_h_2[j,t]
            #     )
            #     for j, t in index_line_period
            # )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_2[j, t] * (
            #         m2_H_H[j,t]*(
            #             self._v_w*self._demand[j-1][t-1]
            #             +2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
            #         )
            #         -2*self._distance[j-1]/self._speed[j-1][t-1]*(
            #             self._gammar
            #             -self._gammar*(1-self._alpha)*y['X'][j,t]*y['delta'][j,t]
            #             +self._beta*(y['delta'][j,t]-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])*m2_S[1]
            #             +self._beta*(1-y['delta'][j,t])*m2_S[2]
            #         )
            #     )
            #     for j, t in index_line_period
            # )
            # m2_obj=m2_obj+gp.quicksum(
            #     lambda_5[j,t]*(
            #         -m2_S[1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
            #         -m2_S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])
            #         +0.05
            #     )
            #     for j,t in index_line_period
            # )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_5[j, t] * (
            #         y['N_hat'][j,t]*m2_H[j,t]
            #         -2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*y['X'][j,t]*y['delta'][j,t]
            #         -2*self._t_u/self._peak_point_demand[j-1][t-1]*m2_S[1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
            #         -2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])
            #     )
            #     for j, t in index_line_period
            # )
            # m2_obj=m2_obj+gp.quicksum(
            #     lambda_lambda[j, t] * (
            #             (self._v_w * self._demand[j - 1][t - 1]
            #              + 2 * self._v_v * self._t_u * self._speed[j - 1][t - 1] * self._demand[j - 1][t - 1] *
            #              self._average_distance[j - 1] / self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] *
            #              y['delta'][
            #                  j, t]) * m2_h_h[j, t]
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                     self._gammar * (1 - (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]))
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (self._beta * y['delta'][j, t] * m2_S[1])
            #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                     self._beta * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t] * m2_S[1])
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * (1 - y['delta'][j, t]) *
            #             m2_S[2]
            #     )
            #     for j,t in index_line_period
            # )
            # m2_obj=m2_obj+gp.quicksum(
            #     lambda_6[t]*(
            #         -m2_N_bar[1]
            #         +m2_N_hat.prod(y['delta'],'*',t)
            #     )
            #     for t in range(1,self._period+1)
            # )
            # m2_obj = m2_obj + gp.quicksum(
            #     lambda_7[t] * (
            #             -m2_N_bar[2]
            #             +m2_N_hat.sum('*',t)
            #             - m2_N_hat.prod(y['delta'], '*', t)
            #     )
            #     for t in range(1, self._period + 1)
            # )
            #m2_obj = m2_obj + lambda_3 * (self._eta * (m2_S[1] - m2_S[2]) + 1)
            #m2_obj=m2_obj+lambda_4*(self._eta*(m2_S[2]-m2_S[1])-6)
            # m2.addConstr(
            #     gp.quicksum(
            #         lambda_0[j, t] * (
            #                 y['q'][j, t] * m2_S[1] - self._eta * (
            #                 m2_S[2] - m2_S[1]) * self._peak_point_demand[j - 1][
            #                     t - 1])
            #         for j, t in index_line_period
            #     )
            #     + gp.quicksum(
            #         lambda_1[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * m2_S[1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                         1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * m2_S[1] *
            #                 y['delta'][j, t]
            #                 - y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * m2_S[2]
            #                 + y['N_hat'][j, t] / self._peak_point_demand[j - 1][t - 1] * m2_S[2] *
            #                 y['delta'][j, t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     + gp.quicksum(
            #         lambda_2[j, t] * (
            #                 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * y['X'][j, t] *
            #                 y['delta'][
            #                     j, t]
            #                 + 2 * self._t_u * y['X'][j, t] * y['delta'][j, t] * y['q'][j, t] /
            #                 self._peak_point_demand[j - 1][t - 1] * m2_S[1]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                         1 - y['X'][j, t] * y['delta'][j, t])
            #                 - y['N_hat'][j, t] * m2_h_2[j,t]
            #         )
            #         for j, t in index_line_period
            #     )
            #     + gp.quicksum(
            #         lambda_5[j, t] * (
            #                 -m2_S[1] / self._peak_point_demand[j - 1][t - 1] * y['delta'][j, t]
            #                 -m2_S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - y['delta'][j, t])
            #                 + 0.05
            #         )
            #         for j, t in index_line_period
            #     )
            #     # +gp.quicksum(
            #     #     lambda_lambda[j, t] * (
            #     #         (self._v_w * self._demand[j - 1][t - 1]
            #     #          + 2 * self._v_v * self._t_u * self._speed[j - 1][t - 1] * self._demand[j - 1][t - 1] *
            #     #          self._average_distance[j - 1] / self._distance[j - 1] * y['q'][j, t] * y['X'][j, t] *
            #     #          y['delta'][
            #     #              j, t]) * m2_h_h[j, t]
            #     #         - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #     #                 self._gammar * (1 - (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]))
            #     #         - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (self._beta * y['delta'][j, t] * m2_S[1])
            #     #         + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #     #                 self._beta * (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t] * m2_S[1])
            #     #         - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * (1 - y['delta'][j, t]) *
            #     #         m2_S[2]
            #     #     )
            #     #     for j,t in index_line_period
            #     # )
            #     >= 1e-4
            # )
            # m2.addConstr(
            #     gp.quicksum(
            #         lambda_0[j, t] * (
            #                 y['q'][j, t] * m2_H[j, t] - self._eta * (m2_S[2] - m2_S[1]))
            #         for j, t in index_line_period
            #     )
            #     + gp.quicksum(
            #         lambda_1[j, t] * (
            #             m2_H[j,t]
            #             -m2_S[1]/self._peak_point_demand[j-1][t-1]*y['delta'][j,t]
            #             -m2_S[2]/self._peak_point_demand[j-1][t-1]*(1-y['delta'][j,t])
            #         )
            #         for j, t in index_line_period
            #     )
            #     + gp.quicksum(
            #         lambda_2[j, t] * (
            #             m2_H_H[j,t]*(
            #                 self._v_w*self._demand[j-1][t-1]
            #                 +2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
            #             )
            #             -2*self._distance[j-1]/self._speed[j-1][t-1]*(
            #                 self._gammar
            #                 -self._gammar*(1-self._alpha)*y['X'][j,t]*y['delta'][j,t]
            #                 +self._beta*(y['delta'][j,t]-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])*m2_S[1]
            #                 +self._beta*(1-y['delta'][j,t])*m2_S[2]
            #             )
            #         )
            #         for j, t in index_line_period
            #     )
            #     + gp.quicksum(
            #         lambda_5[j, t] * (
            #             y['N_hat'][j,t]*m2_H[j,t]
            #             -2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*y['X'][j,t]*y['delta'][j,t]
            #             -2*self._t_u/self._peak_point_demand[j-1][t-1]*m2_S[1]*y['q'][j,t]*y['X'][j,t]*y['delta'][j,t]
            #             -2*self._distance[j-1]/self._speed[j-1][t-1]*(1-y['X'][j,t]*y['delta'][j,t])
            #         )
            #         for j, t in index_line_period
            #     )
            #         >= 1e-4
            # )
            '''
            m2.addConstrs(((self._v_w * self._demand[j - 1][t - 1] + 2 * self._alpha * self._v_v * self._t_u * y['q'][
                j, t] * self._speed[j - 1][t - 1] * self._demand[j - 1][t - 1] * self._peak_point_demand[j - 1][t - 1] *
                            self._average_distance[j - 1] * y['X'][j, t] * y['delta'][j, t] / (
                                        self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] *
                                        self._speed[j - 1][t - 1])) * m2_h_2[j, t] * m2_h_2[j, t] == (
                                       2 * self._distance[j - 1] * (self._gammar * (
                                           1 - (1 - self._alpha) * y['X'][j, t] * y['delta'][j, t]) + self._beta *
                                                                    y['delta'][j, t] * (
                                                                                1 - (1 - self._alpha) * y['X'][j, t]) *
                                                                    m2_S[1] + self._beta * (1 - y['delta'][j, t]) * m2_S[
                                                                        2])) / self._speed[j - 1][t - 1] for j, t in
                           index_line_period), name='m2_aux_0')
            m2.addConstr(gp.quicksum(lambda_0[j,t]+lambda_1[j,t]+lambda_2[j,t] for j,t in index_line_period)+lambda_3==1,'scale_lambda')

            m2_obj=gp.quicksum(
                lambda_0[j,t]*(y['q'][j, t] * m2_S[1] - self._eta * (m2_S[2] - m2_S[1]) * self._peak_point_demand[j-1][t-1])
                for j,t in index_line_period
            )
            m2_obj=m2_obj+gp.quicksum(
                lambda_1[j,t]*(2 * self._distance[j - 1] * y['X'][j, t] * y['delta'][j, t] * (
                            self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] + self._t_u *
                            y['q'][j, t] * self._speed[j - 1][t - 1] * m2_S[1]) / (
                            self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1])
                + 2 * self._distance[j - 1] * (1 - y['X'][j, t] * y['delta'][j, t]) / (
                            self._speed[j - 1][t - 1] )
                - y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] * m2_S[1]*y['N_hat'][j,t]
                - (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] * m2_S[2]*y['N_hat'][j,t])
                for j,t in index_line_period
            )
            m2_obj=m2_obj+gp.quicksum(
                lambda_2[j,t]*(2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*m2_S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1])
                       -m2_h_2[j,t]*y['N_hat'][j,t])
                for j,t in index_line_period
            )
            m2_obj=m2_obj+lambda_3*(self._eta*(m2_S[1]-m2_S[2])+1)
            m2.addConstr(m2_obj>=10)
            '''
            m2.setObjective(m2_inf,gp.GRB.MINIMIZE)
            m2.update()
            m2.optimize()
            print(m2.status)
            result_dict['objval'] = float('inf')
            result_dict['S'] = dict(m2.getAttr('x', m2_S))
            result_dict['headway'] = dict(m2.getAttr('x', m2_H))
            #result_dict['h_2']=dict(m2.getAttr('x',m2_h_2))
            # result_dict['S'] = s_initial
            # result_dict['h_2'] = h_initial
            # result_dict['N_hat']=dict(m2.getAttr('x',m2_N_hat))
            # result_dict['N_bar'] = dict(m2.getAttr('x', m2_N_bar))
            #print(result_dict['h_2'])

            # result_dict['headway'] = {
            #     (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
            #             result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1]
            #     for j, t in index_line_period}
            result_dict['lambda_0'] = dict(m2.getAttr('x', lambda_0))
            #result_dict['lambda_1'] = dict(m2.getAttr('x', lambda_1))
            result_dict['lambda_1']=dict(m2.getAttr('x',lambda_1))
            result_dict['lambda_2'] = dict(m2.getAttr('x', lambda_2))
            result_dict['lambda_5'] = dict(m2.getAttr('x', lambda_5))
            result_dict['lambda_6'] = dict(m2.getAttr('x', lambda_6))
            #result_dict['lambda_lambda']=dict(m2.getAttr('x',lambda_lambda))
            # result_dict['lambda_6'] = dict(m2.getAttr('x', lambda_6))
            # result_dict['lambda_7'] = dict(m2.getAttr('x', lambda_7))
            #result_dict['lambda_3'] = m2.getAttr('x', [lambda_3])[0]
            #result_dict['lambda_4'] = m2.getAttr('x', [lambda_4])[0]
            # logger.info("lambda_0 is \n %s"%(result_dict['lambda_0']))
            # logger.info("lambda_1 is \n %s" % (result_dict['lambda_1']))
            # logger.info("lambda_2 is \n %s" % (result_dict['lambda_2']))
            # logger.info("lambda_3 is \n %s" % (result_dict['lambda_3']))
            #logger.info("lambda is %s"%(gp.quicksum(m2.getAttr('x', lambda_0)[item]+m2.getAttr('x', lambda_1)[item]+m2.getAttr('x', lambda_2)[item] for item in index_line_period)+m2.getAttr('x', [lambda_3])[0]))
            # logger.info(result_dict['u_3'])

            result_dict['v_hat'] = {
                (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] / (
                        self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] + self._t_u *
                        y['q'][j, t] *
                        self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            result_dict['h_1'] = {
                (j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
                        result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for j, t in
                index_line_period}
            #print(result_dict['S'])
            #result_dict['headway'] = {key: np.min((result_dict['h_1'][key], result_dict['h_2'][key])) for key in index_line_period}
            '''
            result_dict['v_hat'] = {
                (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] / (
                        self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] + self._t_u *
                        y['q'][j, t] *
                        self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            '''
            '''
            h1_hat = {(j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
                              result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for
                      j, t in index_line_period}
            h2_hat = m2.getAttr('x', m2_h_2)
            result_dict['headway'] = {key: np.min((h1_hat[key], h2_hat[key])) for key in index_line_period}
            result_dict['headway_1'] = h1_hat
            result_dict['headway_2'] = h2_hat
            '''
            '''
            result_dict['headway_2']= dict(m2.getAttr('x', m2_h_2))
            '''
            result_dict['status'] = 2

            logger.info('Sub Problem status (infeasible): {}'.format(m2.status))
            logger.info('The objective value of subproblem is %s' % (m2.objVal))
        return result_dict


            # self._result = m.getAttr('x', [c_o, c_uw, c_uv, c_u, c_total, c_p])
            # self._size_volume = m.getAttr('x', size_volume)
            # self._bus_operating = m.getAttr('x', bus_operating)
            # self._h_jts = m.getAttr('x', h_jts)
            # self._delta_jts = m.getAttr('x', delta_jts)
            # self._n_s = m.getAttr('x', n_s)
            # self._h_jts_1 = m.getAttr('x', h_jts_1)
            # self._h_jts_2 = m.getAttr('x', h_jts_2)
            # self._c_o_j_t_s = m.getAttr('x', c_o_j_t_s)
            # self._c_uw_j_t_s = m.getAttr('x', c_uw_j_t_s)
            # self._c_uv_j_t_s = m.getAttr('x', c_uv_j_t_s)
            # self._c_u_j_t_s = m.getAttr('x', c_u_j_t_s)
            # self._c_j_t_s = m.getAttr('x', c_j_t_s)
            # self._c_o_j_t = m.getAttr('x', c_o_j_t)
            # self._c_uw_j_t = m.getAttr('x', c_uw_j_t)
            # self._c_uv_j_t = m.getAttr('x', c_uv_j_t)
            # self._c_u_j_t = m.getAttr('x', c_u_j_t)
            # self._c_j_t = m.getAttr('x', c_j_t)
            # self._n_jts = m.getAttr('x', n_jts)
            # self._n_jt = m.getAttr('x', n_jt)
            # self._n_ts = m.getAttr('x', n_ts)

    def SetupMasterProblemModel(self):
        m=gp.Model('MasterProblem')
        m.Params.timeLimit = 60
        index_line_period = gp.tuplelist(
            [(line, time) for line in range(1, self._routeNo + 1) for time in range(1, self._period + 1)])

        # N_hat: N_j_t
        # N_bar: N_s
        # q: q_j_t
        # X: X_j_t
        # delta: delta_j_t
        # xi: xi_j_t
        # zeta: zeta_j_t
        y_0 = m.addVar(name='y_0')
        N_hat=m.addVars(index_line_period,lb=1,ub=100,name='N_hat')
        N_tilde=m.addVars(index_line_period,name='N_tilde')
        N_bar=m.addVars(range(1,3),name='N_bar')
        q=m.addVars(index_line_period,ub=30,name='q')
        X=m.addVars(index_line_period,vtype=GRB.BINARY,name='X')
        delta=m.addVars(index_line_period,vtype=GRB.BINARY,name='delta')
        xi=m.addVars(index_line_period,vtype=GRB.BINARY,name='xi')
        zeta=m.addVars(index_line_period,name='zeta')

        m.addConstrs((xi[item]<=delta[item] for item in index_line_period),name='c_1')
        m.addConstrs((xi[item]<=X[item] for item in index_line_period),name='c_2')
        m.addConstrs((xi[item]>=X[item]+delta[item]-1 for item in index_line_period),name='c_3')
        m.addConstrs((delta[item]+X[item]<=2 for item in index_line_period),name='c_4')
        m.addConstrs((delta[item]>=X[item] for item in index_line_period),name='c_5')
        m.addConstr(gp.quicksum(delta[item] for item in index_line_period)>=1,name='c_6')#> -> >=
        m.addConstrs((q.sum(j,'*')<=self._d_j[j-1] for j in range(1,self._routeNo+1)),name='c_7')
        m.addConstrs((q[j,t]>=0 for j,t in index_line_period),name='c_8')
        m.addConstrs((q[j,t]<=self._d_j[j-1] for j,t in index_line_period),name='c_9')
        m.addConstrs((q[j,t]>=(X[j,t]-1)*self._d_j[j-1]+1 for j,t in index_line_period),name='c_10')#> -> >=
        m.addConstrs((q[j,t]<=X[j,t]*self._d_j[j-1] for j,t in index_line_period),name='c_11')
        m.addConstrs((zeta[j,t]-xi[j,t]*self._d_j[j-1]<=0 for j,t in index_line_period),name='c_12')
        m.addConstrs((zeta[j,t]>=0 for j,t in index_line_period),name='c_13')
        m.addConstrs((zeta[j,t]-q[j,t]<=0 for j,t in index_line_period), name='c_14')
        m.addConstrs((zeta[j,t]-q[j,t]+self._d_j[j-1]-xi[j,t]*self._d_j[j-1]>=0 for j,t in index_line_period), name=
                     'c_15')
        m.addConstrs((zeta[j,t]<=self._d_j[j-1] for j,t in index_line_period),name='c_15_0')
        if self._alpha>=1:
            m.addConstrs((N_hat[j,t]>=10*self._distance[j-1]/self._speed[j-1][t-1] for j,t in index_line_period),name='c_16_0')
            m.addConstrs((N_hat[j,t]<=40*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]+1/3 for j,t in index_line_period),name='c_16_1')
        else:
            m.addConstrs(
                (N_hat[j, t] >= 10*self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1]+1/3 for j, t in index_line_period),
                name='c_16_0')
            m.addConstrs(
                (N_hat[j, t] <= 40 * self._distance[j - 1] / self._speed[j - 1][t - 1] for j, t in index_line_period),
                name='c_16_1')
        m.addConstrs((N_tilde[j,t]-100*delta[j,t]<=0 for j,t in index_line_period),name='c_17')
        m.addConstrs((N_tilde[j,t]-delta[j,t]>=0 for j,t in index_line_period),name='c_18')
        m.addConstrs((N_tilde[j,t]-N_hat[j,t]+100-100*delta[j,t]>=0 for j,t in index_line_period),name='c_19')
        m.addConstrs((N_tilde[j,t]-N_hat[j,t]<=0 for j,t in index_line_period),name='c_20')
        m.addConstrs((N_bar[1]>=N_tilde.sum('*',t) for t in range(1,self._period+1)),name='c_21')
        m.addConstrs((N_bar[2]>=N_hat.sum('*',t)-N_tilde.sum('*',t) for t in range(1,self._period+1)),name='c_22')
        m.addConstrs((N_hat[j,t]>=1 for j,t in index_line_period),name='c_23')#> -> >=
        m.addConstrs((N_bar[s]>=0 for s in range(1,3)),name='c_24')#> -> >=

        m.setObjective(y_0,sense=GRB.MINIMIZE)
        #m.Params.lazyConstraints=1
        m.update()
        return m

    def solveMaster(self,m,sub_result_dict):
        index_line_period = gp.tuplelist(
            [(line, time) for line in range(1, self._routeNo + 1) for time in range(1, self._period + 1)])
        S=sub_result_dict['S']#S[s]
        H=sub_result_dict['headway']
        v_hat=sub_result_dict['v_hat']
        h_1=sub_result_dict['h_1']
        #h_2=sub_result_dict['h_2']
        # N_hat=sub_result_dict['N_hat']
        # N_bar=sub_result_dict['N_bar']
        if sub_result_dict['status']==1:
            u_0=sub_result_dict['u_0']#u_0[j,t]
            u_1 = sub_result_dict['u_1']  # u_1[j,t]
            u_2 = sub_result_dict['u_2']  # u_2[j,t]
            u_5=sub_result_dict['u_5']
            u_6 = sub_result_dict['u_6']
            #u_u=sub_result_dict['u_u']
            # u_6=sub_result_dict['u_6']
            # u_7=sub_result_dict['u_7']
            #u_3 = sub_result_dict['u_3']
            #u_4 = sub_result_dict['u_4']# u_3
        else:
            lambda_0=sub_result_dict['lambda_0']#lambda_0[j,t]
            lambda_1=sub_result_dict['lambda_1']#lambda_1[j,t]
            lambda_2=sub_result_dict['lambda_2']#lambda_2[j,t]
            lambda_5=sub_result_dict['lambda_5']
            lambda_6 = sub_result_dict['lambda_6']
            #lambda_lambda=sub_result_dict['lambda_lambda']
            # lambda_6=sub_result_dict['lambda_6']
            # lambda_7=sub_result_dict['lambda_7']
            #lambda_3=sub_result_dict['lambda_3']#lambda_3
            #lambda_4 = sub_result_dict['lambda_4']  # lambda_3
        '''
        v_hat=sub_result_dict['v_hat']#v_hat[j,t]
        #headway = sub_result_dict['headway']  # headway[j,t]
        #headway_1 = sub_result_dict['headway_1']
        headway_2 = sub_result_dict['headway_2']
        '''

        m_y_0=m.getVarByName('y_0')
        m_N_hat=gp.tupledict({(j,t):m.getVarByName('N_hat['+str(j)+','+str(t)+']') for j,t in index_line_period})
        m_N_tilde = gp.tupledict({(j, t): m.getVarByName('N_tilde[' + str(j) + ',' + str(t) + ']') for j, t in index_line_period})
        m_N_bar=gp.tupledict({j:m.getVarByName('N_bar['+str(j)+']') for j in range(1,3)})
        m_q = gp.tupledict(
            {(j, t): m.getVarByName('q[' + str(j) + ',' + str(t) + ']') for j, t in index_line_period})
        m_X = gp.tupledict(
            {(j, t): m.getVarByName('X[' + str(j) + ',' + str(t) + ']') for j, t in index_line_period})
        m_delta = gp.tupledict(
            {(j, t): m.getVarByName('delta[' + str(j) + ',' + str(t) + ']') for j, t in index_line_period})
        m_xi = gp.tupledict(
            {(j, t): m.getVarByName('xi[' + str(j) + ',' + str(t) + ']') for j, t in index_line_period})
        m_zeta = gp.tupledict(
            {(j, t): m.getVarByName('zeta[' + str(j) + ',' + str(t) + ']') for j, t in index_line_period})

        #xi[j,t]:2*(self._alpha-1)*self._distance[j-1]*(self._gammar+self._beta*S[1])*self._peak_point_demand[j-1][t-1]/(self._speed[j-1][t-1]*S[1])+2*(self._alpha-1)*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1]
        #delta[j,t]:2*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._gammar*(S[2]-S[1])/(self._speed[j-1][t-1]*S[1]*S[2])+self._v_w*self._demand[j-1][t-1]*(S[1]-S[2])/self._peak_point_demand[j-1][t-1]
        #zeta[j,t]:2*(self._gammar+self._beta*S[1])*self._t_u+2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]*self._t_u*S[1]/(self._distance[j-1]*self._peak_point_demand[j-1][t-1])
        #2*self._distance[j-1]*(self._gammar+self._beta*S[2])*self._peak_point_demand[j-1][t-1]/(self._speed[j-1][t-1]*S[2])+self._v_w*self._demand[j-1][t-1]*S[2]/self._peak_point_demand[j-1][t-1]+2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1]
        #m_N_bar[s]*(self._c+self._e*S[s])*self._recovery/365
        #u_0:m_q[j,t]*S[1]-self._eta*(S[2]-S[1])*self._peak_point_demand[j-1][t-1]
        #u_1:2*self._distance[j-1]/v_hat[j,t]*m_xi[j,t]+2*self._distance[j-1]/self._speed[j-1][t-1]*(1-m_xi[j,t])-headway[j,t]*m_N_hat[j,t]
        #u_2:2 * self._distance[j - 1] / v_hat[j, t] * m_xi[j, t] + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t]) - headway[j, t] * m_N_hat[j, t]
        # u_3:S[1]-S[2]+0.5

        if sub_result_dict['status']==1:
            m.addConstr(m_y_0 >=
                        gp.quicksum(
                            m_xi[j, t] * (2 * (self._alpha - 1) * self._distance[j - 1] * (
                                        self._gammar + self._beta * S[1]) * self._peak_point_demand[j - 1][t - 1] / (
                                                      self._speed[j - 1][t - 1] * S[1]) + 2 * (
                                                      self._alpha - 1) * self._v_v * self._demand[j - 1][t - 1] *
                                          self._average_distance[j - 1] / self._speed[j - 1][t - 1]) +
                            m_delta[j, t] * (2 * self._distance[j - 1] * self._peak_point_demand[j - 1][
                                t - 1] * self._gammar * (S[2] - S[1]) / (
                                                         self._speed[j - 1][t - 1] * S[1] * S[2]) + self._v_w *
                                             self._demand[j - 1][t - 1] * (S[1] - S[2]) /
                                             self._peak_point_demand[j - 1][t - 1]) +
                            m_zeta[j, t] * (2 * (self._gammar + self._beta * S[1]) * self._t_u + 2 * self._v_v *
                                            self._demand[j - 1][t - 1] * self._average_distance[j - 1] * self._t_u * S[
                                                1] / (self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1])) +
                            2 * self._distance[j - 1] * (self._gammar + self._beta * S[2]) *
                            self._peak_point_demand[j - 1][t - 1] / (self._speed[j - 1][t - 1] * S[2]) + self._v_w *
                            self._demand[j - 1][t - 1] * S[2] / self._peak_point_demand[j - 1][t - 1] + 2 * self._v_v *
                            self._demand[j - 1][t - 1] * self._average_distance[j - 1] / self._speed[j - 1][t - 1]
                            for j, t in index_line_period
                        ) +
                        gp.quicksum(
                            m_N_bar[s] * (self._c + self._e * S[s]) * self._recovery / 365
                            for s in range(1, 3)
                        ) +
                        gp.quicksum(
                            u_0[j, t] * (m_q[j,t]*H[j,t]-self._eta*(S[2]-S[1]))
                            for j, t in index_line_period
                        ) +
                        # gp.quicksum(
                        #     u_1[j, t] * (
                        #                  +2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*m_xi[j,t]
                        #                  +2*self._t_u*S[1]/self._peak_point_demand[j-1][t-1]*m_zeta[j,t]
                        #                  +2*self._distance[j-1]/self._speed[j-1][t-1]*(1-m_xi[j,t])
                        #                  -S[1]/self._peak_point_demand[j-1][t-1]*m_N_tilde[j,t]
                        #                  -S[2]/self._peak_point_demand[j-1][t-1]*m_N_hat[j,t]
                        #                  +S[2]/self._peak_point_demand[j-1][t-1]*m_N_tilde[j,t]
                        #                  )
                        #     for j, t in index_line_period
                        # ) +
                        # gp.quicksum(
                        #     u_2[j, t] * (
                        #             +2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                        #             + 2 * self._t_u * S[1] / self._peak_point_demand[j - 1][t - 1] * m_zeta[j, t]
                        #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                        #             - h_2[j,t]*m_N_hat[j,t]
                        #     )
                        #     for j, t in index_line_period
                        # )
                        # +gp.quicksum(
                        #     u_5[j,t]*(
                        #         -S[1]/self._peak_point_demand[j-1][t-1]*m_delta[j,t]
                        #         -S[2]/self._peak_point_demand[j-1][t-1]*(1-m_delta[j,t])
                        #         +0.05
                        #     )
                        #     for j,t in index_line_period
                        # )
                        # +gp.quicksum(
                        #     u_u[j,t]*(
                        #         (self._v_w*self._demand[j-1][t-1]+
                        #          2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*m_zeta[j,t])
                        #         *h_2[j,t]*h_2[j,t]
                        #         -2*self._distance[j-1]/self._speed[j-1][t-1]*(self._gammar-(1-self._alpha)*m_xi[j,t])
                        #         -2*self._distance[j-1]/self._speed[j-1][t-1]*self._beta*S[1]*m_delta[j,t]
                        #         +2*self._distance[j-1]/self._speed[j-1][t-1]*self._beta*S[1]*(1-self._alpha)*m_xi[j,t]
                        #         -2*self._distance[j-1]/self._speed[j-1][t-1]*self._beta*S[2]*(1-m_delta[j,t])
                        #     )
                        #     for j,t in index_line_period
                        # )
                        gp.quicksum(
                            u_1[j, t] * (
                                    H[j,t]
                                    -S[1]/self._peak_point_demand[j-1][t-1]*m_delta[j,t]
                                    -S[2]/self._peak_point_demand[j-1][t-1]*(1-m_delta[j,t])
                            )
                            for j, t in index_line_period
                        ) +
                        gp.quicksum(
                            u_2[j, t] * (
                                H[j,t]*H[j,t]*(
                                    self._v_w*self._demand[j-1][t-1]
                                    +2*self._v_v*self._t_u*self._demand[j-1][t-1]*self._average_distance[j-1]/self._distance[j-1]*m_zeta[j,t]
                                )
                                -2*self._distance[j-1]/self._speed[j-1][t-1]*(
                                    self._gammar*(1-(1-self._alpha)*m_xi[j,t])
                                    +self._beta*(m_delta[j,t]-(1-self._alpha)*m_xi[j,t])*S[1]
                                    +self._beta*(1-m_delta[j,t])*S[2]
                                )
                            )
                            for j, t in index_line_period
                        )
                        # + gp.quicksum(
                        #     u_5[j, t] * (
                        #         m_N_hat[j,t]*H[j,t]
                        #         -2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*m_xi[j,t]
                        #         -2*self._t_u/self._peak_point_demand[j-1][t-1]*S[1]*m_zeta[j,t]
                        #         -2*self._distance[j-1]/self._speed[j-1][t-1]*(1-m_xi[j,t])
                        #     )
                        #     for j, t in index_line_period
                        # )
                        + gp.quicksum(
                            u_5[j, t] * (
                                m_N_hat[j, t] * H[j, t]
                                - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                                - 2 * self._t_u * H[j,t] * m_zeta[j, t]
                                - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                                -1e-3
                            )
                            for j, t in index_line_period
                        )
                        + gp.quicksum(
                            u_6[j, t] * (
                                -1e-3
                                -m_N_hat[j, t] * H[j, t]
                                + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                                + 2 * self._t_u * H[j, t] * m_zeta[j, t]
                                + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                            )
                            for j, t in index_line_period
                        )
            #             + gp.quicksum(
            #     u_u[j, t] * (
            #             (self._v_w * self._demand[j - 1][t - 1] +
            #              2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
            #              self._distance[j - 1] * m_zeta[j, t])
            #             * h_2[j, t] * h_2[j, t]
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #                         self._gammar - (1 - self._alpha) * m_xi[j, t])
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * S[1] * m_delta[j, t]
            #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * S[1] * (
            #                         1 - self._alpha) * m_xi[j, t]
            #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * S[2] * (
            #                         1 - m_delta[j, t])
            #     )
            #     for j, t in index_line_period
            # )
                        #+
                        # gp.quicksum(
                        #     u_6[t]*(
                        #         -N_bar[1]
                        #         +m_delta.prod(N_hat,'*',t)
                        #     )
                        #     for t in range(1,self._period+1)
                        # )+
                        # gp.quicksum(
                        #     u_7[t]*(
                        #         -N_bar[2]
                        #         +sum(N_hat[item1,item2] for item1,item2 in N_hat.keys() if item2==t)
                        #         -m_delta.prod(N_hat,'*',t)
                        #     )
                        #     for t in range(1,self._period+1)
                        # )+
                        #u_3 * (self._eta*(S[1] - S[2]) + 1) +
                        #u_4*(self._eta*(S[2]-S[1])-6)+
                        +(gp.quicksum(self._d_j) - gp.quicksum(m_q[j, t] for j, t in index_line_period)) * self._v_p
                        )
            '''
            m.addConstr(
                gp.quicksum(
                    u_0[j, t] * (m_q[j, t] * H[j, t] - self._eta * (S[2] - S[1]))
                    for j, t in index_line_period
                ) +
                gp.quicksum(
                    u_1[j, t] * (
                            H[j, t] - S[1] / self._peak_point_demand[j - 1][t - 1]
                    )
                    for j, t in index_line_period
                ) +
                gp.quicksum(
                    u_2[j, t] * (
                            H[j, t] * H[j, t] * (
                            self._v_w * self._demand[j - 1][t - 1]
                            + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                            self._distance[j - 1] * m_zeta[j, t]
                    )
                            - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                    self._gammar * (1 - (1 - self._alpha) * m_xi[j, t])
                                    + self._beta * (m_delta[j, t] - (1 - self._alpha) * m_xi[j, t]) * S[1]
                                    + self._beta * (1 - m_delta[j, t]) * S[2]
                            )
                    )
                    for j, t in index_line_period
                )
                + gp.quicksum(
                    u_5[j, t] * (
                            -m_N_hat[j, t] * H[j, t]
                            + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                            + 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * S[1] * m_zeta[j, t]
                            + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                    )
                    for j, t in index_line_period
                )<=-1e-4
            )
            '''
        else:
            '''
            m.addConstr(
                gp.quicksum(
                    lambda_0[j,t]*S[1]*m_q[j,t]-lambda_0[j,t]*(S[2]-S[1])*self._eta*self._peak_point_demand[j-1][t-1]
                    for j,t in index_line_period
                )+
                gp.quicksum(
                    lambda_1[j,t]*(2 * self._distance[j - 1] / v_hat[j, t] * m_xi[j, t]
                    + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t]) - headway[j, t] * m_N_hat[j, t])
                    for j,t in index_line_period
                )+
                gp.quicksum(
                    lambda_2[j, t] * (2 * self._distance[j - 1] / v_hat[j, t] * m_xi[j, t]
                    + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t]) - headway[j, t] * m_N_hat[j, t])
                    for j,t in index_line_period
                )+
                lambda_3*(self._eta*(S[1]-S[2])+1)<=0
            )
            '''
            # m.addConstr(
            #     gp.quicksum(
            #         lambda_0[j,t]*(m_q[j,t]*S[1]-self._eta*(S[2]-S[1])*self._peak_point_demand[j-1][t-1])
            #         for j,t in index_line_period
            #     )+
            #     gp.quicksum(
            #         lambda_1[j,t]*(
            #                 + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
            #                 + 2 * self._t_u * S[1] / self._peak_point_demand[j - 1][t - 1] * m_zeta[j, t]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
            #                 -S[1]/self._peak_point_demand[j-1][t-1]*m_N_tilde[j,t]
            #                 -S[2]/self._peak_point_demand[j-1][t-1]*m_N_hat[j,t]
            #                 +S[2]/self._peak_point_demand[j-1][t-1]*m_N_tilde[j,t]
            #         )
            #         for j,t in index_line_period
            #     )+
            #     gp.quicksum(
            #         lambda_2[j, t] * (
            #                 + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
            #                 + 2 * self._t_u * S[1] / self._peak_point_demand[j - 1][t - 1] * m_zeta[j, t]
            #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
            #                 - h_2[j,t]*m_N_hat[j,t]
            #         )
            #         for j, t in index_line_period
            #     )+
            #     gp.quicksum(
            #         lambda_5[j, t] * (
            #                 -S[1] / self._peak_point_demand[j - 1][t - 1] * m_delta[j, t]
            #                 - S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - m_delta[j, t])
            #                 + 0.05
            #         )
            #         for j, t in index_line_period
            #     )
            #     # + gp.quicksum(
            #     #     lambda_lambda[j, t] * (
            #     #             (self._v_w * self._demand[j - 1][t - 1] +
            #     #              2 * self._v_v * self._t_u * self._speed[j - 1][t - 1] * self._demand[j - 1][
            #     #                  t - 1] * self._average_distance[j-1] / self._distance[j-1] * m_zeta[j, t])
            #     #             * h_2[j, t] * h_2[j, t]
            #     #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
            #     #                         self._gammar - (1 - self._alpha) * m_xi[j, t])
            #     #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * S[1] * m_delta[j, t]
            #     #             + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * S[1] * (
            #     #                         1 - self._alpha) * m_xi[j, t]
            #     #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * self._beta * S[2] * (
            #     #                         1 - m_delta[j, t])
            #     #     )
            #     #     for j, t in index_line_period
            #     # )
            #     # +gp.quicksum(
            #     #     lambda_6[t]*(
            #     #         -N_bar[1]
            #     #         +m_delta.prod(N_hat,'*',t)
            #     #     )
            #     #     for t in range(1,self._period+1)
            #     # )+gp.quicksum(
            #     #     lambda_7[t]*(
            #     #         -N_bar[2]
            #     #         +sum(N_hat[item1,item2] for item1,item2 in N_hat.keys() if item2==t)
            #     #         -m_delta.prod(N_hat,'*',t)
            #     #     )
            #     #     for t in range(1,self._period+1)
            #     # )
            #     <=0
            # )
            m.addConstr(
                gp.quicksum(
                    lambda_0[j, t] * (
                        m_q[j, t] * H[j,t] - self._eta * (S[2] - S[1])
                    )
                    for j, t in index_line_period
                ) +
                gp.quicksum(
                    lambda_1[j, t] * (
                        H[j,t]
                        -S[1]/self._peak_point_demand[j-1][t-1]*m_delta[j,t]
                        -S[2]/self._peak_point_demand[j-1][t-1]*(1-m_delta[j,t])
                    )
                    for j, t in index_line_period
                ) +
                gp.quicksum(
                    lambda_2[j, t] * (
                        H[j, t] * H[j, t] * (
                            self._v_w * self._demand[j - 1][t - 1]
                            + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                            self._distance[j - 1] * m_zeta[j, t]
                            )
                            - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                    self._gammar * (1 - (1 - self._alpha) * m_xi[j, t])
                                    + self._beta * (m_delta[j, t] - (1 - self._alpha) * m_xi[j, t]) * S[1]
                                    + self._beta * (1 - m_delta[j, t]) * S[2]
                            )
                    )
                    for j, t in index_line_period
                )
                +
                # gp.quicksum(
                #     lambda_5[j, t] * (
                #             m_N_hat[j, t] * H[j, t]
                #             - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1]*m_xi[j,t]
                #             - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * S[1] * m_zeta[j, t]
                #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                #     )
                #     for j, t in index_line_period
                # )
                gp.quicksum(
                    lambda_5[j, t] * (
                            m_N_hat[j, t] * H[j, t]
                            - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                            - 2 * self._t_u * H[j,t] * m_zeta[j, t]
                            - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                            -1e-3
                    )
                    for j, t in index_line_period
                )
                +gp.quicksum(
                    lambda_6[j, t] * (
                            -1e-3
                            -m_N_hat[j, t] * H[j, t]
                            + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                            + 2 * self._t_u * H[j, t] * m_zeta[j, t]
                            + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                        )
                        for j, t in index_line_period
                    )
                <= -1e-4
            )
        '''
        add0_constr=[cons for cons in m.getConstrs() if 'add0' in cons.ConstrName]
        add1_constr = [cons for cons in m.getConstrs() if 'add1' in cons.ConstrName]
        add2_constr = [cons for cons in m.getConstrs() if 'add2' in cons.ConstrName]
        add5_constr = [cons for cons in m.getConstrs() if 'add5' in cons.ConstrName]
        if not add0_constr:
            m.addConstrs(
                (
                    m_q[j, t] * H[j, t] - self._eta * (S[2] - S[1])<=0
                    for j,t in index_line_period
                ),name='add0'
            )
        else:
            m.remove(add0_constr)
            m.addConstrs(
                (
                    m_q[j, t] * H[j, t] - self._eta * (S[2] - S[1]) <= 0
                    for j, t in index_line_period
                ),name='add0'
            )
        if not add1_constr:
            m.addConstrs(
                (
                    H[j, t]
                    - S[1] / self._peak_point_demand[j - 1][t - 1] * m_delta[j, t]
                    - S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - m_delta[j, t])<=0
                    for j,t in index_line_period
                ),name='add1'
            )
        else:
            m.remove(add1_constr)
            m.addConstrs(
                (
                    H[j, t]
                    - S[1] / self._peak_point_demand[j - 1][t - 1] * m_delta[j, t]
                    - S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - m_delta[j, t])<=0
                    for j, t in index_line_period
                ), name='add1'
            )
        if not add2_constr:
            m.addConstrs(
                (
                    H[j, t] * H[j, t] * (
                    self._v_w * self._demand[j - 1][t - 1]
                    + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                    self._distance[j - 1] * m_zeta[j, t]
                )
                    - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                    self._gammar * (1 - (1 - self._alpha) * m_xi[j, t])
                    + self._beta * (m_delta[j, t] - (1 - self._alpha) * m_xi[j, t]) * S[1]
                    + self._beta * (1 - m_delta[j, t]) * S[2]
                    )<=0
                    for j,t in index_line_period
                ),name='add2'
            )
        else:
            m.remove(add2_constr)
            m.addConstrs(
                (
                    H[j, t] * H[j, t] * (
                            self._v_w * self._demand[j - 1][t - 1]
                            + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                            self._distance[j - 1] * m_zeta[j, t]
                    )
                    - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                            self._gammar * (1 - (1 - self._alpha) * m_xi[j, t])
                            + self._beta * (m_delta[j, t] - (1 - self._alpha) * m_xi[j, t]) * S[1]
                            + self._beta * (1 - m_delta[j, t]) * S[2]
                    ) <= 0
                    for j, t in index_line_period
                ), name='add2'
            )
        if not add5_constr:
            m.addConstrs(
                (
                    -m_N_hat[j, t] * H[j, t]
                    + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                    + 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * S[1] * m_zeta[j, t]
                    + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])<=-1e-4
                    for j,t in index_line_period
                ),name='add5'
            )
        else:
            m.remove(add5_constr)
            m.addConstrs(
                (
                    -m_N_hat[j, t] * H[j, t]
                    + 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                    + 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * S[1] * m_zeta[j, t]
                    + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t]) <= -1e-4
                    for j, t in index_line_period
                ), name='add5'
            )
        '''
        # if not add_constr:
        #     m.addConstrs(
        #         (
        #             m_N_hat[j, t] * H[j, t]
        #             - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
        #             - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * S[1] * m_zeta[j, t]
        #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])==0
        #             for j,t in index_line_period
        #          ),name='add'
        #     )
        # else:
        #     m.remove(add_constr)
        #     m.addConstrs(
        #         (
        #             m_N_hat[j, t] * H[j, t]
        #             - 2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
        #             - 2 * self._t_u / self._peak_point_demand[j - 1][t - 1] * S[1] * m_zeta[j, t]
        #             - 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t]) == 0
        #             for j, t in index_line_period
        #         ), name='add'
        #     )

        # add1_constr = [cons for cons in m.getConstrs() if 'add1' in cons.ConstrName]
        # add2_constr = [cons for cons in m.getConstrs() if 'add2' in cons.ConstrName]
        # add3_constr = [cons for cons in m.getConstrs() if 'add3' in cons.ConstrName]
        # if not add1_constr:
        #     m.addConstrs((m_N_hat[j, t] == 2 * self._distance[j - 1] / (v_hat[j, t] * headway[j, t]) * m_xi[j, t]
        #                   + 2 * self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (1 - m_xi[j, t])
        #                   for j, t in index_line_period), name='add1')
        # else:
        #     m.remove(add1_constr)
        #     m.addConstrs((m_N_hat[j, t] == 2 * self._distance[j - 1] / (v_hat[j, t] * headway[j, t]) * m_xi[j, t]
        #                   + 2 * self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (1 - m_xi[j, t])
        #                   for j, t in index_line_period), name='add1')
        # if not add2_constr:
        #     m.addConstrs((m_N_bar[1] >= gp.quicksum(
        #             2 * self._distance[j - 1] / (v_hat[j, t] * headway[j, t]) * m_xi[j, t]
        #             + 2 * self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (m_delta[j, t] - m_xi[j, t])
        #             for j in range(1, self._routeNo + 1)) for t in range(1, self._period + 1)), name='add2')
        # else:
        #     m.remove(add2_constr)
        #     m.addConstrs((m_N_bar[1] >= gp.quicksum(
        #             2 * self._distance[j - 1] / (v_hat[j, t] * headway[j, t]) * m_xi[j, t]
        #             + 2 * self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (m_delta[j, t] - m_xi[j, t])
        #             for j in range(1, self._routeNo + 1)) for t in range(1, self._period + 1)), name='add2')
        # if not add3_constr:
        #     m.addConstrs((m_N_bar[2] >= gp.quicksum(
        #             2 * self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (1 - m_delta[j, t])
        #             for j in range(1, self._routeNo + 1)) for t in range(1, self._period + 1)), name='add3')
        # else:
        #     m.remove(add3_constr)
        #     m.addConstrs((m_N_bar[2] >= gp.quicksum(
        #             2 * self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (1 - m_delta[j, t])
        #             for j in range(1, self._routeNo + 1)) for t in range(1, self._period + 1)), name='add3')

        m.update()
        m.write('out1.lp')
        # N_hat: N_j_t
        # N_bar: N_s
        # q: q_j_t
        # X: X_j_t
        # delta: delta_j_t
        # xi: xi_j_t
        # zeta: zeta_j_t

        try:
            m.optimize()
            logger.info('Master Problem status: {}'.format(m.status))
            logger.info('The objective value of Master problem is %s'%(m.objVal))
            if m.status==GRB.OPTIMAL:
                y_dict={}
                y_dict['y_0']=m.objVal
                y_dict['N_hat']=dict(m.getAttr('x',m_N_hat))
                y_dict['N_tilde']=dict(m.getAttr('x',m_N_tilde))
                y_dict['N_bar']=dict(m.getAttr('x',m_N_bar))
                y_dict['q']=dict(m.getAttr('x',m_q))
                y_dict['X']=dict(m.getAttr('x',m_X))
                y_dict['delta']=dict(m.getAttr('x',m_delta))
                y_dict['xi']=dict(m.getAttr('x',m_xi))
                y_dict['zeta']=dict(m.getAttr('x',m_zeta))
                y_dict['S']=S
                y_dict['H']=H
                return y_dict
            else:
                y_dict = {}
                y_dict['y_0'] = m.objVal
                y_dict['N_hat'] = dict(m.getAttr('x', m_N_hat))
                y_dict['N_tilde'] = dict(m.getAttr('x', m_N_tilde))
                y_dict['N_bar'] = dict(m.getAttr('x', m_N_bar))
                y_dict['q'] = dict(m.getAttr('x', m_q))
                y_dict['X'] = dict(m.getAttr('x', m_X))
                y_dict['delta'] = dict(m.getAttr('x', m_delta))
                y_dict['xi'] = dict(m.getAttr('x', m_xi))
                y_dict['zeta'] = dict(m.getAttr('x', m_zeta))
                y_dict['S']=S
                y_dict['H']=H
                return y_dict
        except gp.GurobiError as e:
            logger.exception('Error'+str(e.errno))
            print('Error code'+str(e.errno)+':'+str(e))
        except AttributeError:
            logger.info('Encountered an attribute error')
            print('Encountered an attribute error')

    def solveBenders(self,epsilon,y_initial,maxiter):
        '''
        :param epsilon:
        :param y_initial:(N_hat,N_bar,...)
        :param maxiter:
        :return:
        '''
        UB_LB_tol_dict={}
        UB=float('inf')
        LB=-float('inf')
        tol=float('inf')

        iter=0

        y=y_initial
        m=self.SetupMasterProblemModel()
        with open(self._path,'wb') as f:
            #result_s = self.SubProblem(y, [], [])
            while epsilon<tol and iter<maxiter:
                result_s=self.SubProblem(y)

                # logger.info("S is \n %s"%(result_s['S']))
                # logger.info('headway is \n %s'%(result_s['headway']))
                # logger.info('headway_1 is \n %s' % (result_s['h_1']))
                # logger.info('headway_2 is \n %s' % (result_s['h_2']))
                # logger.info('v_hat is \n %s'%(result_s['v_hat']))
                # logger.info('N_hat is \n %s' % (result_s['N_hat']))
                # logger.info('N_bar is \n %s' % (result_s['N_bar']))

                pickle.dump(result_s,f)
                ob=result_s['objval']
                UB=min(UB,ob)

                y=self.solveMaster(m,result_s)
                # print(result_s['S'])
                # print(result_s['u_0'])
                # print(result_s['u_1'])
                # print(result_s['u_2'])
                # print(result_s['u_5'])
                print(result_s['S'])
                print(result_s['headway'])
                #print(result_s['u_1'])
                #print(result_s['u_2'])
                #print(result_s['u_5'])
                # print(result_s['u_6'])
                # print(result_s['u_7'])
                print(result_s['h_1'])
                #print(result_s['h_2'])
                print(result_s['v_hat'])

                #print(result_s['u_2'])
                print(y['N_hat'])
                print(y['N_bar'])
                print(y['X'])
                print(y['delta'])
                print(y['q'])
                # print(y['N_bar'])
                # print(y['q'])
                # print(y['X'])
                # print(y['delta'])
                # print(y['xi'])
                # print(y['zeta'])

                # N_hat: N_j_t
                # N_bar: N_s
                # q: q_j_t
                # X: X_j_t
                # delta: delta_j_t
                # xi: xi_j_t
                # zeta: zeta_j_t
                pickle.dump(y,f)

                obj=y['y_0']
                LB=max(LB,obj)

                tol=UB-LB
                logger.info("tol: %s"%(tol))
                logger.info("y[q]: %s"%(y['q']))
                iter+=1
                UB_LB_tol_dict[iter]=(UB,LB,tol)
                #result_s = self.SubProblem(y,result_s['S'],result_s['h_2'])
                S = result_s['S']
                H = result_s['headway']
                if result_s['status']==1:
                    u_0 = result_s['u_0']
                    u_1 = result_s['u_1']
                    u_2 = result_s['u_2']
                    u_5 = result_s['u_5']
                    u_6 = result_s['u_6']
                else:
                    u_0 = result_s['lambda_0']
                    u_1 = result_s['lambda_1']
                    u_2 = result_s['lambda_2']
                    u_5 = result_s['lambda_5']
                    u_6 = result_s['lambda_6']
                m_N_hat = y['N_hat']
                m_N_tilde = y['N_tilde']
                m_N_bar = y['N_bar']
                m_q = y['q']
                m_X = y['X']
                m_delta = y['delta']
                m_xi = y['xi']
                m_zeta = y['zeta']
                print(result_s['status'])
                print('u_0:',u_0)
                print('u_1:', u_1)
                print('u_2:', u_2)
                print('u_5:', u_5)
                print('u_6:', u_6)
                print('constraint 0')
                sum_zc=0
                for j in range(1,self._routeNo+1):
                    for t in range(1,self._period+1):
                        zc=m_q[j, t] * H[j,t] - self._eta * (S[2] - S[1])
                        zc_zc=u_0[j,t]*zc
                        sum_zc=sum_zc+zc_zc
                        print('({},{}): {},{}'.format(j,t,zc,zc_zc))
                print('sum_zc:',sum_zc)
                print('constraint 1')
                for j in range(1,self._routeNo+1):
                    for t in range(1,self._period+1):
                        zc=H[j,t]-S[1] / self._peak_point_demand[j - 1][t - 1] * m_delta[j, t]- S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - m_delta[j, t])
                        zc_zc = u_1[j, t] * zc
                        sum_zc = sum_zc + zc_zc
                        print('({},{}): {},{}'.format(j,t,zc,zc_zc))
                print('sum_zc:',sum_zc)
                print('constraint 2')
                for j in range(1,self._routeNo+1):
                    for t in range(1,self._period+1):
                        zc=H[j, t] * H[j, t] * (
                            self._v_w * self._demand[j - 1][t - 1]
                            + 2 * self._v_v * self._t_u * self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                            self._distance[j - 1] * m_zeta[j, t]
                    )- 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (
                                    self._gammar * (1 - (1 - self._alpha) * m_xi[j, t])
                                    + self._beta * (m_delta[j, t] - (1 - self._alpha) * m_xi[j, t]) * S[1]
                                    + self._beta * (1 - m_delta[j, t]) * S[2]
                            )
                        zc_zc = u_2[j, t] * zc
                        sum_zc = sum_zc + zc_zc
                        print('({},{}): {},{}'.format(j,t,zc,zc_zc))
                print('sum_zc:',sum_zc)
                print('constraint 5')
                for j in range(1,self._routeNo+1):
                    for t in range(1,self._period+1):
                        zc=m_N_hat[j,t]*H[j,t]-(
                            2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*m_xi[j,t]
                            +2*self._t_u*m_zeta[j,t]*H[j,t]
                            +2*self._distance[j-1]/self._speed[j-1][t-1]*(1-m_xi[j,t])
                        )-1e-3
                        zc_zc = u_5[j, t] * zc
                        sum_zc = sum_zc + zc_zc
                        print('({},{}): {},{}'.format(j,t,zc,zc_zc))
                print('sum_zc',sum_zc)
                print('constraint 6')
                for j in range(1,self._routeNo+1):
                    for t in range(1,self._period+1):
                        zc=-1e-3-m_N_hat[j,t]*H[j,t]+(
                            2*self._alpha*self._distance[j-1]/self._speed[j-1][t-1]*m_xi[j,t]
                            +2*self._t_u*m_zeta[j,t]*H[j,t]
                            +2*self._distance[j-1]/self._speed[j-1][t-1]*(1-m_xi[j,t])
                        )
                        zc_zc = u_6[j, t] * zc
                        sum_zc = sum_zc + zc_zc
                        print('({},{}): {},{}'.format(j,t,zc,zc_zc))
                print('sum_zc',sum_zc)

                # print("okokokokokoko")
                # sum_zc = 0
                # for j in range(1, self._routeNo + 1):
                #     for t in range(1, self._period + 1):
                #         sum_zc = sum_zc + (m_xi[j, t] * (2 * (self._alpha - 1) * self._distance[j - 1] * (
                #                 self._gammar + self._beta * S[1]) * self._peak_point_demand[j - 1][t - 1] / (
                #                                                  self._speed[j - 1][t - 1] * S[1]) + 2 * (
                #                                                  self._alpha - 1) * self._v_v * self._demand[j - 1][
                #                                              t - 1] *
                #                                          self._average_distance[j - 1] / self._speed[j - 1][t - 1]) +
                #                            m_delta[j, t] * (2 * self._distance[j - 1] * self._peak_point_demand[j - 1][
                #                     t - 1] * self._gammar * (S[2] - S[1]) / (
                #                                                     self._speed[j - 1][t - 1] * S[1] * S[
                #                                                 2]) + self._v_w *
                #                                             self._demand[j - 1][t - 1] * (S[1] - S[2]) /
                #                                             self._peak_point_demand[j - 1][t - 1]) +
                #                            m_zeta[j, t] * (2 * (
                #                             self._gammar + self._beta * S[1]) * self._t_u + 2 * self._v_v *
                #                                            self._demand[j - 1][t - 1] * self._average_distance[
                #                                                j - 1] * self._t_u * S[
                #                                                1] / (self._distance[j - 1] *
                #                                                      self._peak_point_demand[j - 1][t - 1])) +
                #                            2 * self._distance[j - 1] * (self._gammar + self._beta * S[2]) *
                #                            self._peak_point_demand[j - 1][t - 1] / (
                #                                        self._speed[j - 1][t - 1] * S[2]) + self._v_w *
                #                            self._demand[j - 1][t - 1] * S[2] / self._peak_point_demand[j - 1][
                #                                t - 1] + 2 * self._v_v *
                #                            self._demand[j - 1][t - 1] * self._average_distance[j - 1] /
                #                            self._speed[j - 1][t - 1])
                #
                #         sum_zc = sum_zc + (
                #                 u_0[j, t] * (
                #                 m_q[j, t] * S[1] - self._eta * (S[2] - S[1]) * self._peak_point_demand[j - 1][t - 1])
                #         )
                #
                #         sum_zc = sum_zc + (
                #                 u_1[j, t] * (
                #                 +2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                #                 + 2 * self._t_u * S[1] / self._peak_point_demand[j - 1][t - 1] * m_zeta[j, t]
                #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                #                 - S[1] / self._peak_point_demand[j - 1][t - 1] * m_N_tilde[j, t]
                #                 - S[2] / self._peak_point_demand[j - 1][t - 1] * m_N_hat[j, t]
                #                 + S[2] / self._peak_point_demand[j - 1][t - 1] * m_N_tilde[j, t]
                #         )
                #         )
                #
                #         sum_zc = sum_zc + (
                #                 u_2[j, t] * (
                #                 +2 * self._alpha * self._distance[j - 1] / self._speed[j - 1][t - 1] * m_xi[j, t]
                #                 + 2 * self._t_u * S[1] / self._peak_point_demand[j - 1][t - 1] * m_zeta[j, t]
                #                 + 2 * self._distance[j - 1] / self._speed[j - 1][t - 1] * (1 - m_xi[j, t])
                #                 - h_2[j, t] * m_N_hat[j, t]
                #         )
                #         )
                #
                #         sum_zc = sum_zc + (
                #                 u_5[j, t] * (
                #                 -S[1] / self._peak_point_demand[j - 1][t - 1] * m_delta[j, t]
                #                 - S[2] / self._peak_point_demand[j - 1][t - 1] * (1 - m_delta[j, t])
                #                 + 0.05
                #         )
                #         )
                # sum_zc = sum_zc + m_N_bar[1] * (self._c + self._e * S[1]) * self._recovery / 365 + m_N_bar[2] * (
                #             self._c + self._e * S[2]) * self._recovery / 365
                # sum_zc = sum_zc + ((sum(self._d_j) - sum(
                #     m_q[j, t] for j in range(1, self._routeNo + 1) for t in range(1, self._period + 1))) * self._v_p)
                # print(sum_zc)
            pickle.dump(UB_LB_tol_dict,f)


        return result_s,y,UB_LB_tol_dict

