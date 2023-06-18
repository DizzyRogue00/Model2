import gurobipy as gp
from gurobipy impot *
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

        self._t_u=1.0/20   #hr/parcel
        self._alpha=1.2 #[0.6,0.8,1,1.2,1.5]

        self._c=16000   #$/veh
        self._e=2400    #$/veh
        self._recovery=0.1359

        self._m_j=np.random.randint(2,4,routeNo)
        self._d_i,self._d_j=self.generate_freight_demand()

        self._eta=0.5
        self._v_p=1500#the weight of one parcel is 0.5 tons.

        self._demand=demand
        self._peak_point_demand=peak_point_demand

        self._distance=distance
        self._average_distance=average_distance

        self._speed=speed
        self._period = len(speed[0])
        self._routeNo=routeNo

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
            d_i[row, :item] = np.random.randint(1, 30, item)
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

        S=m1.addVars(range(1,3),name='S')
        S_inverse=m1.addVars(range(1,3),name='S_inverse')
        h_2=m1.addVars(index_line_period,name='h_2')
        u_0=m1.addVars(index_line_period,name='u_0')
        u_1=m1.addVars(index_line_period,name='u_1')
        u_2=m1.addVars(index_line_period,name='u_2')
        u_3=m1.addVar(name='u_3')

        #2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
        #2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
        #y['delta'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]
        #(1-y['delta'][j, t])/ self._peak_point_demand[j - 1][t - 1] * S[2]
        #self._v_w*self._demand[j-1][t-1]+2*self._alpha*self._v_v*self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._peak_point_demand[j-1][t-1]*self._average_distance[j-1]*y['X'][j,t]*y['delta'][j,t]/(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._speed[j-1][t-1])
        # self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])
        # self._beta*y['delta'][j,t]*(1-(1-self._alpha)*y['X'][j,t])*S[1]
        # self._beta*(1-y['delta'][j,t])*S[2]
        #(2*self._distance[j-1]*(self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])+self._beta*y['delta'][j,t]*(1-(1-self._alpha)*y['X'][j,t])*S[1]+self._beta*(1-y['delta'][j,t])*S[2]))/self._speed[j-1][t-1]

        m1.addConstrs((S[s]*S_inverse[s]==1 for s in range(1,3)),name='aux_0')
        m1.addConstrs(((self._v_w*self._demand[j-1][t-1]+2*self._alpha*self._v_v*self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._peak_point_demand[j-1][t-1]*self._average_distance[j-1]*y['X'][j,t]*y['delta'][j,t]/(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._speed[j-1][t-1]))*h_2[j,t]*h_2[j,t]==(2*self._distance[j-1]*(self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])+self._beta*y['delta'][j,t]*(1-(1-self._alpha)*y['X'][j,t])*S[1]+self._beta*(1-y['delta'][j,t])*S[2]))/self._speed[j-1][t-1] for j,t in index_line_period),name='aux_1')
        m1.addConstrs((y['q'][j, t] * S[1] - self._eta * (S[2] - S[1]) * self._peak_point_demand[j-1][t-1] <= 0 for j,t in index_line_period),name='sub_0')
        m1.addConstrs((2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
                       -y['delta'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]
                       -(1-y['delta'][j, t])/ self._peak_point_demand[j - 1][t - 1] * S[2] for j,t in index_line_period),name='sub_1')
        m1.addConstrs((2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
                       -h_2[j,t] for j,t in index_line_period),name='sub_2')
        m1.addConstr(S[1] - S[2] + 0.5 <= 0,name='sub_3')
        m1.addConstr(gp.quicksum(
            u_0[j,t]*(y['q'][j, t] * S[1] - self._eta * (S[2] - S[1]) * self._peak_point_demand[j-1][t-1])+
            u_1[j,t]*(2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
                       -y['delta'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]
                       -(1-y['delta'][j, t])/ self._peak_point_demand[j - 1][t - 1] * S[2])+
            u_2[j,t]*(2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
                       -h_2[j,t]) for j,t in index_line_period
        )+u_3*(S[1]-S[2]+0.5)>=-1e-4,name='sub_4')

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
        obj=obj+gp.quicksum(
            u_0[j,t]*(y['q'][j, t] * S[1] - self._eta * (S[2] - S[1]) * self._peak_point_demand[j-1][t-1])+
            u_1[j,t]*(2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
                       -y['delta'][j,t]/self._peak_point_demand[j-1][t-1]*S[1]
                       -(1-y['delta'][j, t])/ self._peak_point_demand[j - 1][t - 1] * S[2])+
            u_2[j,t]*(2*self._distance[j-1]*y['X'][j,t]*y['delta'][j,t]/y['N_hat'][j,t]*(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]+self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*S[1])/(self._speed[j-1][t-1]*self._distance[j-1]*self._peak_point_demand[j-1][t-1])
                       +2*self._distance[j-1]*(1-y['X'][j,t]*y['delta'][j,t])/(self._speed[j-1][t-1]*y['N_hat'][j,t])
                       -h_2[j,t]) for j,t in index_line_period
        )+u_3*(S[1]-S[2]+0.5)

        m1.setObjective(obj,gp.GRB.MINIMIZE)
        m1.update()

        print(m1.status)
        logger.info('Sub Problem status: {}'.format(m1.status))
        result_dict = {}
        if m1.status=GRB.OPTIMAL:
            result_dict['objval']=m1.objVal
            result_dict['S']=m1.getAttr('x',S)
            result_dict['u_0']=m1.getAttr('x',u_0)
            result_dict['u_1'] = m1.getAttr('x', u_1)
            result_dict['u_2'] = m1.getAttr('x', u_2)
            result_dict['u_3'] = m1.getAttr('x', u_3)
            result_dict['v_hat']={(j,t):self._speed[j-1][t-1] * self._distance[j-1] * self._peak_point_demand[j-1][t-1] / (
                        self._alpha * self._distance[j-1] * self._peak_point_demand[j-1][t-1] + self._t_u * y['q'][j, t] *
                        self._speed[j-1][t-1] * result_dict['S'][1]) for j,t in index_line_period}
            h1_hat={(j,t):result_dict['S'][1]*y['delta'][j,t]/self._peak_point_demand[j-1][t-1]+result_dict['S'][2]*(1-y['delta'][j,t])/self._peak_point_demand[j-1][t-1] for j,t in index_line_period}
            h2_hat={(j,t):np.sqrt(1/((self._v_w*self._demand[j-1][t-1]+2*self._alpha*self._v_v*self._t_u*y['q'][j,t]*self._speed[j-1][t-1]*self._demand[j-1][t-1]*self._peak_point_demand[j-1][t-1]*self._average_distance[j-1]*y['X'][j,t]*y['delta'][j,t]/(self._alpha*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._speed[j-1][t-1]))*self._speed[j-1][t-1]/(2*self._distance[j-1]*(self._gammar*(1-(1-self._alpha)*y['X'][j,t]*y['delta'][j,t])+self._beta*y['delta'][j,t]*result_dict['S'][1]*(1-(1-self._alpha)*y['X'][j,t])+self._beta*(1-y['delta'][j,t])*result_dict['S'][2]))))
                for j,t in index_line_period
                    }
            result_dict['headway']={key:np.min((h1_hat[key],h2_hat[key])) for key in index_line_period}
        elif m1.status == GRB.TIME_LIMIT:
            result_dict['objval'] = m1.objVal
            result_dict['S'] = m1.getAttr('x', S)
            result_dict['u_0'] = m1.getAttr('x', u_0)
            result_dict['u_1'] = m1.getAttr('x', u_1)
            result_dict['u_2'] = m1.getAttr('x', u_2)
            result_dict['u_3'] = m1.getAttr('x', u_3)
            result_dict['v_hat'] = {
                (j, t): self._speed[j - 1][t - 1] * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] / (
                        self._alpha * self._distance[j - 1] * self._peak_point_demand[j - 1][t - 1] + self._t_u *
                        y['q'][j, t] *
                        self._speed[j - 1][t - 1] * result_dict['S'][1]) for j, t in index_line_period}
            h1_hat = {(j, t): result_dict['S'][1] * y['delta'][j, t] / self._peak_point_demand[j - 1][t - 1] +
                              result_dict['S'][2] * (1 - y['delta'][j, t]) / self._peak_point_demand[j - 1][t - 1] for
                      j, t in index_line_period}
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
            h2_hat=m1.getAttr('x',h_2)
            result_dict['headway'] = {key: np.min((h1_hat[key], h2_hat[key])) for key in index_line_period}
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
        N_hat=m.addVars(index_line_period,name='N_hat')
        N_bar=m.addVars(range(1,3),name='N_bar')
        q=m.addVars(index_line_period,name='q')
        X=m.addVars(index_line_period,vtype=GRB.BINARY,name='X')
        delta=m.addVars(index_line_period,vtype=GRB.BINARY,name='delta')
        xi=m.addVars(index_line_period,vtype=GRB.BINARY,name='xi')
        zeta=m.addVars(index_line_period,name='zeta')

        m.addConstrs((xi[item]<=delta[item] for item in index_line_period),name='c_1')
        m.addConstrs((xi[item]<=X[item] for item in index_line_period),name='c_2')
        m.addConstrs((xi[item]>=X[item]+delta[item]-1 for item in index_line_period),name='c_3')
        m.addConstrs((delta[item]+X[item]<=2 for item in index_line_period),name='c_4')
        m.addConstrs((delta[item]>=X[item] for item in index_line_period),name='c_5')
        m.addConstr(gp.quicksum(delta[item] for item in index_line_period)>0,name='c_6')
        m.addConstrs((q.sum(j,'*')<=self._d_j[j-1] for j in range(1,self._routeNo+1)),name='c_7')
        m.addConstrs((q[j,t]>=0 for j,t in index_line_period),name='c_8')
        m.addConstrs((q[j,t]<=self._d_j[j-1] for j,t in index_line_period),name='c_9')
        m.addConstrs((q[j,t]>(X[j,t]-1)*self._d_j[j-1] for j,t in index_line_period),name='c_10')
        m.addConstrs((q[j,t]<=X[j,t]*self._d_j[j-1] for j,t in index_line_period),name='c_11')
        m.addConstrs((zeta[j,t]-xi[j,t]*self._d_j[j-1]<=0 for j,t in index_line_period),name='c_12')
        m.addConstrs((zeta[j,t]>=0 for j,t in index_line_period),name='c_13')
        m.addConstrs((zeta[j,t]-q[j,t]<=0 for j,t in index_line_period), name='c_14')
        m.addConstrs((zeta[j,t]-q[j,t]+self._d_j[j-1]-xi[j,t]*self._d_j[j-1]>=0 for j,t in index_line_period), name=
                     'c_15')
        m.addConstrs((N_hat[j,t]>0 for j,t in index_line_period),name='c_16')
        m.addConstrs((N_bar[s]>0 for s in range(1,3)),name='c_17')

        m.setObjective(y_0,sense=GRB.MINIMIZE)
        m.Params.lazyConstraints=1
        m.update()
        return m

    def solveMaster(self,m,sub_result_dict):
        index_line_period = gp.tuplelist(
            [(line, time) for line in range(1, self._routeNo + 1) for time in range(1, self._period + 1)])
        S=sub_result_dict['S']#S[s]
        u_0=sub_result_dict['u_0']#u_0[j,t]
        u_1 = sub_result_dict['u_1']  # u_1[j,t]
        u_2 = sub_result_dict['u_2']  # u_2[j,t]
        u_3 = sub_result_dict['u_3']  # u_3
        v_hat=sub_result_dict['v_hat']#v_hat[j,t]
        headway=sub_result_dict['headway']#headway[j,t]

        m_y_0=m.getVarByName('y_0')
        m_N_hat=gp.tupledict({(j,t):m.getVarByName('N_hat['+str(j)+','+str(t)+']') for j,t in index_line_period})
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

        m.addConstr(m_y_0>=
                    gp.quicksum(
                        m_xi[j,t]*(2*(self._alpha-1)*self._distance[j-1]*(self._gammar+self._beta*S[1])*self._peak_point_demand[j-1][t-1]/(self._speed[j-1][t-1]*S[1])+2*(self._alpha-1)*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]/self._speed[j-1][t-1])+
                        m_delta[j,t]*(2*self._distance[j-1]*self._peak_point_demand[j-1][t-1]*self._gammar*(S[2]-S[1])/(self._speed[j-1][t-1]*S[1]*S[2])+self._v_w*self._demand[j-1][t-1]*(S[1]-S[2])/self._peak_point_demand[j-1][t-1])+
                        m_zeta[j,t]*(2*(self._gammar+self._beta*S[1])*self._t_u+2*self._v_v*self._demand[j-1][t-1]*self._average_distance[j-1]*self._t_u*S[1]/(self._distance[j-1]*self._peak_point_demand[j-1][t-1]))+
                        2 * self._distance[j - 1] * (self._gammar + self._beta * S[2]) * self._peak_point_demand[j - 1][t - 1] / (self._speed[j - 1][t - 1] * S[2]) + self._v_w * self._demand[j - 1][t - 1] * S[2] / self._peak_point_demand[j - 1][t - 1] + 2 * self._v_v * self._demand[j - 1][t - 1] *self._average_distance[j - 1] / self._speed[j - 1][t - 1]
                    for j,t in index_line_period
                    )+
                    gp.quicksum(
                        m_N_bar[s] * (self._c + self._e * S[s]) * self._recovery / 365
                        for s in range(1,3)
                    )+
                    gp.quicksum(
                        u_0[j,t]*(m_q[j,t]*S[1]-self._eta*(S[2]-S[1])*self._peak_point_demand[j-1][t-1])
                        for j,t in index_line_period
                    )+
                    gp.quicksum(
                        u_1[j,t]*(2*self._distance[j-1]/v_hat[j,t]*m_xi[j,t]+2*self._distance[j-1]/self._speed[j-1][t-1]*(1-m_xi[j,t])-headway[j,t]*m_N_hat[j,t])
                        for j,t in index_line_period
                    )+
                    gp.quicksum(
                        u_2[j,t]*(2*self._distance[j-1]/v_hat[j,t]*m_xi[j,t]+2*self._distance[j-1]/self._speed[j-1][t-1]*(1-m_xi[j,t])-headway[j,t]*m_N_hat[j,t])
                        for j,t in index_line_period
                    )+
                    u_3*(S[1]-S[2]+0.5)+
                    (gp.quicksum(self._d_j)-gp.quicksum(m_q[j,t] for j,t in index_line_period))*self._v_p
        )
        add1_constr=[cons for cons in m.getConstrs() if 'add1' in cons.ConstrName]
        add2_constr = [cons for cons in m.getConstrs() if 'add2' in cons.ConstrName]
        add3_constr = [cons for cons in m.getConstrs() if 'add3' in cons.ConstrName]
        if not add1_constr:
            m.addConstrs((m_N_hat[j, t] == 2 * self._distance[j - 1] / (v_hat[j, t] * headway[j, t]) * m_xi[j, t] + 2 *self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (1 - m_xi[j, t]) for j, t in index_line_period), name='add1')
        else:
            m.remove(add1_constr)
            m.addConstrs((m_N_hat[j,t]==2*self._distance[j-1]/(v_hat[j,t]*headway[j,t])*m_xi[j,t]+2*self._distance[j-1]/(self._speed[j-1][t-1]*headway[j,t])*(1-m_xi[j,t]) for j,t in index_line_period),name='add1')
        if not add2_constr:
            m.addConstrs((m_N_bar[1]>=gp.quicksum(2*self._distance[j-1]/(v_hat[j,t]*headway[j,t])*m_xi[j,t]+2*self._distance[j-1]/(self._speed[j-1][t-1]*headway[j,t])*(m_delta[j,t]-m_xi[j,t]) for j in range(1,self._routeNo+1)) for t in range(1,self._period+1)),name='add2')
        else:
            m.remove(add2_constr)
            m.addConstrs((m_N_bar[1] >= gp.quicksum(2 * self._distance[j - 1] / (v_hat[j, t] * headway[j, t]) * m_xi[j, t] + 2 * self._distance[j - 1] / (self._speed[j - 1][t - 1] * headway[j, t]) * (m_delta[j, t] - m_xi[j, t]) for j in range(1, self._routeNo + 1)) for t in range(1, self._period + 1)), name='add2')
        if not add3_constr:
            m.addConstrs((m_N_bar[2]>=gp.quicksum(2*self._distance[j-1]/(self._speed[j-1][t-1]*headway[j,t])*(1-m_delta[j,t]) for j in range(1,self._routeNo+1)) for t in range(1,self._period+1)),name='add3')
        else:
            m.remove(add3_constr)
            m.addConstrs((m_N_bar[2]>=gp.quicksum(2*self._distance[j-1]/(self._speed[j-1][t-1]*headway[j,t])*(1-m_delta[j,t]) for j in range(1,self._routeNo+1)) for t in range(1,self._period+1)),name='add3')
        m.update()
        # N_hat: N_j_t
        # N_bar: N_s
        # q: q_j_t
        # X: X_j_t
        # delta: delta_j_t
        # xi: xi_j_t
        # zeta: zeta_j_t
        try:
            m.optimize()
            print(m.status)
            logger.info('Master Problem status: {}'.format(m.status))
            if m.status==GRB.OPTIMAL:
                y_dict={}
                y_dict['y_0']=m.objVal
                y_dict['N_hat']=m.getAttr('x',m_N_hat)
                y_dict['N_bar']=m.getAttr('x',m_N_bar)
                y_dict['q']=m.getAttr('x',m_q)
                y_dict['X']=m.getAttr('x'.m_X)
                y_dict['delta']=m.getAttr('x',m_delta)
                y_dict['xi']=m.getAttr('x',m_xi)
                y_dict['zeta']=m.getAttr('x',m_zeta)
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
            while epsilon<tol and iter<maxiter:
                result_s=self.SubProblem(y)
                pickle.dump(result_s,f)
                ob=result_s['objval']
                UB=min(UB,ob)

                y=self.solveMaster(m,result_s)
                pickle.dump(y,f)
                obj=y['objval']
                LB=max(LB,obj)

                tol=UB-LB
                iter+=1
                UB_LB_tol_dict[iter]=(UB,LB,tol)
            pickle.dump(UB_LB_tol_dict,f)

        return result_s,y,UB_LB_tol_dict

