#RouteNo=5
#distance=[16,20,16,24,20]
#peak hour:4
#off-peak hour: 14
#speed=[(40,48)] #peak hour vs. off-peak hour
#demand=[(200,100),(300,150),(150,60).(400,200),(250,125)] # peak hour vs. off-peak hour

import OneSizeSystemwide as oss
import TwosizeSystemwide as tss
import FOT as fot
import pickle
import os

import logging
import logging.config
import yaml

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

def Freight_Systemwide(routeNo,distance,average_distance,speed,demand,peak_point_demand,type):
    typeList=['OneS','TwoS','TwoSF']
    result=None
    if type not in typeList:
        logger.debug('The type is wrong, please reset it.')
    elif type==typeList[0]:
        case1 = oss.OneSize(routeNo, distance, average_distance, speed,demand, peak_point_demand)
        result1 = case1.Optimal()
        result={}
        result['objval']=result1[0]
        result['result']=result1[1]#[c_o, c_uw, c_uv, c_u, c_total, c_p]
        result['S']=result1[2]
        result['fleet_size']=result1[3][0]
        result['bus_operating_cost']=result1[4]
        result['h_jt']=dict(result1[5])
        result['h_jt_1']=dict(result1[6])
        result['h_jt_2']=dict(result1[7])
        result['c_o_jt']=dict(result1[8])
        result['c_uw_jt']=dict(result1[9])
        result['c_uv_jt']=dict(result1[10])
        result['c_u_jt']=dict(result1[11])
        result['c_jt']=dict(result1[12])
        result['n_jt']=dict(result1[13])
        result['n_t']=dict(result1[14])
        with open('data_one.pickle','wb') as f:
            pickle.dump(result,f)
        logger.info('One size systemwide finished!')
    elif type==typeList[1]:
        case2 = tss.TwoSize(routeNo, distance, average_distance, speed, demand,peak_point_demand, size_type=2)
        result = case2.BFGS()
        #one item in the result
        #iter_no, {"solution": x, "delta": delta_x, "headway": headway_x,"fleet": fleet_size_x,"objValue": ctp_x}
        with open('data_two.pickle','wb') as f:
            pickle.dump(result,f)
        logger.info('Two size systemwide finished! ')
    elif type==typeList[2]:
        case3=fot.FOT(routeNo, distance, average_distance, speed, demand, peak_point_demand)
        #generate y_initial
        # if os.path.exists('data_two.pickle'):
        #     with open('data_two.pickle','rb') as f:
        #         result_init = pickle.load(f)
        #         result_init=result_init[len(result_init)-1]
        # else:
        #     result_init = Freight_Systemwide(routeNo, distance, average_distance, speed, demand, peak_point_demand, 'TwoS')
        #     result_init=result_init[len(result_init)-1]
        if os.path.exists('data_one.pickle'):
            with open('data_one.pickle','rb') as f:
                result_init = pickle.load(f)
                #result_init=result_init[len(result_init)-1]
        else:
            result_init = Freight_Systemwide(routeNo, distance, average_distance, speed, demand, peak_point_demand, 'OneS')
        period=len(demand[0])
        y_initial={}
        y_initial['y_0']=-float("inf")
        #y_initial['N_hat']=result_init['n_jt']
        y_initial['N_hat'] = {(j,t):20 for j in range(1,routeNo+1) for t in range(1,period+1)}
        #y_initial['N_bar']={1:0,2:result_init['fleet_size']}
        y_initial['N_bar'] = {1: 0, 2: 100}
        y_initial['q']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['X']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['delta']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['xi']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        y_initial['zeta']={(j,t):0 for j in range(1,routeNo+1) for t in range(1,period+1)}
        print(y_initial['N_hat'])
        print(y_initial['N_bar'])

        # y_initial['N_hat'] = {(1, 1): 4.734449891290562, (1, 2): 8.435317511990648, (1, 3): 8.435317511990648, (1, 4): 4.734449891290562, (1, 5): 4.734449891290562, (1, 6): 4.734449891290562, (1, 7): 4.734449891290562, (1, 8): 4.734449891290562, (1, 9): 4.734449891290562, (1, 10): 4.734449891290562, (1, 11): 4.734449891290562, (1, 12): 8.435317511990648, (1, 13): 8.435317511990648, (1, 14): 4.734449891290562, (1, 15): 4.734449891290562, (1, 16): 4.734449891290562, (1, 17): 4.734449891290562, (1, 18): 5.681339869548674, (2, 1): 6.590091806242694, (2, 2): 15.816220334982468, (2, 3): 15.816220334982468, (2, 4): 6.590091806242694, (2, 5): 6.590091806242694, (2, 6): 6.590091806242694, (2, 7): 6.590091806242694, (2, 8): 6.590091806242694, (2, 9): 6.590091806242694, (2, 10): 6.590091806242694, (2, 11): 7.908110167491234, (2, 12): 15.816220334982468, (2, 13): 15.816220334982468, (2, 14): 6.590091806242694, (2, 15): 6.590091806242694, (2, 16): 6.590091806242694, (2, 17): 6.590091806242694, (2, 18): 6.590091806242694, (3, 1): 3.667289116484373, (3, 2): 6.351931075795312, (3, 3): 6.351931075795312, (3, 4): 3.667289116484373, (3, 5): 3.667289116484373, (3, 6): 3.667289116484373, (3, 7): 3.667289116484373, (3, 8): 3.667289116484373, (3, 9): 3.667289116484373, (3, 10): 3.667289116484373, (3, 11): 3.667289116484373, (3, 12): 6.351931075795312, (3, 13): 6.351931075795312, (3, 14): 3.667289116484373, (3, 15): 4.400746939781247, (3, 16): 3.667289116484373, (3, 17): 3.667289116484373, (3, 18): 3.667289116484373, (4, 1): 10.544146889988312, (4, 2): 25.305952535971947, (4, 3): 25.305952535971947, (4, 4): 10.544146889988312, (4, 5): 12.652976267985972, (4, 6): 10.544146889988312, (4, 7): 10.544146889988312, (4, 8): 10.544146889988312, (4, 9): 10.544146889988312, (4, 10): 10.544146889988312, (4, 11): 10.544146889988312, (4, 12): 25.305952535971947, (4, 13): 25.305952535971947, (4, 14): 10.544146889988312, (4, 15): 10.544146889988312, (4, 16): 10.544146889988312, (4, 17): 10.544146889988312, (4, 18): 10.544146889988312, (5, 1): 5.918062364113203, (5, 2): 13.18018361248539, (5, 3): 13.18018361248539, (5, 4): 5.918062364113203, (5, 5): 5.918062364113203, (5, 6): 5.918062364113203, (5, 7): 5.918062364113203, (5, 8): 5.918062364113203, (5, 9): 5.918062364113203, (5, 10): 5.918062364113203, (5, 11): 5.918062364113203, (5, 12): 13.18018361248539, (5, 13): 13.18018361248539, (5, 14): 5.918062364113203, (5, 15): 7.1016748369358424, (5, 16): 5.918062364113203, (5, 17): 5.918062364113203, (5, 18): 5.918062364113203}
        # y_initial['N_bar'] = {1: 27.96743220027136, 2: 41.12217287095437}
        # y_initial['q'] = {(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, (1, 6): 0.0, (1, 7): 0.0, (1, 8): 0.0, (1, 9): 0.0, (1, 10): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): 0.0, (1, 15): 0.0, (1, 16): 0.0, (1, 17): 0.0, (1, 18): 58.0, (2, 1): 0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0, (2, 6): 0.0, (2, 7): 0.0, (2, 8): 0.0, (2, 9): 0.0, (2, 10): 0.0, (2, 11): 35.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): 0.0, (2, 15): 0.0, (2, 16): 0.0, (2, 17): 0.0, (2, 18): 0.0, (3, 1): 0.0, (3, 2): 0.0, (3, 3): 0.0, (3, 4): 0.0, (3, 5): 0.0, (3, 6): 0.0, (3, 7): 0.0, (3, 8): 0.0, (3, 9): 0.0, (3, 10): 0.0, (3, 11): 0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): 0.0, (3, 15): 12.0, (3, 16): 0.0, (3, 17): 0.0, (3, 18): 0.0, (4, 1): 0.0, (4, 2): 0.0, (4, 3): 0.0, (4, 4): 0.0, (4, 5): 54.0, (4, 6): 0.0, (4, 7): 0.0, (4, 8): 0.0, (4, 9): 0.0, (4, 10): 0.0, (4, 11): 0.0, (4, 12): 0.0, (4, 13): 0.0, (4, 14): 0.0, (4, 15): 0.0, (4, 16): 0.0, (4, 17): 0.0, (4, 18): 0.0, (5, 1): 0.0, (5, 2): 0.0, (5, 3): 0.0, (5, 4): 0.0, (5, 5): 0.0, (5, 6): 0.0, (5, 7): 0.0, (5, 8): 0.0, (5, 9): 0.0, (5, 10): 0.0, (5, 11): 0.0, (5, 12): 0.0, (5, 13): 0.0, (5, 14): 0.0, (5, 15): 54.0, (5, 16): 0.0, (5, 17): 0.0, (5, 18): 0.0}
        # y_initial['X'] = {(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, (1, 6): 0.0, (1, 7): 0.0, (1, 8): 0.0, (1, 9): 0.0, (1, 10): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): 0.0, (1, 15): 0.0, (1, 16): 0.0, (1, 17): 0.0, (1, 18): 1.0, (2, 1): 0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0, (2, 6): 0.0, (2, 7): 0.0, (2, 8): 0.0, (2, 9): 0.0, (2, 10): 0.0, (2, 11): 1.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): 0.0, (2, 15): 0.0, (2, 16): 0.0, (2, 17): 0.0, (2, 18): 0.0, (3, 1): 0.0, (3, 2): 0.0, (3, 3): 0.0, (3, 4): 0.0, (3, 5): 0.0, (3, 6): 0.0, (3, 7): 0.0, (3, 8): 0.0, (3, 9): 0.0, (3, 10): 0.0, (3, 11): 0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): 0.0, (3, 15): 1.0, (3, 16): 0.0, (3, 17): 0.0, (3, 18): 0.0, (4, 1): 0.0, (4, 2): 0.0, (4, 3): 0.0, (4, 4): 0.0, (4, 5): 1.0, (4, 6): 0.0, (4, 7): 0.0, (4, 8): 0.0, (4, 9): 0.0, (4, 10): 0.0, (4, 11): 0.0, (4, 12): 0.0, (4, 13): 0.0, (4, 14): 0.0, (4, 15): 0.0, (4, 16): 0.0, (4, 17): 0.0, (4, 18): 0.0, (5, 1): 0.0, (5, 2): 0.0, (5, 3): 0.0, (5, 4): 0.0, (5, 5): 0.0, (5, 6): 0.0, (5, 7): 0.0, (5, 8): 0.0, (5, 9): 0.0, (5, 10): 0.0, (5, 11): 0.0, (5, 12): 0.0, (5, 13): 0.0, (5, 14): 0.0, (5, 15): 1.0, (5, 16): 0.0, (5, 17): 0.0, (5, 18): 0.0}
        # y_initial['delta'] = {(1, 1): 1.0, (1, 2): 1.0, (1, 3): 1.0, (1, 4): 1.0, (1, 5): 1.0, (1, 6): 1.0, (1, 7): 1.0, (1, 8): 1.0, (1, 9): 1.0, (1, 10): 1.0, (1, 11): 1.0, (1, 12): 1.0, (1, 13): 1.0, (1, 14): 1.0, (1, 15): 1.0, (1, 16): 1.0, (1, 17): 1.0, (1, 18): 1.0, (2, 1): 1.0, (2, 2): -0.0, (2, 3): 0.0, (2, 4): 1.0, (2, 5): -0.0, (2, 6): 1.0, (2, 7): 1.0, (2, 8): 1.0, (2, 9): 1.0, (2, 10): 1.0, (2, 11): 1.0, (2, 12): -0.0, (2, 13): 0.0, (2, 14): 1.0, (2, 15): 1.0, (2, 16): 1.0, (2, 17): 1.0, (2, 18): 1.0, (3, 1): 1.0, (3, 2): 1.0, (3, 3): 1.0, (3, 4): 1.0, (3, 5): 1.0, (3, 6): 1.0, (3, 7): 1.0, (3, 8): 1.0, (3, 9): 1.0, (3, 10): 1.0, (3, 11): 1.0, (3, 12): 1.0, (3, 13): 1.0, (3, 14): 1.0, (3, 15): 1.0, (3, 16): 1.0, (3, 17): 1.0, (3, 18): 1.0, (4, 1): -0.0, (4, 2): -0.0, (4, 3): -0.0, (4, 4): 0.0, (4, 5): 1.0, (4, 6): 0.0, (4, 7): -0.0, (4, 8): -0.0, (4, 9): -0.0, (4, 10): 0.0, (4, 11): -0.0, (4, 12): -0.0, (4, 13): -0.0, (4, 14): 0.0, (4, 15): -0.0, (4, 16): -0.0, (4, 17): 0.0, (4, 18): -0.0, (5, 1): 1.0, (5, 2): 1.0, (5, 3): 1.0, (5, 4): 1.0, (5, 5): 1.0, (5, 6): 1.0, (5, 7): 1.0, (5, 8): 1.0, (5, 9): 1.0, (5, 10): 1.0, (5, 11): 1.0, (5, 12): 1.0, (5, 13): 1.0, (5, 14): 1.0, (5, 15): 1.0, (5, 16): 1.0, (5, 17): 1.0, (5, 18): 1.0}
        # y_initial['xi'] = {(1, 1): -0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): -0.0, (1, 5): -0.0, (1, 6): -0.0, (1, 7): -0.0, (1, 8): -0.0, (1, 9): -0.0, (1, 10): -0.0, (1, 11): -0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): -0.0, (1, 15): -0.0, (1, 16): -0.0, (1, 17): -0.0, (1, 18): 1.0, (2, 1): -0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): -0.0, (2, 5): -0.0, (2, 6): -0.0, (2, 7): -0.0, (2, 8): -0.0, (2, 9): -0.0, (2, 10): -0.0, (2, 11): 1.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): -0.0, (2, 15): -0.0, (2, 16): -0.0, (2, 17): -0.0, (2, 18): -0.0, (3, 1): -0.0, (3, 2): 0.0, (3, 3): 0.0, (3, 4): -0.0, (3, 5): -0.0, (3, 6): -0.0, (3, 7): -0.0, (3, 8): -0.0, (3, 9): -0.0, (3, 10): -0.0, (3, 11): -0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): -0.0, (3, 15): 1.0, (3, 16): -0.0, (3, 17): -0.0, (3, 18): -0.0, (4, 1): -0.0, (4, 2): 0.0, (4, 3): 0.0, (4, 4): -0.0, (4, 5): 1.0, (4, 6): -0.0, (4, 7): -0.0, (4, 8): -0.0, (4, 9): -0.0, (4, 10): -0.0, (4, 11): -0.0, (4, 12): 0.0, (4, 13): 0.0, (4, 14): -0.0, (4, 15): -0.0, (4, 16): -0.0, (4, 17): -0.0, (4, 18): -0.0, (5, 1): -0.0, (5, 2): 0.0, (5, 3): 0.0, (5, 4): -0.0, (5, 5): -0.0, (5, 6): -0.0, (5, 7): -0.0, (5, 8): -0.0, (5, 9): -0.0, (5, 10): -0.0, (5, 11): -0.0, (5, 12): 0.0, (5, 13): 0.0, (5, 14): -0.0, (5, 15): 1.0, (5, 16): -0.0, (5, 17): -0.0, (5, 18): -0.0}
        # y_initial['zeta'] = {(1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0, (1, 6): 0.0, (1, 7): 0.0, (1, 8): 0.0, (1, 9): 0.0, (1, 10): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 13): 0.0, (1, 14): 0.0, (1, 15): 0.0, (1, 16): 0.0, (1, 17): 0.0, (1, 18): 58.0, (2, 1): 0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0, (2, 6): 0.0, (2, 7): 0.0, (2, 8): 0.0, (2, 9): 0.0, (2, 10): 0.0, (2, 11): 35.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): 0.0, (2, 15): 0.0, (2, 16): 0.0, (2, 17): 0.0, (2, 18): 0.0, (3, 1): 0.0, (3, 2): 0.0, (3, 3): 0.0, (3, 4): 0.0, (3, 5): 0.0, (3, 6): 0.0, (3, 7): 0.0, (3, 8): 0.0, (3, 9): 0.0, (3, 10): 0.0, (3, 11): 0.0, (3, 12): 0.0, (3, 13): 0.0, (3, 14): 0.0, (3, 15): 12.0, (3, 16): 0.0, (3, 17): 0.0, (3, 18): 0.0, (4, 1): 0.0, (4, 2): 0.0, (4, 3): 0.0, (4, 4): 0.0, (4, 5): 54.0, (4, 6): 0.0, (4, 7): 0.0, (4, 8): 0.0, (4, 9): 0.0, (4, 10): 0.0, (4, 11): 0.0, (4, 12): 0.0, (4, 13): 0.0, (4, 14): 0.0, (4, 15): 0.0, (4, 16): 0.0, (4, 17): 0.0, (4, 18): 0.0, (5, 1): 0.0, (5, 2): 0.0, (5, 3): 0.0, (5, 4): 0.0, (5, 5): 0.0, (5, 6): 0.0, (5, 7): 0.0, (5, 8): 0.0, (5, 9): 0.0, (5, 10): 0.0, (5, 11): 0.0, (5, 12): 0.0, (5, 13): 0.0, (5, 14): 0.0, (5, 15): 54.0, (5, 16): 0.0, (5, 17): 0.0, (5, 18): 0.0}

        #result_s
        #y:
        # N_hat: N_j_t
        # N_bar: N_s
        # q: q_j_t
        # X: X_j_t
        # delta: delta_j_t
        # xi: xi_j_t
        # zeta: zeta_j_t
        result=case3.solveBenders(100,y_initial,1000)#result_s,y,UB_LB_tol_dict
                                                                #UB_LB_tol_dict:{iter:(UB,LB,UB-LB)}

    return result

if __name__=="__main__":
    distance = [16, 20, 16, 24, 20]
    average_distance = [5, 7, 5, 8, 6]
    speed=[
        [48,40,40,48,48,48,48,48,48,48,48,40,40,48,48,48,48,48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48],
        [48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48]
    ]
    demand=[
        [100,200,200,100,100,100,100,100,100,100,100,200,200,100,100,100,100,100],
        [150, 300, 300, 150, 150, 150, 150, 150, 150, 150, 150, 300, 300, 150, 150, 150, 150, 150],
        [60, 150, 150, 60, 60, 60, 60, 60, 60, 60, 60, 150, 150, 60, 60, 60, 60,60],
        [200, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200, 400, 400, 200, 200, 200, 200, 200],
        [125, 250, 250, 125, 125, 125, 125, 125, 125, 125, 125, 250, 250, 125, 125, 125, 125, 125]
    ]
    peak_point_demand=[
        [100,200,200,100,100,100,100,100,100,100,100,200,200,100,100,100,100,100],
        [150, 300, 300, 150, 150, 150, 150, 150, 150, 150, 150, 300, 300, 150, 150, 150, 150, 150],
        [60, 150, 150, 60, 60, 60, 60, 60, 60, 60, 60, 150, 150, 60, 60, 60, 60,60],
        [200, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200, 400, 400, 200, 200, 200, 200, 200],
        [125, 250, 250, 125, 125, 125, 125, 125, 125, 125, 125, 250, 250, 125, 125, 125, 125, 125]
    ]

    '''
    #OneSizeSystemwide
    #def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand):
    case1=oss.OneSize(routeNo=5,distance=[16,20,16,24,20],average_distance=[5,7,5,8,6],speed=speed,demand=demand,peak_point_demand=peak_point_demand)
    result1=case1.Optimal()
    print(result1)
    '''

    '''
    #TwoSizeSystemwide
    #def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand,size_type):
    case2=tss.TwoSize(routeNo=5,distance=distance,average_distance=average_distance,speed=speed,demand=demand,peak_point_demand=peak_point_demand,size_type=2)
    # can not run #
    #result2=case2.Optimal()
    result2=case2.BFGS()
    print(result2[len(result2)-1]['solution'])
    print(result2[len(result2) - 1]['objValue'])
    print(result2[len(result2) - 1]['headway'])
    print(result2[len(result2) - 1]['delta'])
    print(result2[len(result2) - 1]['fleet'])
    print(result2[0]['solution'])
    print(result2[0]['objValue'])

    pass
    '''

    # #typeList=['OneS','TwoS','TwoSF']
    # result_oneSize=Freight_Systemwide(5, distance, average_distance, speed, demand, peak_point_demand, 'OneS')
    # print(result_oneSize)
    # logger.info('The result of one size is\n {}'.format(result_oneSize))
    # result_twoSize = Freight_Systemwide(5, distance, average_distance, speed, demand, peak_point_demand, 'TwoS')
    # print(result_twoSize[len(result_twoSize)-1]['fleet'])
    # print(result_twoSize[len(result_twoSize) - 1]['solution'])
    # logger.info('The result of two-size systemwide is \n {}'.format(result_twoSize[len(result_twoSize)-1]))
    result_twoSizeF=Freight_Systemwide(5, distance, average_distance, speed, demand, peak_point_demand, 'TwoSF')
    print(result_twoSizeF)
    print(result_twoSizeF[0]['headway'])
    print(result_twoSizeF[1]['N_hat'])
    print(result_twoSizeF[1]['N_bar'])
    print(result_twoSizeF[1]['y_0'])
    print(result_twoSizeF[1]['q'])
    print(result_twoSizeF[1]['X'])
    print(result_twoSizeF[1]['delta'])
    print(result_twoSizeF[1]['xi'])
    #print(result_twoSizeF[1]['N_bar'])
    #logger.info('The result of two-size systemwide with freight on transit is \n {}'.format(result_twoSizeF))

    # with open('data_two.pickle', 'rb') as f:
    #     result_init = pickle.load(f)
    #     result_init = result_init[len(result_init) - 1]
    # print(result_init['solution'])
    # print(result_init['delta'])
    # print(result_init['headway'])
    # print(result_init['fleet'])
    # print(result_init['objValue'])
