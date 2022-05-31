#RouteNo=5
#distance=[16,20,16,24,20]
#peak hour:4
#off-peak hour: 14
#speed=[(40,48)] #peak hour vs. off-peak hour
#demand=[(200,100),(300,150),(150,60).(400,200),(250,125)] # peak hour vs. off-peak hour

import OneSizeSystemwide as oss
import TwosizeSystemwide as tss
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


    #OneSizeSystemwide
    #def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand):
    case1=oss.OneSize(routeNo=5,distance=[16,20,16,24,20],average_distance=[5,7,5,8,6],speed=speed,demand=demand,peak_point_demand=peak_point_demand)
    result1=case1.Optimal()
    print(result1)

    '''
    #TwoSizeSystemwide
    #def __init__(self,routeNo,distance,average_distance,speed,demand,peak_point_demand,size_type):
    case2=tss.TwoSize(routeNo=5,distance=distance,average_distance=average_distance,speed=speed,demand=demand,peak_point_demand=peak_point_demand,size_type=2)
    result2=case2.Optimal()
    print(result2)
    for i in range(22):
        print(result2[i])
    '''
    pass