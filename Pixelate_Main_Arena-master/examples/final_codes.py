#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                        #####  #  #  #####     ##   ##  #####  #####  #####  #####  #   #
                          #    #  #  #         # # # #  #   #    #    #   #    #     # #
                          #    ####  ###       #  #  #  #####    #    #####    #      #
                          #    #  #  #         #     #  #   #    #    # #      #     # #
                          #    #  #  #####     #     #  #   #    #    #   #  #####  #   #

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

import cv2
import cv2.aruco as aruco
import numpy as np
import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
import os
import math
from collections import deque, namedtuple

#######################################    defining some global variables    ##########################################

points ={}
cross_product=[]
names = {}
weights=[]
hospitals = {}
patients=[]
last_position = []
last_block=0
num = 12
side=num*50

#######################################    determining weights from arena    ##########################################

def weight_matrix():
    """This fucntion will read the arena image and provide weights of all blocks"""

    # img = env.camera_feed()
    img=env.camera_feed()
    img = cv2.resize(img,(480,480))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,threshold=cv2.threshold(gray,5,255,cv2.THRESH_BINARY_INV)
    contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    cv2.drawContours(gray,contours,-1,(100),1)
    approx=cv2.approxPolyDP(contours[0],0.05*cv2.arcLength(contours[0],True),True)
    # print(approx)
    approx=np.squeeze(approx)
    xs=approx[:,0]
    ys=approx[:,1]
    lx=np.min(xs)+5
    ux=np.max(xs)+5
    ly=np.min(ys)+5
    uy=np.max(ys)+5

    arena=img[ly:uy,lx:ux]
    arena=cv2.resize(arena,(side,side))
    gray1=cv2.cvtColor(arena,cv2.COLOR_BGR2GRAY)
    _, th =cv2.threshold(gray1,50,255,cv2.THRESH_BINARY)

    def getweight(color,weight,x,y):
        # print("color :",color)
        # print("x:y :",x,y)
        if 0<=color[0]<=10 and 0<=color[1]<=10 and 140<=color[2]<=150:
            weight=4
        if 0<=color[0]<=10 and 220<=color[1]<=232 and 220<=color[2]<=232:
            weight=3
        if 0<=color[0]<=10 and 220<=color[1]<=235 and 0<=color[2]<=10:
            weight=2
        if 223<=color[0]<=230 and 223<=color[1]<=230 and 223<=color[2]<=230:
            weight=1
        if 205<=color[0]<=220 and 107<=color[1]<=122 and 205<=color[2]<=220:
            weight=99995
        # if 220<=color[0]<=235 and 0<=color[1]<=10 and 0<=color[2]<=10:
        #     blue color
        if 0<=color[0]<=5 and 0<=color[1]<=5 and 0<=color[2]<=5:
            weight=0

        return weight

    contours,hierarchy=cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(th,contours,-1,(100),3)
    x_coordinates=list()
    y_coordinates=list()

    for i in range(len(contours)):
        approx_weight=cv2.approxPolyDP(contours[i],0.035*cv2.arcLength(contours[i],True),True)
        if len(approx_weight)==4:
            M=cv2.moments(contours[i])
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            x_coordinates.append(cx)
            y_coordinates.append(cy)
    x_coordinates.sort()
    y_coordinates.sort()
    ratio=(x_coordinates[-1]-x_coordinates[0])//(num-1)
    finalx=list()
    finaly=list()
    for i in range(num):
        finalx.append(x_coordinates[0]+(i*ratio))
        finaly.append(y_coordinates[0]+(i*ratio))
    weights=np.ones([num,num], dtype=int)
    for i in range(num):
        for j in range(num):
            flag=99999
            x=finalx[j]
            y=finaly[i]
            color=arena[y][x]
            value=getweight(color,flag,x,y)
            weights[i][j]=value

    def oneway(flag,x_index,y_index,img):
        # Checking horizontal direction
        if abs(approx_blue[0][0] - approx_blue[1][0])<=5 :
            if approx_blue[2][0] > approx_blue[0][0]:
                flag=1
            if approx_blue[2][0] < approx_blue[0][0]:
                flag=3
        if abs(approx_blue[1][0] - approx_blue[2][0])<=5 :
            if approx_blue[0][0] > approx_blue[1][0]:
                flag=1
            if approx_blue[0][0] < approx_blue[1][0]:
                flag=3
        if abs(approx_blue[0][0] - approx_blue[2][0])<=5 :
            if approx_blue[1][0] > approx_blue[0][0]:
                flag=1
            if approx_blue[1][0] < approx_blue[0][0]:
                flag=3
        # Checking verical direction
        if abs(approx_blue[0][1] - approx_blue[1][1])<=5 :
            if approx_blue[2][1] > approx_blue[0][1]:
                flag=2
            if approx_blue[2][1] < approx_blue[0][1]:
                flag=4
        if abs(approx_blue[1][1] - approx_blue[2][1])<=5 :
            if approx_blue[0][1] > approx_blue[1][1]:
                flag=2
            if approx_blue[0][1] < approx_blue[1][1]:
                flag=4
        if abs(approx_blue[0][1] - approx_blue[2][1])<=5 :
            if approx_blue[1][1] > approx_blue[0][1]:
                flag=2
            if approx_blue[1][1] < approx_blue[0][1]:
                flag=4

        # print(flag)
        for i in range(len(finalx)):
            if cx-finalx[i]<=15 and cx-finalx[i]>=-15:
                x_index=i
        for j in range(len(finaly)):
            if cy-finaly[j]<=15 and cy-finaly[j]>=-15:
                y_index=j
        # print(x_index)
        # print(y_index)
        range_list=[5,8,10,12,14,16]
        if flag==3 or flag==4:
            for i in range_list:
                x=cx-i
                y=cy-i
                intens=img[y][x]
                z=0
                parent_weight=getweight(intens,z,x,y)
                if not parent_weight==0:
                    break
        else:
            for i in range_list:
                x=cx+i
                y=cy+i
                intens=img[y][x]
                z=0
                parent_weight=getweight(intens,z,x,y)
                if not parent_weight==0:
                    break
        # print(x)
        # print(y)
        # print(intens)
        newweight=(parent_weight*10)+flag
        # print(newweight)
        weights[y_index][x_index]=newweight

    lb=np.array([150,0,0])
    ub=np.array([255,40,40])
    blue_mask=cv2.inRange(arena,lb,ub)
    bcontours,_=cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for contour in bcontours:
        M=cv2.moments(contour)
        cx=int(M['m10']/M['m00'])
        cy=int(M['m01']/M['m00'])
        # print(cx)
        # print(cy)
        approx_blue=cv2.approxPolyDP(contour,0.035*cv2.arcLength(contour,True),True)
        approx_blue=np.squeeze(approx_blue)
        # print(approx_blue)
        if len(approx_blue) ==3:
            # print ("Triangle")
            flag=0
            x_index=0
            y_index=0
            oneway(flag,x_index,y_index,arena)

        if len(approx_blue) ==4:
            # print ("Square")
            # print(cx,cy)
            h_flag=1
            x_index=0
            y_index=0
            for i in range(len(finalx)):
                if cx-finalx[i]<=int(ratio/2 -5) and cx-finalx[i]>=-int(ratio/2 -5):
                    x_index=i
            for j in range(len(finaly)):
                if cy-finaly[j]<=int(ratio/2 -5) and cy-finaly[j]>=-int(ratio/2 -5):
                    y_index=j
            index=(y_index,x_index)
            hospitals[h_flag]=index

        if len(approx_blue) >=5:
            # print("Circle")
            h_flag=0
            x_index=0
            y_index=0
            for i in range(len(finalx)):
                if cx-finalx[i]<=int(ratio/2 -5) and cx-finalx[i]>=-int(ratio/2 -5):
                    x_index=i
            for j in range(len(finaly)):
                if cy-finaly[j]<=int(ratio/2 -5) and cy-finaly[j]>=-int(ratio/2 -5):
                    y_index=j
            index=(y_index,x_index)
            hospitals[h_flag]=index


    for i in range(num*num):
        coordinate=np.array([finalx[int(i/num)],finaly[int(i%num)]])
        points[i+1]=coordinate


    order = len(weights)
    c = 0
    for i in range(1, order + 1):
        for j in range(1, order + 1):
            c += 1
            names[(j, i)] = c


    print("Weights : ",weights)
    print("Hosptials : ",hospitals)
    return weights
    # print(x_coordinates)
    # print(finalx)
    # print(finaly)
    # print(points)
    # cv2.imshow("arena",arena)
    # cv2.imshow("gray",gray1)
    # cv2.imshow("threshold",th)
    # cv2.imshow("blue",blue_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


########################################    processing weights data    ##############################################

def path(begin, stop):
    """Provide the starting and end postion names and function will return shortest path"""

    arr= weights

    order = len(arr)
    arr = np.array(arr)
    arr = np.pad(arr, 1, mode="constant")

    adjacency_lists = []
    normal = [1, 2, 3, 4 , 99995, 99999]
    one_ways = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44]
    inf = 9999999999

    # c = 0
    # for i in range(1, order + 1):
    #     for j in range(1, order + 1):
    #         c += 1
    #         names[(j, i)] = c

    for i in range(1, order + 1):
        for j in range(1, order + 1):

            if arr[i][j] in normal:

                if arr[i][j - 1] in one_ways:
                    if arr[i][j - 1] % 10 == 1:  # left one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i, j - 1)]), inf)
                        )
                    else:  # valid left one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i, j - 1)]),
                                arr[i][j - 1] // 10,
                            )
                        )
                elif arr[i][j - 1]:  # normal left
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i, j - 1)]), arr[i][j - 1])
                    )



                if arr[i][j + 1] in one_ways:
                    if arr[i][j + 1] % 10 == 3:  # right one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i, j + 1)]), inf)
                        )
                    else:  # valid right one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i, j + 1)]),
                                arr[i][j + 1] // 10,
                            )
                        )
                elif arr[i][j + 1]:  # normal right
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i, j + 1)]), arr[i][j + 1])
                    )



                if arr[i - 1][j] in one_ways:
                    if arr[i - 1][j] % 10 == 2:  # up one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i - 1, j)]), inf)
                        )
                    else:  # valid up one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i - 1, j)]),
                                arr[i - 1][j] // 10,
                            )
                        )
                elif arr[i - 1][j]:  # normal up
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i - 1, j)]), arr[i - 1][j])
                    )



                if arr[i + 1][j] in one_ways:
                    if arr[i + 1][j] % 10 == 4:  # down one way
                        adjacency_lists.append(
                            (str(names[(i, j)]), str(names[(i + 1, j)]), inf)
                        )
                    else:  # valid down one way
                        adjacency_lists.append(
                            (
                                str(names[(i, j)]),
                                str(names[(i + 1, j)]),
                                arr[i + 1][j] // 10,
                            )
                        )
                elif arr[i + 1][j]:  # normal down
                    adjacency_lists.append(
                        (str(names[(i, j)]), str(names[(i + 1, j)]), arr[i + 1][j])
                    )



            elif arr[i][j] in one_ways:



                if arr[i][j] % 10 == 1:  # left invalid
                    if arr[i][j-1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i, j - 1)]), inf   )  )
                    if arr[i-1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i - 1, j)]), inf   )  )
                    if arr[i+1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i + 1, j)]), inf   )  )

                    if arr[i][j+1] in one_ways: #right one way
                        if arr[i][j + 1] % 10 == 3:  # invalid right one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j + 1)]), inf  )  )
                        else:  # valid rigt one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j + 1)]), arr[i][j + 1] // 10 )  )
                    elif arr[i][j+1] in normal:
                        adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j + 1)]), arr[i][j + 1] )  )



                if arr[i][j] % 10 == 2: #up invalid
                    if arr[i][j-1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i, j - 1)]), inf   )  )
                    if arr[i-1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i - 1, j)]), inf   )  )
                    if arr[i][j+1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i , j + 1)]), inf   )  )

                    if arr[i+1][j] in one_ways: #down one way
                        if arr[i+1][j] % 10 == 4:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i + 1, j)]), inf  )  )
                        else: #valid down one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i + 1, j)]), arr[i+1][j] // 10  )  )
                    elif arr[i+1][j] in normal:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i + 1, j)]), arr[i+1][j]  )  )



                if arr[i][j] % 10 == 3:  # right invalid
                    if arr[i-1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i - 1, j)]), inf   )  )
                    if arr[i+1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i + 1, j)]), inf   )  )
                    if arr[i][j+1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i , j + 1)]), inf   )  )

                    if arr[i][j-1] in one_ways: #left one way
                        if arr[i][j - 1] % 10 == 1:  # invalid left one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j - 1)]), inf  )  )
                        else:  # valid rigt one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j - 1)]), arr[i][j - 1] // 10 )  )
                    elif arr[i][j-1] in normal:
                        adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i, j - 1)]), arr[i][j - 1] )  )



                if arr[i][j] % 10 == 4: #down invalid
                    if arr[i][j-1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i, j - 1)]), inf   )  )
                    if arr[i+1][j]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i + 1, j)]), inf   )  )
                    if arr[i][j+1]:
                        adjacency_lists.append(  (   str(names[(i, j)]), str(names[(i , j + 1)]), inf   )  )

                    if arr[i-1][j] in one_ways: #up one way
                        if arr[i-1][j] % 10 == 2:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i - 1, j)]), inf  )  )
                        else: #valid up one way
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i - 1, j)]), arr[i-1][j] // 10  )  )
                    elif arr[i-1][j] in normal:
                            adjacency_lists.append(  (  str(names[(i, j)]), str(names[(i - 1, j)]), arr[i-1][j]  )  )




    # print(arr)
    arr = arr[1:-1, 1:-1]
    # print(arr)

    print("Block numbers : ",names)

##################################################    path planning    #########################################################

    inf = float('inf')
    Edge = namedtuple('Edge', 'start, end, cost')


    def make_edge(start, end, cost=1):
        return Edge(start, end, cost)


    class Graph:
        def __init__(self, edges):
            # let's check that the data is right
            wrong_edges = [i for i in edges if len(i) not in [2, 3]]
            if wrong_edges:
                raise ValueError('Wrong edges data: {}'.format(wrong_edges))

            self.edges = [make_edge(*edge) for edge in edges]

        @property
        def vertices(self):
            return set(
                sum(
                    ([edge.start, edge.end] for edge in self.edges), []
                )
            )

        def get_node_pairs(self, n1, n2, both_ends=True):
            if both_ends:
                node_pairs = [[n1, n2], [n2, n1]]
            else:
                node_pairs = [[n1, n2]]
            return node_pairs

        def remove_edge(self, n1, n2, both_ends=True):
            node_pairs = self.get_node_pairs(n1, n2, both_ends)
            edges = self.edges[:]
            for edge in edges:
                if [edge.start, edge.end] in node_pairs:
                    self.edges.remove(edge)

        def add_edge(self, n1, n2, cost=1, both_ends=True):
            node_pairs = self.get_node_pairs(n1, n2, both_ends)
            for edge in self.edges:
                if [edge.start, edge.end] in node_pairs:
                    return ValueError('Edge {} {} already exists'.format(n1, n2))

            self.edges.append(Edge(start=n1, end=n2, cost=cost))
            if both_ends:
                self.edges.append(Edge(start=n2, end=n1, cost=cost))

        @property
        def neighbours(self):
            neighbours = {vertex: set() for vertex in self.vertices}
            for edge in self.edges:
                neighbours[edge.start].add((edge.end, edge.cost))

            return neighbours

        def dijkstra(self, source, dest):
            assert source in self.vertices, 'Such source node doesn\'t exist'
            distances = {vertex: inf for vertex in self.vertices}
            previous_vertices = {
                vertex: None for vertex in self.vertices
            }
            distances[source] = 0
            vertices = self.vertices.copy()

            while vertices:
                current_vertex = min(
                    vertices, key=lambda vertex: distances[vertex])
                vertices.remove(current_vertex)
                if distances[current_vertex] == inf:
                    break
                for neighbour, cost in self.neighbours[current_vertex]:
                    alternative_route = distances[current_vertex] + cost
                    if alternative_route < distances[neighbour]:
                        distances[neighbour] = alternative_route
                        previous_vertices[neighbour] = current_vertex

            path, current_vertex = deque(), dest
            while previous_vertices[current_vertex] is not None:
                path.appendleft(current_vertex)
                current_vertex = previous_vertices[current_vertex]
            if path:
                path.appendleft(current_vertex)
            return path


    graph = Graph(adjacency_lists)

    return list(graph.dijkstra(str(begin), str(stop)))


###################################################    direction    ######################################################

def vector(path_list,i):
    '''Takes input as shortest path and current block number. Reads aruco. Returns the cross product , path length , angle , align index , dot product, centre angle'''

    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

    # img=env.camera_feed()
    img = env.camera_feed()
    img=cv2.resize(img,(480,480))

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,threshold=cv2.threshold(gray,5,255,cv2.THRESH_BINARY_INV)
    contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    cv2.drawContours(gray,contours,-1,(100),1)
    approx=cv2.approxPolyDP(contours[0],0.05*cv2.arcLength(contours[0],True),True)
    # print(approx)
    approx=np.squeeze(approx)
    xs=approx[:,0]
    ys=approx[:,1]
    lx=np.min(xs)+5
    ux=np.max(xs)+5
    ly=np.min(ys)+5
    uy=np.max(ys)+5

    gray=gray[ly:uy,lx:ux]
    gray = cv2.resize(gray,(side,side))
    cv2.imwrite("current_arena.png",img)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    # cv2.imshow("aruco",gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corners= np.array(corners)
    ids= np.array(ids)

    # corners=corners.squeeze()
    # ids=ids.squeeze()

    if ids.size!=1:
        corners=corners.squeeze()
        ids=ids.squeeze()
        ids = list(ids)
        ind = ids.index(107)
        corners = corners[ind]
    else:
        corners=corners.squeeze()
        ids=ids.squeeze()

    print("ID = ",ids)
    print("Corners = ",corners)

    if len(corners)==0:

        print("centre!@@###$%^&******************")

        c = side/2 #########@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@###########@@@@@@@@########@@###$$$$$$$$$$
        centre_vector = np.array([c - last_position[0] , c - last_position[1]])
        centre_product = np.array([last_position[2][0]*centre_vector[1] - last_position[2][1]*centre_vector[0] , 999999 , 0 ,1 , 99999])

        centre_angle = math.asin(centre_product[0]/(math.sqrt(last_position[2][0]**2 + last_position[2][1]**2)*math.sqrt(centre_vector[0]**2 + centre_vector[1]**2)))
        centre_product[2] = centre_angle
        return centre_product

    c=side/2
    position_x=(corners[0][0]+corners[2][0])/2
    position_y=(corners[0][1]+corners[2][1])/2

    position=[position_x,position_y]

    centre_vector_x= np.array([c-position_x,c-position_y])

    p=0.002
    position_x+=p*centre_vector_x[0]
    position_y+=p*centre_vector_x[1]

    corners[0][0]+=p*centre_vector_x[0]
    corners[0][1]+=p*centre_vector_x[1]
    corners[1][0]+=p*centre_vector_x[0]
    corners[1][1]+=p*centre_vector_x[1]
    print(position)


    bot_vector = np.array([(corners[0][0] + corners[1][0])/2.0 - position[0] , (corners[0][1] + corners[1][1])/2.0 -position[1] ])
    path_vector = np.array([points[path_list[i+1]][0] - position[0] , points[path_list[i+1]][1] - position[1]])
    dot_product = bot_vector[0]*path_vector[0] + bot_vector[1]*path_vector[1]

    path_length = math.sqrt(pow((points[path_list[i+1]][0] - position[0]),2) + pow((points[path_list[i+1]][1] - position[1]),2))
    cross_product = np.array([bot_vector[0]*path_vector[1] - bot_vector[1]*path_vector[0] , path_length,0,0,0,0])

    angle = math.asin(cross_product[0]/(math.sqrt(bot_vector[0]**2 + bot_vector[1]**2)*math.sqrt(path_vector[0]**2 + path_vector[1]**2)))
    cross_product[2] = angle
    cross_product[4] = dot_product

    print(angle * 180 / 3.1415)

    last_position.clear()
    last_position.append(position_x)
    last_position.append(position_y)
    last_position.append(bot_vector)
    last_position.append(0)

    c = side/2 #########@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@###########@@@@@@@@########@@###$$$$$$$$$$
    centre_vector = np.array([c - last_position[0] , c - last_position[1]])
    centre_product = np.array([last_position[2][0]*centre_vector[1] - last_position[2][1]*centre_vector[0] , 999999 , 0 ,1 , 99999])

    centre_angle = math.asin(centre_product[0]/(math.sqrt(last_position[2][0]**2 + last_position[2][1]**2)*math.sqrt(centre_vector[0]**2 + centre_vector[1]**2)))
    centre_product[2] = centre_angle

    cross_product[5] = centre_angle

    print("Bot vector : ",bot_vector)
    print("Path vector : ",path_vector)
    print("Cross product : ", cross_product)
    # print("last_position : ",last_position)

    #cross_product(cross product , path length , angle , align index , dot product, centre angle)

    return cross_product



##################################################   jugaad bot control    ###################################################

# def bkwd():
#     print("bck")
#     x=0
#     while True:
#         p.stepSimulation()
#         if x<2665:
#             env.move_husky(-0.5,-0.5,-0.5,-0.5)
#             x+=1
#         else:
#             env.move_husky(0,0,0,0)
#             break

# def frwd():
#     print("fr")
#     x=0
#     while True:
#         p.stepSimulation()
#         if x<2700:
#             env.move_husky(0.5,0.5,0.5,0.5)
#             x+=1
#         else:
#             env.move_husky(0,0,0,0)
#             break

# def turn_right():
#     print("Right")
#     x=0
#     y=0
#     z=0
#     while True:
#         p.stepSimulation()
#         if y<300:
#             env.move_husky(0.5,0.5,0.5,0.5)
#             y+=1
#         elif x<510:
#             env.move_husky(5,-5,5,-5)
#             x+=1
#         elif z<940:
#             env.move_husky(-0.5,-0.5,-0.5,-0.5)
#             z+=1
#         else:
#             frwd()
#             break

# def turn_left():
#     print("left")
#     x=0
#     y=0
#     z=0
#     while True:
#         p.stepSimulation()
#         if y<300:
#             env.move_husky(0.5,0.5,0.5,0.5)
#             y+=1
#         elif x<510:
#             env.move_husky(-5,5,-5,5)
#             x+=1
#         elif z<940:
#             env.move_husky(-0.5,-0.5,-0.5,-0.5)
#             z+=1
#         else:
#             frwd()
#             break

# def cover_plate(x,y):
#     env.remove_cover_plate(x,y)

# def go_go_go(path_list):
#     i=0
#     k=0
#     n = len(path_list)
#     for i in range(n):
#         path_list[i]=int(path_list[i])

#     while True:
#         for i in range(n):

#             if i+1==n-1 :
#                 cover_plate(0,0)

#             if k==0 :
#                 if path_list[i+1]==path_list[i]-6 :
#                     frwd()

#                 elif path_list[i+1]==path_list[i]+6 :
#                     bkwd()

#                 elif path_list[i+1]==path_list[i]-1 :
#                     turn_right()
#                     k=1

#                 elif path_list[i+1]==path_list[i]+1 :
#                     turn_left()
#                     k=1

#             elif k==1 :
#                 if path_list[i+1]==path_list[i]-6 :
#                     turn_left()
#                     k=0

#                 elif path_list[i+1]==path_list[i]+6 :
#                     turn_right()
#                     k=0

#                 elif path_list[i+1]==path_list[i]-1 :
#                     frwd()

#                 elif path_list[i+1]==path_list[i]+1 :
#                     bkwd()

#             elif i==n-1 :
#                 env.move_husky(0,0,0,0)


##################################################   robust bot control    ###################################################

def go_go_go(path_list, path_code):

    for i in range(len(path_list)):
        path_list[i]=int(path_list[i])

    # i=len(path_list)-1
    # env.remove_cover_plat(0,0)
    print("Length path list",len(path_list))
    k=0
    check=0
    for i in range(len(path_list)):
        turn=0
        turn1=0
        g=0
        if i==len(path_list)-1:
            for u in range(0,50):
                env.move_husky(1,1,1,1)
                p.stepSimulation()
            print("Success")
            break
        while True:
            print("-------------------------------------------------",i)
            # time.sleep(2)
            cross_product = vector(path_list,i)

                #
                # if len(cross_product)==0:
                #     env.move_husky(20,5,20,5)
                #     p.stepSimulation()
            f=180/3.1415
            if cross_product[3]==1  :
                k+=1

                if cross_product[2]*f >= 0:
                    print("align right")
                    env.move_husky(20,-5,20,-5)

                else:
                    print("align left")
                    env.move_husky(-5,20,-5,20)

                for ji in range(5):
                    p.stepSimulation()

            elif path_code%2==0 and i==len(path_list)-2 and check!=1:
                print("cover plate")
                cover_plate(path_list[i+1])
                check=1

            elif i==len(path_list)-1 and -15<cross_product[1]<15:
                print(i,"else #######################################################")
                break

            elif -7<cross_product[1]<7:

                for j in range(0,60):
                    print("inertia")
                    env.move_husky(0,0,0,0)
                    p.stepSimulation()
                k=0
                print(i,"else #######################################################")
                break

            elif (-7 < cross_product[2]*f < 7) and cross_product[4]>= 0 :
                r=0.85
                k+=1
                if 3<cross_product[2]*f<7:
                    r=1
                    print("straight right")
                    env.move_husky(r*cross_product[1] +1,r*cross_product[1] -5,r*cross_product[1] +1,r*cross_product[1] -5)



                elif -3>cross_product[2]*f>-7:
                    r=1
                    print("straight left")
                    env.move_husky(r*cross_product[1] -5,r*cross_product[1] +1,r*cross_product[1] -5,r*cross_product[1] +1)


                else:
                    print("straight")
                    env.move_husky(r*cross_product[1] +1,r*cross_product[1] +1,r*cross_product[1] +1,r*cross_product[1] +1)

                for t in range(7):
                    p.stepSimulation()

            elif cross_product[2]*f > 5:
                if turn==1 and turn1 > 175:
                    time.sleep(1)
                    for s in range(0,25):
                        print("bool back ffffff")
                        env.move_husky(-1,-1,-1,-1)
                        p.stepSimulation()
                    turn1=0
                    turn=0
                elif k>=20 and cross_product[2]*f > 20:
                    for j in range(0,50):
                        print("inertia")
                        env.move_husky(0,0,0,0)
                        p.stepSimulation()
                    k=0

                else:
                    print("turn right")
                    if cross_product[2]*f > 60:
                        env.move_husky(10,-10,10,-10)
                    else:
                        env.move_husky(abs(cross_product[2])*21+1,-abs(cross_product[2])*21-1,abs(cross_product[2])*21+1,-abs(cross_product[2])*21-1)
                    for t in range(5):
                        p.stepSimulation()
                    k=0
                    turn1 +=1


            elif cross_product[2]*f < -5:

                if turn==1 and turn1 >200:
                    time.sleep(5)
                    for s in range(0,75):
                        print("bool back ffffff")
                        env.move_husky(-2,-2,-2,-2)
                        p.stepSimulation()
                    turn1=0
                    turn=0

                elif k>=20 and cross_product[2]*f > 20:
                    for j in range(0,50):
                        print("inertia")
                        env.move_husky(0,0,0,0)
                        p.stepSimulation()
                    k=0

                else:
                    print("turn left")
                    if cross_product[2]*f < -60:
                        env.move_husky(-10,10,-10,10)
                    else:
                        env.move_husky(-abs(cross_product[2])*21-1,abs(cross_product[2])*21+1,-abs(cross_product[2])*21-1,21*abs(cross_product[2])+1)
                    for t in range(5):
                        p.stepSimulation()
                    k=0
                    turn1+=1

            elif cross_product[4]<0:
                # time.sleep(1)
                turn = 1
                if g==50:
                    # time.sleep(5)
                    for h in range(0,50):
                        print("bool back")
                        env.move_husky(-2,-2,-2,-2)
                        p.stepSimulation()
                        g=0

                if cross_product[5] >= 0:
                    print("bool right")
                    env.move_husky(20,-20,20,-20)
                    p.stepSimulation()
                    g+=1

                else:
                    print("bool left")
                    env.move_husky(-20,20,-20,20)
                    p.stepSimulation()
                    g+=1

            elif cross_product[4]<0 and cross_product[2]*f >=0 :
                if k>=10:
                    for j in range(0,50):
                        print("inertia")
                        env.move_husky(0,0,0,0)
                        p.stepSimulation()
                    k=0

                else:
                    print("turn left fffffffffff")

                    env.move_husky(-10,10,-10,10)
                    p.stepSimulation()
                    k=0

            elif cross_product[4]<0 and cross_product[2]*f <0 :
                if k>=10:
                    for j in range(0,50):
                        print("inertia")
                        env.move_husky(0,0,0,0)
                        p.stepSimulation()
                    k=0

                else:
                    print("turn right fffffff")

                    env.move_husky(10,-10,10,-10)
                    p.stepSimulation()
                    k=0

            # elif cross_product[1] < 20:
            #     print()

def cover_plate(i):
    '''Removes the cover plate over a block by taking its number'''

    x,y=list(names.keys())[list(names.values()).index(i)]
    x,y=x-1,y-1
    print("Coordinates of lid = ",(x,y))

    env.remove_cover_plate(x,y)
    # env.remove_cover_plate((i-1)%num,(i-1)//num)

###########################################################    start & end points    #############################################################

def patient_locations():
    '''Creates a list of patient locations in the arena'''

    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if weights[i][j] == 99995:
                patients.append((i+1, j+1))


def start_end(path_code):
    '''Takes the path code as input and returns the starting and ending positions in the arena of shortest path'''

    print("Path code =",path_code)
    global last_block

    if path_code == 0:
        start = len(weights)**2

        destination = names[patients[-1]]

        last_block= destination

        print("Start : ",start)
        print("Destination : ",destination)

        return (start, destination)

    elif path_code == 1:
        start = last_block
        print("Hospitals : ",hospitals)
        if check_patient(patients[-1]):
            destination = ((hospitals[1][0]+1),( hospitals[1][1]+1))
            del hospitals[1]
        else:
            destination = (( hospitals[0][0]+1),( hospitals[0][1]+1))
            del hospitals[0]

        destination = names[ destination]

        last_block= destination

        print("Start : ",start)
        print("Destination : ",destination)

        return (start, destination)

    elif path_code==2:
        start = last_block
        destination = names[patients[-2]]

        last_block= destination

        print("Start : ",start)
        print("Destination : ",destination)

        return (start, destination)

    elif path_code==3:
        start= last_block

        destination = (hospitals[list(hospitals.keys())[0]][0]+1 , hospitals[list(hospitals.keys())[0]][1]+1)
        destination = names[ destination]

        print("Start : ",start)
        print("Destination : ",destination)

        return (start, destination)

##################################################   covid status   #######################################################

def check_patient(patient):
    '''Checks the covid status of a patient. Returns 1 for covid and 0 for non-covid'''

    # image1= env.camera_feed()
    image1= env.camera_feed()
    image1 = cv2.resize(image1,(480,480))

    gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    _,threshold=cv2.threshold(gray,5,255,cv2.THRESH_BINARY_INV)
    contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    cv2.drawContours(gray,contours,-1,(100),1)
    approx=cv2.approxPolyDP(contours[0],0.05*cv2.arcLength(contours[0],True),True)
    # print(approx)
    approx=np.squeeze(approx)
    xs=approx[:,0]
    ys=approx[:,1]
    lx=np.min(xs)+5
    ux=np.max(xs)+5
    ly=np.min(ys)+5
    uy=np.max(ys)+5

    arena=image1[ly:uy,lx:ux]
    arena=cv2.resize(arena,(side,side))

    lb=np.array([150,0,0])
    ub=np.array([255,40,40])
    blue_mask=cv2.inRange(arena,lb,ub)
    bcontours,_=cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for contour in bcontours:
            M=cv2.moments(contour)
            cx=int(M['m10']/M['m00'])
            cy=int(M['m01']/M['m00'])
            # print(cx)
            # print(cy)
            approx_blue=cv2.approxPolyDP(contour,0.05*cv2.arcLength(contour,True),True)
            if abs(patient[0]-cx)<=5 and abs(patient[1]-cy)<=5:
                if(len(approx_blue)==4):
                    p_flag=1
                    print("COVID Patient")
                    return 1
                if(len(approx_blue)==5):
                    p_flag=0
                    print("NON COVID Patient")
                    return 0



####################################################    main    ########################################################


if __name__ == "__main__":

    parent_path = os.path.dirname(os.getcwd())
    os.chdir(parent_path)
    env = gym.make("pix_main_arena-v0")
    weights= weight_matrix()

    patient_locations()
    pat1=names[patients[-1]]
    pat2=names[patients[-2]]

    if len(path(144,pat1)) > len(path(144,pat2)):
        temp = patients[-1]
        patients[-1] = patients[-2]
        patients[-2] = temp

    print("Patients = ",patients)

    shuru= time.time()

    for path_code in range(0,num//3):

        start,destination = start_end(path_code)
        path_list = path(start,destination)
        # path_list = ['36','30','24','30']
        print("Shortest Path : ", path_list)
        print("Points : ",points)
        go_go_go(path_list,path_code)

    else:
        for u in range(0,50):
            env.move_husky(1,1,1,1)
            p.stepSimulation()

        print("MISSION PASSED!")
        print("Respect +")

        khatam = time.time()
        print("Time taken = ",khatam-shuru)
        time.sleep(69)
