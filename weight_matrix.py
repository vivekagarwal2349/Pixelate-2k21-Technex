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