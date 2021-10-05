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