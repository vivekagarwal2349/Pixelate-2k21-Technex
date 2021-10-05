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