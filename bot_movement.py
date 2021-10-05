##################################################   robust bot control    ###################################################

def go_go_go(path_list, path_code):

    for i in range(len(path_list)):
        path_list[i]=int(path_list[i])
 
    # i=len(path_list)-1
    # env.remove_cover_plat(0,0)
    print("Length path list",len(path_list))
    global diff_pre
    diff_nxt = path_list[0]-path_list[1]
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
            print(cross_product[2]*f)
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

            elif -8<cross_product[1]<8:

                for j in range(0,60):
                    print("inertia")
                    env.move_husky(0,0,0,0)
                    p.stepSimulation()
                k=0
                print(i,"else #######################################################")
                break

            elif diff_pre == diff_nxt and i==0: 
                print("qwertyuio")
                # time.sleep(1)
                for hu in range(5):
                    env.move_husky(1,-1,1,-1)
                    for we in range(855):
                        p.stepSimulation()
                    env.move_husky(-1,-1,-1,-1)
                    for er in range(55):
                        p.stepSimulation()
                env.move_husky(1,1,1,1)
                for io in range(520):
                    p.stepSimulation()

                i+=1
            elif i+2 < len(path_list) and -2<cross_product[2]*f<2 and path_list[i+1]-path_list[i]==path_list[i+2]-path_list[i+1]:
                
                print("flashhhhh")
                # time.sleep(2)
                env.move_husky(2,2,2,2)
                for mk in range(350):
                    p.stepSimulation()
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
    
    diff_pre = path_list[len(path_list)-1]-path_list[len(path_list)-2]

def cover_plate(i):
    '''Removes the cover plate over a block by taking its number'''

    x,y=list(names.keys())[list(names.values()).index(i)]
    x,y=x-1,y-1
    print("Coordinates of lid = ",(x,y))

    env.remove_cover_plate(x,y)
    # env.remove_cover_plate((i-1)%num,(i-1)//num)