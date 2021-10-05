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