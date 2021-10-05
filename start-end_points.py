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
