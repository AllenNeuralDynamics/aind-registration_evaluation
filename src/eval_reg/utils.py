import numpy as np

def sample_points_in_overlap(bounds1, bounds2, args ):
        """samples points in the overlap regions 
           and returns a list of points
           sampling types : random, grid, feature extracted
        """
        
        #check if there is intersection
        #######NEED TO DO!!!!

        #if there is, then : 
        if args['type'] == "random":
            o_min = [np.max([bounds1[0][0], bounds2[0][0]]),
                        np.max([bounds1[0][1], bounds2[0][1]]),
                            np.max([bounds1[0][2], bounds2[0][2]])]

            o_max = [np.min([bounds1[1][0], bounds2[1][0]]),
                        np.min([bounds1[1][1], bounds2[1][1]]),
                            np.min([bounds1[1][2], bounds2[1][2]])]

            print(range(o_min[0], o_max[0]))
            x = list(np.random.choice( range(o_min[0], o_max[0]), args['numpoints'] ))
            y = list(np.random.choice(range(o_min[1], o_max[1]), args['numpoints']))
            z = list(np.random.choice(range(o_min[2], o_max[2]), args['numpoints']))

        return list(np.array([x,y,z]).transpose())

    

def calculate_bounds(image1_shape, image2_shape, transform):
    """
    Calculate bounds of coverage for two images and a transform
    where image1 is in its own coordinate system and image 2 is mapped
    to image 1's coords with the transform
    """
    b1 = [[0,0,0],list(image1_shape)]
    pt_min = np.matrix([0,0,0,1]).transpose()
    pt_max = np.matrix(list(image2_shape)+[1]).transpose()
    b2 = [np.squeeze(transform*pt_min).tolist()[0][:3],
            np.squeeze(transform*pt_max).tolist()[0][:3]]  
            
    return b1,b2 
