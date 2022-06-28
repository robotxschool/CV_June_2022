def get_robot_coords(coords):
    offset_x = -143
    offset_y = 200
    robot_coords = []
    for obj in coords:
        coord = obj["centroid_coord"]
        x_r = coord[0]+offset_x
        y_r = coord[1]+offset_y
        x_r, y_r = -y_r, x_r
        robot_coords.append((x_r, y_r, obj["object_class"]))
    return robot_coords        
    
