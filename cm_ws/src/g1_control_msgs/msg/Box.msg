# Oriented box obstacle, e.g. fitted to a segmented object.

std_msgs/Header header               # frame the pose is expressed in
string label                         # what it is, e.g. the segmentation prompt
float32 score                        # detection confidence
geometry_msgs/Pose pose              # box centre and orientation
geometry_msgs/Vector3 half_extents   # half sizes along the box's local x, y, z
