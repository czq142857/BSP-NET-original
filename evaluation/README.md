# evaluation

*all_vox256_img_test.txt* is a list of names of the shapes in the testing set.

Change the folder paths (*gt_point_dir*, *bspnet_point_dir*) in *eval_bspnet_pc.py*, *edge_from_point.py* and *eval_bspnet_edge_cd.py*.

Run *eval_bspnet_pc.py* to get Chamfer Distance and Normal Consistency.

The Chamfer Distance and Normal Consistency for each object is written to *result_per_obj.txt*.

The average Chamfer Distance and Normal Consistency for each category is written to *result_per_category.txt*.


Run *edge_from_point.py* to get edge points from the sampled dense point cloud, then run *eval_bspnet_edge_cd.py* to get Chamfer Distance between edge points.

The Chamfer Distance and Normal Consistency for each object is written to *result_ECD_per_obj.txt*.

The average Chamfer Distance and Normal Consistency for each category is written to *result_ECD_per_category.txt*.

