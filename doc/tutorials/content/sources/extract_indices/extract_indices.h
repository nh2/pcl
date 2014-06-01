#include <iostream>
#include <stdexcept>
#include <pcl/common/intersections.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h> // needs HAVE_QHULL
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
