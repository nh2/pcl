#include "extract_indices.h"

using namespace std;
using namespace pcl;

const int num_colors = 4;
int plane_colors[num_colors][3] = {
  {   0, 255,   0 },
  {   0, 255, 255 },
  {   0,   0, 255 },
  { 255,   0, 255 },
};


template <typename Point>
void
projectOntoPlane (const PointCloud<Point> &plane_points,
                  Eigen::Vector4f &coeffs,
                  PointCloud<Point> &projected_plane_points)
{
  for (size_t i = 0; i < plane_points.points.size(); ++i)
  {
    Point projection;
    projectPoint (plane_points[i], coeffs, projection);
    projected_plane_points.push_back(projection);
  }
}


int
main (int argc, char** argv)
{
  PointCloud<PointXYZRGBNormal>::Ptr cloud_blob (new PointCloud<PointXYZRGBNormal>),
                                     cloud_filtered_blob (new PointCloud<PointXYZRGBNormal>);
  PointCloud<PointXYZRGBNormal>::Ptr cloud_filtered (new PointCloud<PointXYZRGBNormal>),
                                     cloud_p (new PointCloud<PointXYZRGBNormal>),
                                     cloud_f (new PointCloud<PointXYZRGBNormal>);

  if (argc != 7)
  {
    throw runtime_error("\n"
                        "Usage: extract_indices [file.pcd] [rest_ratio] [max_iterations] [distance_threshold] [optimize_coefficients] [downsample_size]\n"
                        "defaults:                             0.3            1000               0.01                   true                 0.01");
  }

  string filename = argv[1];
  double rest_ratio = atof(argv[2]);          // default: 0.3
  int max_iterations = atoi(argv[3]);         // default: 1000
  double distance_threshold = atof(argv[4]);  // default: 0.01
  string opt_coeff_str = argv[5];
  bool optimize_coefficients =                // default: true
       opt_coeff_str == "true"  ? true
    : (opt_coeff_str == "false" ? false : throw runtime_error("bad optimize_coefficients value"));
  double downsample_size = atof(argv[6]);     // default: 0.01
  bool downsample = downsample_size != atof("0");

  // Fill in the cloud data
  io::loadPCDFile (filename, *cloud_blob);

  cerr << "PointCloud: " << cloud_blob->width * cloud_blob->height << " data points." << endl;

  if (downsample)
  {
    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    VoxelGrid<PointXYZRGBNormal> sor;
    sor.setInputCloud (cloud_blob);
    sor.setLeafSize (downsample_size, downsample_size, downsample_size);
    sor.filter (*cloud_filtered_blob);

    *cloud_filtered = *cloud_filtered_blob;

    cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << endl;
  }
  else
  {
    *cloud_filtered = *cloud_filtered_blob;
  }

  // Write the downsampled version to disk
  io::savePCDFile ("cloud_downsampled.pcd", *cloud_filtered, true);

  ModelCoefficients::Ptr coefficients (new ModelCoefficients ());
  PointIndices::Ptr inliers (new PointIndices ());
  // Create the segmentation object
  // SACSegmentation<PointXYZ> seg;
  SACSegmentationFromNormals<PointXYZRGB, Normal> seg; // for normals
  // Optional
  seg.setOptimizeCoefficients (optimize_coefficients);
  // Mandatory
  // seg.setModelType (SACMODEL_PLANE);
  seg.setModelType (SACMODEL_NORMAL_PLANE); // for normals
  seg.setNormalDistanceWeight (0.1); // for normals

  seg.setMethodType (SAC_RANSAC);
  seg.setMaxIterations (max_iterations);
  seg.setDistanceThreshold (distance_threshold);

  // Copy XYZRGBNNormal cloud to two clouds (XYZRGB + Normal) because SACSegmentationFromNormals wants them separate
  PointCloud<PointXYZRGB>::Ptr cloud_filtered_xyz (new PointCloud<PointXYZRGB>);
  PointCloud<Normal>::Ptr      cloud_filtered_normals (new PointCloud<Normal>);
  cloud_filtered_xyz->resize(cloud_filtered->size());
  cloud_filtered_normals->resize(cloud_filtered->size());
  for (int i = 0; i < cloud_filtered->size(); ++i)
  {
    (*cloud_filtered_xyz)[i].getVector4fMap() = (*cloud_filtered)[i].getVector4fMap();
    (*cloud_filtered_normals)[i].getNormalVector4fMap() = (*cloud_filtered)[i].getNormalVector4fMap();
  }

  // Create the filtering object
  ExtractIndices<PointXYZRGBNormal> extract;

  vector<Eigen::Vector4f> planes;

  ofstream planes_file("planes.txt");

  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  // While (rest_ratio * 100%) of the original cloud is still there
  while (cloud_filtered->points.size () > rest_ratio * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    // seg.setInputCloud (cloud_filtered);
    seg.setInputCloud (cloud_filtered_xyz);
    seg.setInputNormals (cloud_filtered_normals); // for normals
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      cerr << "Could not estimate a planar model for the given dataset." << endl;
      break;
    }

    cout << "Cloud " << i << " coefficients: " << *coefficients << endl;

    Eigen::Vector4f plane = Eigen::Vector4f::Map(&coefficients->values[0]);
    planes.push_back(plane);

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);
    cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << endl;

    // Set plane color
    PointCloud<PointXYZRGB> cloud_colored;
    for (int pi = 0; pi < cloud_p->size(); ++pi)
    {
      PointXYZRGBNormal p = (*cloud_p)[pi];
      PointXYZRGB cp (plane_colors[i % num_colors][0],
                           plane_colors[i % num_colors][1],
                           plane_colors[i % num_colors][2]);
      cp.x = p.x;
      cp.y = p.y;
      cp.z = p.z;
      cloud_colored.push_back(cp);
    }

    {
      stringstream ss;
      ss << "cloud_plane_" << i << ".pcd";
      // io::savePCDFile (ss.str (), *cloud_p, true);
      io::savePCDFile (ss.str (), cloud_colored, true);
    }

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);

    ExtractIndices<PointXYZRGB> extract_xyz;
    extract_xyz.setInputCloud (cloud_filtered_xyz);
    extract_xyz.setIndices (inliers);
    extract_xyz.setNegative (true);
    extract_xyz.filter (*cloud_filtered_xyz);

    ExtractIndices<Normal> extract_normals;
    extract_normals.setInputCloud (cloud_filtered_normals);
    extract_normals.setIndices (inliers);
    extract_normals.setNegative (true);
    extract_normals.filter (*cloud_filtered_normals);


    // Write parameters to file
    {
      Eigen::Vector4f p = plane;
      planes_file << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << endl;
    }

    // For each point belonging to that plane, project it onto the
    // estimated plane parameters (since it could be slightly displaced).
    cout << "Projecting points to plane" << endl;
    PointCloud<PointXYZ> plane_points;
    PointCloud<PointXYZ>::Ptr projected_plane_points (new PointCloud<PointXYZ>);
    // PointCloud<PointXYZ> projected_plane_points;
    projected_plane_points->resize(plane_points.size());
    {
      stringstream ss;
      ss << "cloud_plane_" << i << ".pcd";
      io::loadPCDFile (ss.str(), plane_points);
    }
    projectOntoPlane(plane_points, plane, *projected_plane_points);
    cout << *projected_plane_points << endl;

    // Then do convex hull on the now flat points on the plane
    // to find their bounding polygon, so that we can draw/export that
    // insteat of an infinite plane.
    cout << "Calculating convex hull" << endl;
    ConvexHull<PointXYZ> ch;
    PointCloud<PointXYZ> hull_points;
    ch.setInputCloud(projected_plane_points);
    ch.reconstruct(hull_points);
    {
      stringstream ss;
      ss << "cloud_plane_hull" << i << ".pcd";
      io::savePCDFile (ss.str(), hull_points);
    }


    i++;
  }


  #if 0
  if (planes.size() < 3)
  {
    cout << "Not enough planes" << endl;
  }
  else
  {
    cout << "Cutting planes" << endl;

    Eigen::VectorXf cut_line1;
    bool ok1 = planeWithPlaneIntersection (planes[0], planes[1], cut_line1, 0.1);

    Eigen::VectorXf cut_line2;
    bool ok2 = planeWithPlaneIntersection (planes[0], planes[2], cut_line2, 0.1);

    if (!ok1 || !ok2)
    {
      cout << "cut failed" << endl;
    }
    else
    {
      cout << endl << "cut line 1:" << endl << cut_line1 << endl;
      cout << endl << "cut line 2:" << endl << cut_line2 << endl;

      Eigen::Vector4f corner;
      bool ok3 = lineWithLineIntersection (cut_line1, cut_line2, corner, 1e-4);

      if (!ok3)
      {
        cout << endl << "finding corner failed" << endl;
      }
      else
      {
        cout << endl << "corner point:" << endl << corner << endl;

        PointXYZ corner_point (corner[0], corner[1], corner[2]);

        PointXYZRGB corner_point_color(255,0,0);
        corner_point_color.x = corner_point.x;
        corner_point_color.y = corner_point.y;
        corner_point_color.z = corner_point.z;

        PointCloud<PointXYZRGB> corner_cloud;
        corner_cloud.push_back (corner_point_color);
        io::savePCDFile ("cloud_corners.pcd", corner_cloud, false);
      }
    }
  }
  #endif


  return (0);
}
