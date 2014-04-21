#include "extract_indices.h"

const int num_colors = 4;
int plane_colors[num_colors][3] = {
  {   0, 255,   0 },
  {   0, 255, 255 },
  {   0,   0, 255 },
  { 255,   0, 255 },
};

int
main (int argc, char** argv)
{
  pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2),
                           cloud_filtered_blob (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>),
                                      cloud_p (new pcl::PointCloud<pcl::PointXYZ>),
                                      cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  if (argc != 7)
  {
    throw std::runtime_error("\n"
                             "Usage: extract_indices [file.pcd] [rest_ratio] [max_iterations] [distance_threshold] [optimize_coefficients] [downsample_size]\n"
                             "defaults:                             0.3            1000               0.01                   true                 0.01");
  }

  std::string filename = argv[1];
  double rest_ratio = atof(argv[2]);          // default: 0.3
  int max_iterations = atoi(argv[3]);         // default: 1000
  double distance_threshold = atof(argv[4]);  // default: 0.01
  std::string opt_coeff_str = argv[5];
  bool optimize_coefficients =                // default: true
       opt_coeff_str == "true"  ? true
    : (opt_coeff_str == "false" ? false : throw std::runtime_error("bad optimize_coefficients value"));
  double downsample_size = atof(argv[6]);     // default: 0.01
  bool downsample = downsample_size != atof("0");

  // Fill in the cloud data
  pcl::PCDReader reader;
  reader.read (filename, *cloud_blob);

  std::cerr << "PointCloud: " << cloud_blob->width * cloud_blob->height << " data points." << std::endl;

  if (downsample)
  {
    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud (cloud_blob);
    sor.setLeafSize (downsample_size, downsample_size, downsample_size);
    sor.filter (*cloud_filtered_blob);

    // Convert to the templated PointCloud
    pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;
  }
  else
  {
    pcl::fromPCLPointCloud2 (*cloud_blob, *cloud_filtered);
  }

  // Write the downsampled version to disk
  pcl::io::savePCDFile ("cloud_downsampled.pcd", *cloud_filtered, true);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg; // for normals
  // Optional
  seg.setOptimizeCoefficients (optimize_coefficients);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  // seg.setModelType (pcl::SACMODEL_NORMAL_PLANE); // for normals
  // seg.setNormalDistanceWeight (0.1); // for normals

  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (max_iterations);
  seg.setDistanceThreshold (distance_threshold);

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  std::vector<Eigen::Vector4f> planes;

  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  // While (rest_ratio * 100%) of the original cloud is still there
  while (cloud_filtered->points.size () > rest_ratio * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    std::cout << "Cloud " << i << " coefficients: " << *coefficients << std::endl;

    planes.push_back(Eigen::Vector4f::Map(&coefficients->values[0]));

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);
    std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

    // Set plane color
    pcl::PointCloud<pcl::PointXYZRGB> cloud_colored;
    for (int pi = 0; pi < cloud_p->size(); ++pi)
    {
      pcl::PointXYZ p = (*cloud_p)[pi];
      pcl::PointXYZRGB cp (plane_colors[i % num_colors][0],
                           plane_colors[i % num_colors][1],
                           plane_colors[i % num_colors][2]);
      cp.x = p.x;
      cp.y = p.y;
      cp.z = p.z;
      cloud_colored.push_back(cp);
    }

    std::stringstream ss;
    ss << "cloud_plane_" << i << ".pcd";
    // pcl::io::savePCDFile (ss.str (), *cloud_p, true);
    pcl::io::savePCDFile (ss.str (), cloud_colored, true);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
  }

  std::ofstream planes_file("planes.txt");
  for (int i = 0; i < planes.size(); ++i)
  {
    Eigen::Vector4f p = planes[i];
    planes_file << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;
  }


  if (planes.size() < 3)
  {
    std::cout << "Not enough planes" << std::endl;
  }
  else
  {
    std::cout << "Cutting planes" << std::endl;

    Eigen::VectorXf cut_line1;
    bool ok1 = pcl::planeWithPlaneIntersection (planes[0], planes[1], cut_line1, 0.1);

    Eigen::VectorXf cut_line2;
    bool ok2 = pcl::planeWithPlaneIntersection (planes[0], planes[2], cut_line2, 0.1);

    if (!ok1 || !ok2)
    {
      std::cout << "cut failed" << std::endl;
    }
    else
    {
      std::cout << std::endl << "cut line 1:" << std::endl << cut_line1 << std::endl;
      std::cout << std::endl << "cut line 2:" << std::endl << cut_line2 << std::endl;

      Eigen::Vector4f corner;
      bool ok3 = pcl::lineWithLineIntersection (cut_line1, cut_line2, corner, 1e-4);

      if (!ok3)
      {
        std::cout << std::endl << "finding corner failed" << std::endl;
      }
      else
      {
        std::cout << std::endl << "corner point:" << std::endl << corner << std::endl;

        pcl::PointXYZ corner_point (corner[0], corner[1], corner[2]);

        pcl::PointXYZRGB corner_point_color(255,0,0);
        corner_point_color.x = corner_point.x;
        corner_point_color.y = corner_point.y;
        corner_point_color.z = corner_point.z;

        pcl::PointCloud<pcl::PointXYZRGB> corner_cloud;
        corner_cloud.push_back (corner_point_color);
        pcl::io::savePCDFile ("cloud_corners.pcd", corner_cloud, false);
      }
    }
  }


  return (0);
}
