#include <iostream>
#include <stdexcept>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

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
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("cloud_downsampled.pcd", *cloud_filtered, false);

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

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);
    std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

    std::stringstream ss;
    ss << "cloud_plane_" << i << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_p, false);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
  }

  return (0);
}
