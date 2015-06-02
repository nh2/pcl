/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */


#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <vector>
#include <cassert>
#include <string>

#include <pcl/console/parse.h>

#include <boost/filesystem.hpp>

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/raycaster.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/tcp_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/exceptions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/simplification_remove_unused_vertices.h>
#include <pcl/common/transforms.h>

#include <pcl/visualization/point_cloud_color_handlers.h>
#include "evaluation.h"

#include <pcl/common/angles.h>

#include "tsdf_volume.h"
#include "tsdf_volume.hpp"

#include "camera_pose.h"

#include <assert.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>

#ifdef HAVE_OPENCV  
  #include <opencv2/highgui/highgui.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
//#include "video_recorder.h"
#endif
typedef pcl::ScopeTime ScopeTimeT;

#include "../src/internal.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

int fixPlyFile(const char *meshIn, const char *meshOut);


namespace pcl
{
  namespace gpu
  {
    void paint3DView (const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
    void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
  }

  namespace visualization
  {
    //////////////////////////////////////////////////////////////////////////////////////
    /** \brief RGB handler class for colors. Uses the data present in the "rgb" or "rgba"
      * fields from an additional cloud as the color at each point.
      * \author Anatoly Baksheev
      * \ingroup visualization
      */
    template <typename PointT>
    class PointCloudColorHandlerRGBCloud : public PointCloudColorHandler<PointT>
    {
      using PointCloudColorHandler<PointT>::capable_;
      using PointCloudColorHandler<PointT>::cloud_;

      typedef typename PointCloudColorHandler<PointT>::PointCloud::ConstPtr PointCloudConstPtr;
      typedef typename pcl::PointCloud<RGB>::ConstPtr RgbCloudConstPtr;

      public:
        typedef boost::shared_ptr<PointCloudColorHandlerRGBCloud<PointT> > Ptr;
        typedef boost::shared_ptr<const PointCloudColorHandlerRGBCloud<PointT> > ConstPtr;
        
        /** \brief Constructor. */
        PointCloudColorHandlerRGBCloud (const PointCloudConstPtr& cloud, const RgbCloudConstPtr& colors)
          : rgb_ (colors)
        {
          cloud_  = cloud;
          capable_ = true;
        }
              
        /** \brief Obtain the actual color for the input dataset as vtk scalars.
          * \param[out] scalars the output scalars containing the color for the dataset
          * \return true if the operation was successful (the handler is capable and 
          * the input cloud was given as a valid pointer), false otherwise
          */
        virtual bool
        getColor (vtkSmartPointer<vtkDataArray> &scalars) const
        {
          if (!capable_ || !cloud_)
            return (false);
         
          if (!scalars)
            scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
          scalars->SetNumberOfComponents (3);
            
          vtkIdType nr_points = vtkIdType (cloud_->points.size ());
          reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples (nr_points);
          unsigned char* colors = reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->GetPointer (0);
            
          // Color every point
          if (nr_points != int (rgb_->points.size ()))
            std::fill (colors, colors + nr_points * 3, static_cast<unsigned char> (0xFF));
          else
            for (vtkIdType cp = 0; cp < nr_points; ++cp)
            {
              int idx = cp * 3;
              colors[idx + 0] = rgb_->points[cp].r;
              colors[idx + 1] = rgb_->points[cp].g;
              colors[idx + 2] = rgb_->points[cp].b;
            }
          return (true);
        }

      private:
        virtual std::string 
        getFieldName () const { return ("additional rgb"); }
        virtual std::string 
        getName () const { return ("PointCloudColorHandlerRGBCloud"); }
        
        RgbCloudConstPtr rgb_;
    };
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
vector<string> getPcdFilesInDir(const string& directory)
{
  namespace fs = boost::filesystem;
  fs::path dir(directory);
 
  std::cout << "path: " << directory << std::endl;
  if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
    PCL_THROW_EXCEPTION (pcl::IOException, "No valid PCD directory given!\n");
    
  vector<string> result;
  fs::directory_iterator pos(dir);
  fs::directory_iterator end;           

  for(; pos != end ; ++pos)
    if (fs::is_regular_file(pos->status()) )
      if (fs::extension(*pos) == ".pcd")
      {
#if BOOST_FILESYSTEM_VERSION == 3
        result.push_back (pos->path ().string ());
#else
        result.push_back (pos->path ());
#endif
        cout << "added: " << result.back() << endl;
      }
    
  return result;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch
{          
  enum { EACH = 33 };
  SampledScopeTime(int& time_ms) : time_ms_(time_ms) {}
  ~SampledScopeTime()
  {
    static int i_ = 0;
    static boost::posix_time::ptime starttime_ = boost::posix_time::microsec_clock::local_time();
    time_ms_ += getTime ();
    if (i_ % EACH == 0 && i_)
    {
      boost::posix_time::ptime endtime_ = boost::posix_time::microsec_clock::local_time();
      cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )"
           << "( real: " << 1000.f * EACH / (endtime_-starttime_).total_milliseconds() << "fps )"  << endl;
      time_ms_ = 0;
      starttime_ = endtime_;
    }
    ++i_;
  }
private:    
  int& time_ms_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f 
getViewerPose (visualization::PCLVisualizer& viewer)
{
  Eigen::Affine3f pose = viewer.getViewerPose();
  Eigen::Matrix3f rotation = pose.linear();

  Matrix3f axis_reorder;  
  axis_reorder << 0,  0,  1,
                 -1,  0,  0,
                  0, -1,  0;

  rotation = rotation * axis_reorder;
  pose.linear() = rotation;
  return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename CloudT> void
writeCloudFile (int format, const CloudT& cloud);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename MergedT, typename PointT>
typename PointCloud<MergedT>::Ptr merge(const PointCloud<PointT>& points, const PointCloud<RGB>& colors)
{    
  typename PointCloud<MergedT>::Ptr merged_ptr(new PointCloud<MergedT>());
    
  pcl::copyPointCloud (points, *merged_ptr);      
  for (size_t i = 0; i < colors.size (); ++i)
    merged_ptr->points[i].rgba = colors.points[i].rgba;
      
  return merged_ptr;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<float4>& device_triangles)
{ 
  if (device_triangles.empty())
      return boost::shared_ptr<pcl::PolygonMesh>();

  vector<float4> dirty_triangles(device_triangles.size());
  device_triangles.download(dirty_triangles);

  vector<float4> triangles(dirty_triangles.size());
  for (int i = 0; i < dirty_triangles.size(); i += 3) {
    pcl::PointXYZRGBA pt1, pt2, pt3;
    pt1.rgb = dirty_triangles[i].w;
    pt2.rgb = dirty_triangles[i+1].w;
    pt3.rgb = dirty_triangles[i+2].w;
    if (pt1.a != 0 && pt2.a != 0 && pt3.a != 0) {
      triangles.push_back(dirty_triangles[i]);
      triangles.push_back(dirty_triangles[i+1]);
      triangles.push_back(dirty_triangles[i+2]);
    }
  }

  pcl::PointCloud<pcl::PointXYZRGB> cloud((int)triangles.size(), 1);



  for (int i = 0; i < triangles.size(); ++i)
  {
    cloud[i].x   = triangles[i].x;
    cloud[i].y   = triangles[i].y;
    cloud[i].z   = triangles[i].z;
    cloud[i].rgb = triangles[i].w;
  }

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() ); 
  pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);
      
  mesh_ptr->polygons.resize (triangles.size() / 3);
  for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
  {
    pcl::Vertices v;
    v.vertices.push_back(i*3+0);
    v.vertices.push_back(i*3+2);
    v.vertices.push_back(i*3+1);              
    mesh_ptr->polygons[i] = v;
  }    
  return mesh_ptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct CurrentFrameCloudView
{
  CurrentFrameCloudView() : cloud_device_ (480, 640), cloud_viewer_ ("Frame Cloud Viewer")
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

    cloud_viewer_.setBackgroundColor (0, 0, 0.15);
    cloud_viewer_.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 1);
    cloud_viewer_.addCoordinateSystem (1.0, "global");
    cloud_viewer_.initCameraParameters ();
    cloud_viewer_.setPosition (0, 500);
    cloud_viewer_.setSize (640, 480);
    cloud_viewer_.setCameraClipDistances (0.01, 10.01);
  }

  void
  show (const KinfuTracker& kinfu)
  {
    kinfu.getLastFrameCloud (cloud_device_);

    int c;
    cloud_device_.download (cloud_ptr_->points, c);
    cloud_ptr_->width = cloud_device_.cols ();
    cloud_ptr_->height = cloud_device_.rows ();
    cloud_ptr_->is_dense = false;

    cloud_viewer_.removeAllPointClouds ();
    cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
    cloud_viewer_.spinOnce ();
  }

  void
  setViewerPose (const Eigen::Affine3f& viewer_pose) {
    ::setViewerPose (cloud_viewer_, viewer_pose);
  }

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  DeviceArray2D<PointXYZ> cloud_device_;
  visualization::PCLVisualizer cloud_viewer_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct ImageView
{
  ImageView(int viz) : viz_(viz), paint_image_ (false), accumulate_views_ (false)
  {
    if (viz_)
    {
        viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
        // viewerDepth_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);

        viewerScene_->setWindowTitle ("View3D from ray tracing");
        viewerScene_->setPosition (0, 0);
        // viewerDepth_->setWindowTitle ("Kinect Depth stream");
        // viewerDepth_->setPosition (640, 0);
        //viewerColor_.setWindowTitle ("Kinect RGB stream");
    }
  }

  void
  showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool registration, Eigen::Affine3f* pose_ptr = 0)
  {
    if (pose_ptr)
    {
        raycaster_ptr_->run(kinfu.volume(), kinfu.colorVolume(), *pose_ptr);
        raycaster_ptr_->generateSceneView(view_device_);
    }
    else
      kinfu.getImage (view_device_);

    if (paint_image_ && registration && !pose_ptr)
    {
      colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
      paint3DView (colors_device_, view_device_);
    }


    int cols;
    view_device_.download (view_host_, cols);
    if (viz_)
        viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());    

    //viewerColor_.showRGBImage ((unsigned char*)&rgb24.data, rgb24.cols, rgb24.rows);

#ifdef HAVE_OPENCV
    if (accumulate_views_)
    {
      views_.push_back (cv::Mat ());
      cv::cvtColor (cv::Mat (480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back (), CV_RGB2GRAY);
      //cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
    }
#endif
  }

  void
  showDepth (const PtrStepSz<const unsigned short>& depth) 
  { 
     // if (viz_)
     //   viewerDepth_->showShortImage (depth.data, depth.cols, depth.rows, 0, 5000, true); 
  }
  
  void
  showGeneratedDepth (KinfuTracker& kinfu, const Eigen::Affine3f& pose)
  {            
    raycaster_ptr_->run(kinfu.volume(), kinfu.colorVolume(), pose);
    raycaster_ptr_->generateDepthImage(generated_depth_);    

    int c;
    vector<unsigned short> data;
    generated_depth_.download(data, c);

    // if (viz_)
    //     viewerDepth_->showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
  }

  void
  toggleImagePaint()
  {
    paint_image_ = !paint_image_;
    cout << "Paint image: " << (paint_image_ ? "On   (requires registration mode)" : "Off") << endl;
  }

  int viz_;
  bool paint_image_;
  bool accumulate_views_;

  visualization::ImageViewer::Ptr viewerScene_;
  visualization::ImageViewer::Ptr viewerDepth_;
  //visualization::ImageViewer viewerColor_;

  KinfuTracker::View view_device_;
  KinfuTracker::View colors_device_;
  vector<KinfuTracker::PixelRGB> view_host_;

  RayCaster::Ptr raycaster_ptr_;

  KinfuTracker::DepthMap generated_depth_;
  
#ifdef HAVE_OPENCV
  vector<cv::Mat> views_;
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SceneCloudView
{
  enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

  SceneCloudView(int viz) : viz_(viz), extraction_mode_ (GPU_Connected6), compute_normals_ (false), valid_combined_ (false), cube_added_(false)
  {
    cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
    normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
    combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
    point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);

    if (viz_)
    {
        cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("Scene Cloud Viewer") );

		// Hide cloud viewer
		HWND hWnd = (HWND)cloud_viewer_->getRenderWindow()->GetGenericWindowId();
		ShowWindow(hWnd, SW_HIDE);

        cloud_viewer_->setBackgroundColor (0, 0, 0);
        cloud_viewer_->addCoordinateSystem (1.0, "global");
        cloud_viewer_->initCameraParameters ();
        cloud_viewer_->setPosition (0, 500);
        cloud_viewer_->setSize (640, 480);
        cloud_viewer_->setCameraClipDistances (0.01, 10.01);

        cloud_viewer_->addText ("H: print help", 2, 15, 20, 34, 135, 246);
    }
  }

  void
  show (KinfuTracker& kinfu, bool integrate_colors)
  {
    viewer_pose_ = kinfu.getCameraPose();

    ScopeTimeT time ("PointCloud Extraction");
    cout << "\nGetting cloud... " << flush;

    valid_combined_ = false;

    if (extraction_mode_ != GPU_Connected6)     // So use CPU
    {
      kinfu.volume().fetchCloudHost (*cloud_ptr_, extraction_mode_ == CPU_Connected26);
    }
    else
    {
      std::cout << "extracting volume_.getSize() " << kinfu.volume().getSize() << std::endl;
      std::cout << "extracting volume_.getResolution() " << kinfu.volume().getResolution() << std::endl;
      std::cout << "extracting volume_.getTsdfTruncDist() " << kinfu.volume().getTsdfTruncDist() << std::endl;

      DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_);             

      if (compute_normals_)
      {
        kinfu.volume().fetchNormals (extracted, normals_device_);
        pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
        combined_device_.download (combined_ptr_->points);
        combined_ptr_->width = (int)combined_ptr_->points.size ();
        combined_ptr_->height = 1;

        valid_combined_ = true;
      }
      else
      {
        extracted.download (cloud_ptr_->points);
        cloud_ptr_->width = (int)cloud_ptr_->points.size ();
        cloud_ptr_->height = 1;
      }

      if (integrate_colors)
      {
        kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
        point_colors_device_.download(point_colors_ptr_->points);
        point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
        point_colors_ptr_->height = 1;
      }
      else
        point_colors_ptr_->points.clear();
    }
    cout << "valid_combined_ " << valid_combined_ << endl;
    size_t points_size = valid_combined_ ? combined_ptr_->points.size () : cloud_ptr_->points.size ();
    cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

    if (viz_)
    {
        cloud_viewer_->removeAllPointClouds ();
        if (valid_combined_)
        {
          visualization::PointCloudColorHandlerRGBCloud<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
          cloud_viewer_->addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
          cloud_viewer_->addPointCloudNormals<PointNormal>(combined_ptr_, 50);
        }
        else
        {
          visualization::PointCloudColorHandlerRGBCloud<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
          cloud_viewer_->addPointCloud<PointXYZ> (cloud_ptr_, rgb);
        }
    }
  }

  void
  toggleCube(const Eigen::Vector3f& size)
  {
      if (!viz_)
          return;

      if (cube_added_)
          cloud_viewer_->removeShape("cube");
      else
        cloud_viewer_->addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

      cube_added_ = !cube_added_;
  }

  void
  toggleExtractionMode ()
  {
    extraction_mode_ = (extraction_mode_ + 1) % 3;

    switch (extraction_mode_)
    {
    case 0: cout << "Cloud exctraction mode: GPU, Connected-6" << endl; break;
    case 1: cout << "Cloud exctraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
    case 2: cout << "Cloud exctraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
    }
    ;
  }

  void
  toggleNormals ()
  {
    compute_normals_ = !compute_normals_;
    cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
  }

  void
  clearClouds (bool print_message = false)
  {
    if (!viz_)
        return;

    cloud_viewer_->removeAllPointClouds ();
    cloud_ptr_->points.clear ();
    normals_ptr_->points.clear ();    
    if (print_message)
      cout << "Clouds/Meshes were cleared" << endl;
  }

  void
  showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
  {
    if (!viz_)
       return;

    ScopeTimeT time ("Mesh Extraction");
    cout << "\nGetting mesh... " << flush;

    if (!marching_cubes_)
      marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

    PtrStep<uchar4> color_volume_data = kinfu.colorVolume().data();
    const uchar4 *colors = color_volume_data.data;

    DeviceArray<float4> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_, colors);
    mesh_ptr_ = convertToMesh(triangles_device);
    
    cloud_viewer_->removeAllPointClouds ();
    if (mesh_ptr_)
      cloud_viewer_->addPolygonMesh(*mesh_ptr_);
    
    cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
  }
    
  int viz_;
  int extraction_mode_;
  bool compute_normals_;
  bool valid_combined_;
  bool cube_added_;

  Eigen::Affine3f viewer_pose_;

  visualization::PCLVisualizer::Ptr cloud_viewer_;

  PointCloud<PointXYZ>::Ptr cloud_ptr_;
  PointCloud<Normal>::Ptr normals_ptr_;

  DeviceArray<PointXYZ> cloud_buffer_device_;
  DeviceArray<Normal> normals_device_;

  PointCloud<PointNormal>::Ptr combined_ptr_;
  DeviceArray<PointNormal> combined_device_;  

  DeviceArray<RGB> point_colors_device_; 
  PointCloud<RGB>::Ptr point_colors_ptr_;

  MarchingCubes::Ptr marching_cubes_;
  DeviceArray<float4> triangles_buffer_device_;

  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KinFuApp
{
  enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };

  static const int default_max_color_integration_weight = 2;

  KinFuApp(pcl::Grabber& source, float vsz, int icp, int viz, boost::shared_ptr<CameraPoseProcessor> pose_processor=boost::shared_ptr<CameraPoseProcessor> (), bool start_at_side = false) : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), volume_scanned_(false), independent_camera_ (false),
    registration_ (false), integrate_colors_ (false), focal_length_(-1.f), capture_ (source), scene_cloud_view_(viz), image_view_(viz), time_ms_(0), icp_(icp), viz_(viz), pose_processor_ (pose_processor)
  {    
    //Init Kinfu Tracker
    Eigen::Vector3f volume_size = Vector3f::Constant (vsz/*meters*/);    
    kinfu_.volume().setSize (volume_size);

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());

    Eigen::Vector3f t;
    if (start_at_side)
    {
      // Start at the center of one side of the scanning volume
      t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);
    } else {
      // Start in center of the scanning volume
      t = volume_size * 0.5f;
    }

    Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

    kinfu_.setInitalCameraPose (pose);
    kinfu_.volume().setTsdfTruncDist (0.030f/*meters*/);    
    kinfu_.setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
    //kinfu_.setDepthTruncationForICP(5.f/*meters*/);
    kinfu_.setCameraMovementThreshold(0.001f);

    if (!icp)
      kinfu_.disableIcp();

    
    //Init KinfuApp            
    tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr (new pcl::PointCloud<pcl::PointXYZI>);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols ()) );

    if (viz_)
    {
        scene_cloud_view_.cloud_viewer_->registerKeyboardCallback (keyboard_callback, (void*)this);
        image_view_.viewerScene_->registerKeyboardCallback (keyboard_callback, (void*)this);
        // image_view_.viewerDepth_->registerKeyboardCallback (keyboard_callback, (void*)this);

        // scene_cloud_view_.toggleCube(volume_size);
    }
  }

  ~KinFuApp()
  {
    if (evaluation_ptr_)
      evaluation_ptr_->saveAllPoses(kinfu_);
  }

  void parseConfig(char *configPath) {
	  FILE* fp;
	  fp = fopen(configPath, "r");

	  char tag[1024];
	  float val;
	  int res;

	  while (1) {
		  res = fscanf(fp, "%s", &tag);
		  res = fscanf(fp, "%f", &val);
		  if (res == EOF) break;

		  if (!strcmp(tag, "camera_fx"))
			  camera_fx = val;
		  else if (!strcmp(tag, "camera_fy"))
			  camera_fy = val;
		  else if (!strcmp(tag, "principal_cx"))
			  principal_cx = val;
		  else if (!strcmp(tag, "principal_cy"))
			  principal_cy = val;
		  else if (!strcmp(tag, "size_multiplier"))
			  size_multiplier = val;
		  else if (!strcmp(tag, "crop_from_nose_mm_y"))
			  crop_from_nose_mm_y = val;
		  else if (!strcmp(tag, "crop_from_nose_mm_z"))
			  crop_from_nose_mm_z = val;
		  else if (!strcmp(tag, "radius_from_middle"))
			  radius_from_middle = val;
		  else if (!strcmp(tag, "nose_y_displacement"))
			  nose_y_displacement = val;
          else if (!strcmp(tag, "accept_angle_deg"))
              accept_angle_deg = val;
		  else
			  std::cerr << "WARNING: invalid configuration entry " << tag << std::endl;
	  }
	  fclose(fp);

	  std::cout << "Focal length " << camera_fx << ", " << camera_fy << std::endl;
	  std::cout << "Principal point " << principal_cx << ", " << principal_cy << std::endl;
	  std::cout << "Size multiplier " << size_multiplier << std::endl;
	  std::cout << "Crop from nose: " << crop_from_nose_mm_y << "(y), " << crop_from_nose_mm_z << "(z)" << std::endl;
  }

  void
  initCurrentFrameView ()
  {
    current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView ());
    current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
    current_frame_cloud_view_->setViewerPose (kinfu_.getCameraPose ());
  }

  void
  initRegistration ()
  {        
    registration_ = capture_.providesCallback<pcl::ONIGrabber::sig_cb_openni_image_depth_image> ();
    cout << "Registration mode: " << (registration_ ? "On" : "Off (not supported by source)") << endl;
    if (registration_)
      kinfu_.setDepthIntrinsics(camera_fx, camera_fy, principal_cx, principal_cy);
  }
  
  void
  setDepthIntrinsics(std::vector<float> depth_intrinsics)
  {
    float fx = depth_intrinsics[0];
    float fy = depth_intrinsics[1];
    
    if (depth_intrinsics.size() == 4)
    {
        float cx = depth_intrinsics[2];
        float cy = depth_intrinsics[3];
        kinfu_.setDepthIntrinsics(fx, fy, cx, cy);
        cout << "Depth intrinsics changed to fx="<< fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << endl;
    }
    else {
        kinfu_.setDepthIntrinsics(fx, fy);
        cout << "Depth intrinsics changed to fx="<< fx << " fy=" << fy << endl;
    }
  }

  void 
  toggleColorIntegrationWithoutRegistration()
  {
    kinfu_.initColorIntegration(default_max_color_integration_weight);
    integrate_colors_ = true;      
  }
  

  void 
  toggleColorIntegration()
  {
    // if(registration_)
    // {
      kinfu_.initColorIntegration(default_max_color_integration_weight);
      integrate_colors_ = true;      
    // }
    cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
  }
  
  void 
  enableTruncationScaling()
  {
    kinfu_.volume().setTsdfTruncDist (kinfu_.volume().getSize()(0) / 100.0f);
  }

  void
  toggleIndependentCamera()
  {
    independent_camera_ = !independent_camera_;
    cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
  }
  
  std::string getDate()
  {
	  time_t rawtime;
	  struct tm *timeinfo;
	  char buffer[80];

	  time(&rawtime);
	  timeinfo = localtime(&rawtime);

	  strftime(buffer, 80, "%Y%m%d%H%M%S", timeinfo);
	  std::string str(buffer);

	  return str;
  }

  void copyFile(std::string srcFn, std::string dstFn) {
	  std::ifstream src(srcFn, std::ios::binary);
	  std::ofstream dst(dstFn, std::ios::binary);
	  dst << src.rdbuf();
  }

  PointXYZRGB findNose(pcl::PointCloud<PointXYZRGB> meshcloud_sane)
  {

	float top_y = -1000000000000000000, bot_y = 1000000000000000000, left_x = 1000000000000000000, right_x = -1000000000000000000;

	for (int i = 0; i < meshcloud_sane.size(); ++i) {
		PointXYZRGB pt = meshcloud_sane[i];
		top_y = max(top_y, pt.y);
		bot_y = min(bot_y, pt.y);
		left_x = min(left_x, pt.x);
		right_x = max(right_x, pt.x);
	}

	float closest_z = -1000000000000000000;
	for (int i = 0; i < meshcloud_sane.size(); ++i) {
		PointXYZRGB pt = meshcloud_sane[i];
		if (pt.z > closest_z) {
			closest_z = pt.z;
		}
	}

	PointXYZRGB middle;
	middle.x = (left_x + right_x) / 2;
	middle.y = nose_y_displacement + (top_y + bot_y) / 2;
	middle.z = closest_z;

	PointXYZRGB nose = middle;
	float closest_z_in_circle = -1000000000000000000;
	float squared_radius = radius_from_middle*radius_from_middle;
	for (int i = 0; i < meshcloud_sane.size(); ++i) {
		PointXYZRGB pt = meshcloud_sane[i];
		if (pt.z > closest_z_in_circle) {
			float diff_x = middle.x - pt.x;
			float diff_y = middle.y - pt.y;
			bool in_radius = (diff_x*diff_x + diff_y*diff_y) < squared_radius;
			if (in_radius) {
				closest_z_in_circle = pt.z;
				nose = pt;
			}
		}
	}

	return nose;
  }

  pcl::PointCloud<PointXYZRGB> cropFaceFromNose(PointCloud<PointXYZRGB> meshcloud_sane, PointXYZRGB nose)
  {
    // Cut back of head
    pcl::PointCloud<PointXYZRGB>::ConstPtr meshCloudPtr = boost::make_shared<pcl::PointCloud<PointXYZRGB> >(meshcloud_sane);
    pcl::PassThrough<PointXYZRGB> pass;
    pass.setInputCloud(meshCloudPtr);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(nose.z - crop_from_nose_mm_z, 1000000); // TODO scale with distance
    pcl::PointCloud<PointXYZRGB>::Ptr cloud_filtered_ptr(new pcl::PointCloud<PointXYZRGB>);
    pass.setKeepOrganized(true);
    pass.setUserFilterValue(0.0);
    pass.filter(*cloud_filtered_ptr);
    pcl::PointCloud<PointXYZRGB> cloud_filtered = *cloud_filtered_ptr;

    // float magic = 9999999.9;
    float magic = 0.0;

    // Cut top of head
    pcl::PointCloud<PointXYZRGB>::ConstPtr meshCloudPtr2 = boost::make_shared<pcl::PointCloud<PointXYZRGB> >(cloud_filtered);
    pcl::PassThrough<PointXYZRGB> pass2;
    pass2.setInputCloud(meshCloudPtr2);
    pass2.setFilterFieldName("y");
    pass2.setFilterLimits(-100000, nose.y + crop_from_nose_mm_y);
    pcl::PointCloud<PointXYZRGB>::Ptr cloud_filtered_ptr2(new pcl::PointCloud<PointXYZRGB>);
    pass2.setKeepOrganized(true);
    pass2.setUserFilterValue(magic);
    pass2.filter(*cloud_filtered_ptr2);

    return *cloud_filtered_ptr2;
  }

  void scaleMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr before, pcl::PointCloud<pcl::PointXYZRGB>::Ptr after)
  {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 0) = size_multiplier;
    transform(1, 1) = size_multiplier;
    transform(2, 2) = size_multiplier;
    //    (row, column)
    pcl::transformPointCloud(*before, *after, transform);
  }

  std::vector<pcl::Vertices> cleanPolygons(pcl::PointCloud<PointXYZRGB> meshcloud_sane, std::vector<pcl::Vertices> polygons)
  {
    float magic = 0.0;
    std::vector<pcl::Vertices> new_polygons;

    for (int i = 0; i < polygons.size(); ++i)
    {
      pcl::Vertices poly = polygons[i];
      bool acceptable = true;
      for (int k = 0; k < poly.vertices.size(); ++k)
      {
        uint32_t ix = poly.vertices[k];
        if (meshcloud_sane[ix].x == magic) {
          acceptable = false;
          break;
        }
      }
      if (acceptable) {
        new_polygons.push_back(poly);
      }
    }

    return new_polygons;
  }

  void translateMesh(PointXYZRGB nose, pcl::PointCloud<pcl::PointXYZRGB>::Ptr before, pcl::PointCloud<pcl::PointXYZRGB>::Ptr after)
  {
    Eigen::Affine3f transform_translate = Eigen::Affine3f::Identity();
    transform_translate.translation() << -nose.x, -nose.y, -nose.z;
    pcl::transformPointCloud(*before, *after, transform_translate);
  }

  void nonEmptyOrExit(pcl::PointCloud<pcl::PointXYZRGB> mesh) {
    if (mesh.size() == 0) {
      cout << "Empty mesh, exiting" << endl;
      exit(0);
    }
  }

  void nonEmptyOrExit2(pcl::PCLPointCloud2 cloud) {
    if (cloud.data.size() == 0) {
      cout << "Empty cloud 2, exiting" << endl;
      exit(0);
    }
  }

  void
  prepareMesh()
  {
    if (!std::ifstream("mesh.ply")) {
      cout << "mesh.ply file does not exist, exiting" << endl;
      exit(0);
    }

    pcl::PointCloud<PointXYZRGB> meshcloud_sane;

    // Copy mesh to dated file
	std::string date = getDate();
	copyFile("mesh.ply", "mesh-" + date + ".ply");
    
    std::cout << "Cleaning and rotating mesh with meshlabserver" << std::endl;
    int res = 0;
	res = system("meshlabserver -i mesh.ply -o mesh-clean.ply -s kinfu\\clean.mlx -om vc vn");
	assert(res == 0);
    copyFile("mesh-clean.ply", "mesh-" + date + "-clean.ply");

    // Converting to ASCII
    res = system("python ply2asc\\ply2asc.py mesh-clean.ply mesh-rotated.asc.ply");
	assert(res == 0);
    copyFile("mesh-rotated.asc.ply", "mesh-" + date + "-rotated.asc.ply");

    // Load it back to PCL
    pcl::PolygonMesh mesh;
    pcl::io::loadPLYFile("mesh-rotated.asc.ply", mesh);
    fromPCLPointCloud2(mesh.cloud, meshcloud_sane);
    nonEmptyOrExit(meshcloud_sane);

    // Scale it
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr before_scaling(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr after_scaling(new pcl::PointCloud<pcl::PointXYZRGB>());
    *before_scaling = meshcloud_sane;
    scaleMesh(before_scaling, after_scaling);
    meshcloud_sane = *after_scaling;
    cout << "Scaled size " << meshcloud_sane.size() << endl;
    nonEmptyOrExit(meshcloud_sane);

    // Crop face from nose
    cout << "Cropping face from nose" << endl;
    PointXYZRGB nose = findNose(meshcloud_sane);
    meshcloud_sane = cropFaceFromNose(meshcloud_sane, nose);
    cout << "Cropped size " << meshcloud_sane.size() << endl;
    nonEmptyOrExit(meshcloud_sane);

    // Clean polygons
    cout << "Cleaning polygons" << endl;
    std::vector<pcl::Vertices> new_polygons = cleanPolygons(meshcloud_sane, mesh.polygons);

    // Remove unused vertices
    cout << "Remove unused vertices" << endl;
    toPCLPointCloud2(meshcloud_sane, mesh.cloud);
    mesh.polygons = new_polygons;
    pcl::PolygonMesh mesh_cleaned;
    pcl::surface::SimplificationRemoveUnusedVertices simplifier;
    simplifier.simplify(mesh, mesh_cleaned);
    nonEmptyOrExit2(mesh_cleaned.cloud);
    fromPCLPointCloud2(mesh_cleaned.cloud, meshcloud_sane);
    nonEmptyOrExit(meshcloud_sane);

    // Translate
    cout << "Translate mesh" << endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr before_translating (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr after_translating (new pcl::PointCloud<pcl::PointXYZRGB> ());
    *before_translating = meshcloud_sane;
    translateMesh(nose, before_translating, after_translating);
    meshcloud_sane = *after_translating;
    nonEmptyOrExit(meshcloud_sane);

    toPCLPointCloud2(meshcloud_sane, mesh_cleaned.cloud);

    // Final pass with meshlabserver
    cout << "Final meshlab pass" << endl;
    pcl::io::savePLYFile("mesh-cropped-uncleaned.ply", mesh_cleaned);
    copyFile("mesh-cropped-uncleaned.ply", "mesh-" + date + "-cropped-uncleaned.ply");
    res = system("meshlabserver -i mesh-cropped-uncleaned.ply -o mesh-cropped.ply -s kinfu\\clean-small.mlx -om vc vn");
    assert(res == 0);
    copyFile("mesh-cropped.ply", "mesh-" + date + "-cropped.ply");

	copyFile("mesh-cropped.ply", "target.ply");
  }

  void
  processEtronStream()
  {
    // Remove existing mesh file, if it exists.
    if (std::ifstream("mesh.ply")) {
      int res = remove("mesh.ply");
      assert(res == 0);
    }

    scene_cloud_view_.showMesh(kinfu_, integrate_colors_);
    writeMesh ((int)'7' - (int)'0');
    prepareMesh();
  }

  void
  processEtronStreamAndQuit()
  {
	  processEtronStream();
	  exit(0);
  }

  void
  toggleEvaluationMode(const string& eval_folder, const string& match_file = string())
  {
    evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );
    if (!match_file.empty())
        evaluation_ptr_->setMatchFile(match_file);

    kinfu_.setDepthIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
    image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols (), 
        evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy) );
  }

  void
  loadTsdf(const string& tsdf_file)
  {
    tsdf_volume_.load (tsdf_file, true);
    cout << "Loaded " << tsdf_volume_.size () << " voxels into host volume" << endl;

    cout << "Uploading TSDF volume to GPU ...";
    kinfu_.volume().uploadTsdfAndWeighs (tsdf_volume_.volume (), tsdf_volume_.weights ());
    cout << "done" << endl;
  }

  void
  loadColor(const string& color_file)
  {
    pcl::console::print_info ("Loading color volume from "); pcl::console::print_value ("%s ... ", color_file.c_str());
    std::cout << std::flush;

    std::ifstream file (color_file.c_str(), std::ios_base::binary);

    if (file.is_open())
    {
      int num_elements = 512*512*512; // TODO nh2 remove hardcode
      host_voxel_colors.resize (num_elements);
      file.read ((char*) &host_voxel_colors[0], num_elements * sizeof(uint32_t));
      file.close ();
      pcl::console::print_info ("done\n");
    }
    else
    {
      pcl::console::print_error ("[loadColor] Error: Couldn't read file %s.\n", color_file.c_str());
      throw new runtime_error("loadColor failed reading file");
    }

    cout << "Uploading color volume to GPU ...";
    kinfu_.colorVolume().uploadVoxelColors (host_voxel_colors);
    cout << "done" << endl;
  }

  void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data)
  {        
    bool has_image = false;
      
    if (has_data)
    {
      depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
      if (integrate_colors_)
          image_view_.colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
    
      {
        SampledScopeTime fps(time_ms_);
    
        //run kinfu algorithm
        if (integrate_colors_)
          has_image = kinfu_ (depth_device_, image_view_.colors_device_);
        else
          has_image = kinfu_ (depth_device_);                  
      }

      // process camera pose
      if (pose_processor_)
      {
        pose_processor_->processPose (kinfu_.getCameraPose ());
      }

      image_view_.showDepth (depth);
      //image_view_.showGeneratedDepth(kinfu_, kinfu_.getCameraPose());
    }

    if (scan_)
    {
      scan_ = false;
      scene_cloud_view_.show (kinfu_, integrate_colors_);
                    
      if (scan_volume_)
      {                
        cout << "Downloading TSDF volume from device ... " << flush;
        kinfu_.volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
        tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_.volume().getSize ());
        cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;
                
        volume_scanned_ = true;

        cout << "Converting volume to TSDF cloud ... " << flush;
        tsdf_volume_.convertToTsdfCloud (tsdf_cloud_ptr_);
        cout << "done [" << tsdf_cloud_ptr_->size () << " points]" << endl << endl;        

        if (integrate_colors_)
        {
          cout << "Downloading color volume from device ... " << flush;
          kinfu_.colorVolume().downloadVoxelColors (host_voxel_colors);
          cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;
        }
      }
      else
        cout << "[!] tsdf volume download is disabled" << endl << endl;
    }

    if (scan_mesh_)
    {
        scan_mesh_ = false;
        scene_cloud_view_.showMesh(kinfu_, integrate_colors_);
    }
     
    if (has_image)
    {
      Eigen::Affine3f viewer_pose = getViewerPose(*scene_cloud_view_.cloud_viewer_);
      image_view_.showScene (kinfu_, rgb24, registration_, independent_camera_ ? &viewer_pose : 0);
    }    

    if (current_frame_cloud_view_)
      current_frame_cloud_view_->show (kinfu_);    
      
    if (!independent_camera_)
      setViewerPose (*scene_cloud_view_.cloud_viewer_, kinfu_.getCameraPose());
  }
  
  void source_cb_tcp(bool continue_, const boost::array<unsigned char, 640*480*3> &rgb_buf, const boost::array<unsigned short, 640*480>& depth_buf)
  {
    if (!continue_)
	{
			processEtronStream();
			exit(0);
    }

    {
      boost::mutex::scoped_try_lock lock(data_ready_mutex_);
      if (exit_ || !lock)
          return;

      rgb24_.cols = 640;
      rgb24_.rows = 480;
      rgb24_.step = rgb24_.cols * 3;

      source_image_data_.resize(rgb24_.cols * rgb24_.rows);
      for (int i = 0; i < (640*480); ++i)
      {
          // make everything white
          PixelRGB x = { rgb_buf[i*3], rgb_buf[i*3+1], rgb_buf[i*3+2] };
          source_image_data_[i] = x;
      }
      rgb24_.data = &source_image_data_[0];           

      depth_.cols = 640;
      depth_.rows = 480;
      depth_.step = depth_.cols * 2; // we want 16 bits depth data per pixel

      source_depth_data_.resize(depth_.cols * depth_.rows);

      // TODO use assignment operator
      for (int i = 0; i < depth_buf.size(); ++i)
      {
        source_depth_data_[i] = depth_buf[i];
      }

      depth_.data = &source_depth_data_[0];

    }
    data_ready_cond_.notify_one();
  }

  void source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)  
  {        
    {
      boost::mutex::scoped_try_lock lock(data_ready_mutex_);
      if (exit_ || !lock)
          return;
      
      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];     
    }
    data_ready_cond_.notify_one();
  }

  void source_cb2_device(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
  {
    {
      boost::mutex::scoped_try_lock lock(data_ready_mutex_);
      if (exit_ || !lock)
          return;
                  
      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];      
      
      // openni_wrapper::Image::Encoding enc = image_wrapper->getEncoding();
      // printf("nh2: source_cb2_device image_wrapper encoding: %s\n",
      //   enc == openni_wrapper::Image::BAYER_GRBG ? "BAYER_GRBG" :
      //   enc == openni_wrapper::Image::YUV422     ? "YUV422"     :
      //   enc == openni_wrapper::Image::RGB        ? "RGB"        : "UNKNOWN_ENCODING");

      rgb24_.cols = image_wrapper->getWidth();
      rgb24_.rows = image_wrapper->getHeight();
      rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 

      source_image_data_.resize(rgb24_.cols * rgb24_.rows);
      image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
      rgb24_.data = &source_image_data_[0];           
    }
    data_ready_cond_.notify_one();
  }


   void source_cb1_oni(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)  
  {        
    {
      boost::mutex::scoped_lock lock(data_ready_mutex_);
      if (exit_)
          return;
      
      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];     
    }
    data_ready_cond_.notify_one();
  }

  void source_cb2_oni(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
  {
    {
      boost::mutex::scoped_lock lock(data_ready_mutex_);
      if (exit_)
          return;
                  
      depth_.cols = depth_wrapper->getWidth();
      depth_.rows = depth_wrapper->getHeight();
      depth_.step = depth_.cols * depth_.elemSize();

      source_depth_data_.resize(depth_.cols * depth_.rows);
      depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
      depth_.data = &source_depth_data_[0];      
      
      // openni_wrapper::Image::Encoding enc = image_wrapper->getEncoding();
      // printf("nh2: source_cb2_oni image_wrapper encoding: %s\n",
      //   enc == openni_wrapper::Image::BAYER_GRBG ? "BAYER_GRBG" :
      //   enc == openni_wrapper::Image::YUV422     ? "YUV422"     :
      //   enc == openni_wrapper::Image::RGB        ? "RGB"        : "UNKNOWN_ENCODING");

      rgb24_.cols = image_wrapper->getWidth();
      rgb24_.rows = image_wrapper->getHeight();
      rgb24_.step = rgb24_.cols * rgb24_.elemSize(); 

      source_image_data_.resize(rgb24_.cols * rgb24_.rows);
      image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
      rgb24_.data = &source_image_data_[0];           
    }
    data_ready_cond_.notify_one();
  }

  void
  source_cb3 (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr & DC3)
  {
    {
      boost::mutex::scoped_try_lock lock(data_ready_mutex_);
      if (exit_ || !lock)
        return;
      int width  = DC3->width;
      int height = DC3->height;
      depth_.cols = width;
      depth_.rows = height;
      depth_.step = depth_.cols * depth_.elemSize ();
      source_depth_data_.resize (depth_.cols * depth_.rows);

      rgb24_.cols = width;
      rgb24_.rows = height;
      rgb24_.step = rgb24_.cols * rgb24_.elemSize ();
      source_image_data_.resize (rgb24_.cols * rgb24_.rows);

      unsigned char *rgb    = (unsigned char *)  &source_image_data_[0];
      unsigned short *depth = (unsigned short *) &source_depth_data_[0];

      for (int i=0; i < width*height; i++) 
      {
        PointXYZRGBA pt = DC3->at (i);
        rgb[3*i +0] = pt.r;
        rgb[3*i +1] = pt.g;
        rgb[3*i +2] = pt.b;
        depth[i]    = pt.z/0.001;
      }
      rgb24_.data = &source_image_data_[0];
      depth_.data = &source_depth_data_[0];
    }
    data_ready_cond_.notify_one ();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  startDisplayOnlyMainLoop ()
  {
    bool scene_view_not_stopped= viz_ ? !scene_cloud_view_.cloud_viewer_->wasStopped () : true;
    bool image_view_not_stopped= viz_ ? !image_view_.viewerScene_->wasStopped () : true;

    while (!exit_ && scene_view_not_stopped && image_view_not_stopped)
    { 
      try { this->execute (depth_, rgb24_, false); }
      catch (const std::bad_alloc& e) { cout << "Bad alloc: " << e.what() << endl; break; }
      catch (const std::exception& e) { cout << "Exception: " << e.what() << endl; break; }
      
      if (viz_)
          scene_cloud_view_.cloud_viewer_->spinOnce (3);
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  startMainLoop (bool triggered_capture)
  {   
    using namespace openni_wrapper;
    typedef boost::shared_ptr<DepthImage> DepthImagePtr;
    typedef boost::shared_ptr<Image> ImagePtr;
        
    // boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1_dev = boost::bind (&KinFuApp::source_cb2_device, this, _1, _2, _3);
    // boost::function<void (const DepthImagePtr&)> func2_dev = boost::bind (&KinFuApp::source_cb1_device, this, _1);
    boost::function<void (bool, const boost::array<unsigned char, 640*480*3> &, const boost::array<unsigned short, 640*480>&)> tcp_func = boost::bind (&KinFuApp::source_cb_tcp, this, _1, _2, _3);

    // boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1_oni = boost::bind (&KinFuApp::source_cb2_oni, this, _1, _2, _3);
    // boost::function<void (const DepthImagePtr&)> func2_oni = boost::bind (&KinFuApp::source_cb1_oni, this, _1);
    
    // bool is_oni = dynamic_cast<pcl::ONIGrabber*>(&capture_) != 0;
    // boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1 = is_oni ? func1_oni : func1_dev;
    // boost::function<void (const DepthImagePtr&)> func2 = is_oni ? func2_oni : func2_dev;

    // boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&) > func3 = boost::bind (&KinFuApp::source_cb3, this, _1);

    bool need_colors = integrate_colors_ || registration_;
    if ( pcd_source_ && !capture_.providesCallback<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)>() )
    {
      std::cout << "grabber doesn't provide pcl::PointCloud<pcl::PointXYZRGBA> callback !\n";
    }
    // boost::signals2::connection c = pcd_source_? capture_.registerCallback (func3) : need_colors ? capture_.registerCallback (func1) : capture_.registerCallback (func2);
    // boost::signals2::connection c = need_colors ? capture_.registerCallback (func1) : capture_.registerCallback (func2);
    // TODO also do the color callback, with need_colors check
    boost::signals2::connection c = capture_.registerCallback (tcp_func);

    {
      boost::unique_lock<boost::mutex> lock(data_ready_mutex_);

      if (!triggered_capture)
          capture_.start (); // Start stream

      bool scene_view_not_stopped= viz_ ? !scene_cloud_view_.cloud_viewer_->wasStopped () : true;
      bool image_view_not_stopped= viz_ ? !image_view_.viewerScene_->wasStopped () : true;
          
      while (!exit_ && scene_view_not_stopped && image_view_not_stopped)
      { 
        if (triggered_capture)
            capture_.start(); // Triggers new frame
        bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(100));        
                       
        try { this->execute (depth_, rgb24_, has_data); }
        catch (const std::bad_alloc& e) { cout << "Bad alloc: " << e.what() << endl; break; }
        catch (const std::exception& e) { cout << "Exception: " << e.what() << endl; break; }
        
        if (viz_)
            scene_cloud_view_.cloud_viewer_->spinOnce (3);
      }
      
      if (!triggered_capture)     
          capture_.stop (); // Stop stream
    }
    c.disconnect();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  writeCloud (int format) const
  {      
    const SceneCloudView& view = scene_cloud_view_;

    // Points to export are either in cloud_ptr_ or combined_ptr_.
    // If none have points, we have nothing to export.
    if (view.cloud_ptr_->points.empty () && view.combined_ptr_->points.empty ())
    {
      cout << "Not writing cloud: Cloud is empty" << endl;
    }
    else
    {
      if(view.point_colors_ptr_->points.empty()) // no colors
      {
        if (view.valid_combined_)
          writeCloudFile (format, view.combined_ptr_);
        else
          writeCloudFile (format, view.cloud_ptr_);
      }
      else
      {        
        if (view.valid_combined_)
          writeCloudFile (format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_));
        else
          writeCloudFile (format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_));
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  writeMesh(int format) const
  {
    if (scene_cloud_view_.mesh_ptr_) 
      writePolygonMeshFile(format, *scene_cloud_view_.mesh_ptr_);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  void
  printHelp ()
  {
    cout << endl;
    cout << "KinFu app hotkeys" << endl;
    cout << "=================" << endl;
    cout << "    H    : print this help" << endl;
    cout << "   Esc   : exit" << endl;
    cout << "    R    : reset" << endl;
    cout << "    T    : take cloud" << endl;
    cout << "    A    : take mesh" << endl;
    cout << "    M    : toggle cloud exctraction mode" << endl;
    cout << "    N    : toggle normals exctraction" << endl;
    cout << "    I    : toggle independent camera mode" << endl;
    cout << "    B    : toggle volume bounds" << endl;
    cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
    cout << "    C    : clear clouds" << endl;    
    cout << "    D    : crop to face" << endl;
    cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
    cout << "    7,8  : save mesh to PLY, VTK" << endl;
    cout << "   X, V  : TSDF volume utility" << endl;
    cout << endl;
  }  

  bool exit_;
  bool scan_;
  bool scan_mesh_;
  bool scan_volume_;
  bool volume_scanned_;

  bool independent_camera_;

  bool registration_;
  bool integrate_colors_;  
  bool pcd_source_ = false;
  float focal_length_;

  float camera_fx = 882.f;
  float camera_fy = 885.f;
  float principal_cx = 339.f;
  float principal_cy = 264.f;
  float size_multiplier = 2000.f;
  float crop_from_nose_mm_y = 80.f;
  float crop_from_nose_mm_z = 80.f;
  float radius_from_middle = 25.f;
  float nose_y_displacement = -15.f;
  float accept_angle_deg = 90.f;
  
  pcl::Grabber& capture_;
  KinfuTracker kinfu_;

  SceneCloudView scene_cloud_view_;
  ImageView image_view_;
  boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;

  KinfuTracker::DepthMap depth_device_;

  pcl::TSDFVolume<float, short> tsdf_volume_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

  vector<uint32_t> host_voxel_colors;

  Evaluation::Ptr evaluation_ptr_;
  
  boost::mutex data_ready_mutex_;
  boost::condition_variable data_ready_cond_;
 
  std::vector<KinfuTracker::PixelRGB> source_image_data_;
  std::vector<unsigned short> source_depth_data_;
  PtrStepSz<const unsigned short> depth_;
  PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;

  int time_ms_;
  int icp_, viz_;

  boost::shared_ptr<CameraPoseProcessor> pose_processor_;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  static void
  keyboard_callback (const visualization::KeyboardEvent &e, void *cookie)
  {
    KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

    int key = e.getKeyCode ();

    if (e.keyUp ())
    {
      string sym = e.getKeySym();
      if (sym == "F1") key = (int) '7';
      if (sym == "F2") key = (int) 'a';
      if (sym == "F3") key = (int) '1';
      if (sym == "F4") key = (int) 't';
      if (sym == "F5") key = (int) 'v';
      if (sym == "F6") key = (int) 't';
      if (sym == "F7") key = (int) 'x';
    }

    if (e.keyUp ())    
      switch (key)
      {
      case 27: app->exit_ = true; break;
      case (int)'t': case (int)'T': app->scan_ = true; break;
      case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
      case (int)'h': case (int)'H': app->printHelp (); break;
      case (int)'m': case (int)'M': app->scene_cloud_view_.toggleExtractionMode (); break;
      case (int)'n': case (int)'N': app->scene_cloud_view_.toggleNormals (); break;      
      case (int)'c': case (int)'C': app->scene_cloud_view_.clearClouds (true); break;
      case (int)'d': case (int)'D': app->prepareMesh(); break;
      case (int)'r': case (int)'R': app->kinfu_.reset(); break;
      case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
      case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kinfu_.volume().getSize()); break;
      case (int)'7': case (int)'8': app->writeMesh (key - (int)'0'); break;
      case (int)'1': case (int)'2': case (int)'3': app->writeCloud (key - (int)'0'); break;      
      case '*': app->image_view_.toggleImagePaint (); break;
	  case (int)' ': app->capture_.stop(); break;

      case (int)'x': case (int)'X':
        app->scan_volume_ = !app->scan_volume_;
        cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
        break;
      case (int)'v': case (int)'V':
        if (!app->volume_scanned_)
        {
          cout << "Volume is not yet scanned, use 'x' and 't' first!" << endl;
        } else {
          cout << "Saving TSDF volume to tsdf_volume.dat ... " << flush;
          app->tsdf_volume_.save ("tsdf_volume.dat", true);
          cout << "done [" << app->tsdf_volume_.size () << " voxels]" << endl;
          cout << "Saving TSDF volume cloud to tsdf_cloud.pcd ... " << flush;
          pcl::io::savePCDFile<pcl::PointXYZI> ("tsdf_cloud.pcd", *app->tsdf_cloud_ptr_, true);
          cout << "done [" << app->tsdf_cloud_ptr_->size () << " points]" << endl;

          if (app->integrate_colors_)
          {
            string filename = "color_volume.dat";
            cout << "Saving color volume to " << filename << " ... " << flush;

            if (app->host_voxel_colors.empty())
            {
              cout << "WARNING: Color volume is empty" << endl;
              break;
            }

            std::ofstream file (filename.c_str(), std::ios_base::binary);

            if (file.is_open())
            {
              file.write ((char*) &(app->host_voxel_colors.at(0)), app->host_voxel_colors.size() * sizeof(uint32_t));
              file.close();
            } else {
              pcl::console::print_error ("Error: Couldn't open file %s.\n", filename.c_str());
              break;
            }

            cout << "done [" << app->host_voxel_colors.size () << " voxels]" << endl << endl;
          }

        }
        break;

      default:
        cout << "Unknown key " << e.getKeySym() << " (code " << key << ")" << endl;
      }    
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename CloudPtr> void
writeCloudFile (int format, const CloudPtr& cloud_prt)
{
  if (format == KinFuApp::PCD_BIN)
  {
    cout << "Saving point cloud to 'cloud_bin.pcd' (binary)... " << flush;
    pcl::io::savePCDFile ("cloud_bin.pcd", *cloud_prt, true);
  }
  else
  if (format == KinFuApp::PCD_ASCII)
  {
    cout << "Saving point cloud to 'cloud.pcd' (ASCII)... " << flush;
    pcl::io::savePCDFile ("cloud.pcd", *cloud_prt, false);
  }
  else   /* if (format == KinFuApp::PLY) */
  {
    cout << "Saving point cloud to 'cloud.ply' (ASCII)... " << flush;
    pcl::io::savePLYFileASCII ("cloud.ply", *cloud_prt);
  
  }
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh)
{
  if (format == KinFuApp::MESH_PLY)
  {
    cout << "Saving mesh to to 'mesh.ply'... " << flush;
    pcl::io::savePLYFile("mesh.ply", mesh);		
  }
  else /* if (format == KinFuApp::MESH_VTK) */
  {
    cout << "Saving mesh to to 'mesh.vtk'... " << flush;
    pcl::io::saveVTKFile("mesh.vtk", mesh);    
  }  
  cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
print_cli_help ()
{
  cout << "\nKinFu parameters:" << endl;
  cout << "    --help, -h                              : print this message" << endl;  
  cout << "    --registration, -r                      : try to enable registration (source needs to support this)" << endl;
  cout << "    --current-cloud, -cc                    : show current frame cloud" << endl;
  cout << "    --save-views, -sv                       : accumulate scene view and save in the end ( Requires OpenCV. Will cause 'bad_alloc' after some time )" << endl;  
  cout << "    --integrate-colors, -ic                 : enable color integration mode (allows to get cloud with colors)" << endl;   
  cout << "    --start-at-side                         : start at the center of one side of the scanning volume" << endl;
  cout << "    --scale-truncation, -st                 : scale the truncation distance and raycaster based on the volume size" << endl;
  cout << "    -volume_size <size_in_meters>           : define integration volume size" << endl;
  cout << "    --depth-intrinsics <fx>,<fy>[,<cx>,<cy> : set the intrinsics of the depth camera" << endl;
  cout << "    -save_pose <pose_file.csv>              : write tracked camera positions to the specified file" << endl;
  cout << "Valid depth data sources:" << endl; 
  cout << "    -dev <device> (default), -oni <oni_file>, -pcd <pcd_file or directory>" << endl;
  cout << "";
  cout << "For loading voxel grids:" << endl; 
  cout << "    -tsdf <tsdf+weights_file>               : load a saved TSDF volume; disables tracking and fusion" << endl;
  cout << "    -color <color_file>                     : load a saved color volume; requires -tsdf" << endl;
  cout << "";
  cout << " For RGBD benchmark (Requires OpenCV):" << endl; 
  cout << "    -eval <eval_folder> [-match_file <associations_file_in_the_folder>]" << endl;
    
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
main (int argc, char* argv[])
{  
  if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
    return print_cli_help ();
  
  int device = 0;
  pc::parse_argument (argc, argv, "-gpu", device);
  pcl::gpu::setDevice (device);
  pcl::gpu::printShortCudaDeviceInfo (device);

//  if (checkIfPreFermiGPU(device))
//    return cout << endl << "Kinfu is supported only for Fermi and Kepler arhitectures. It is not even compiled for pre-Fermi by default. Exiting..." << endl, 1;
  
  boost::shared_ptr<pcl::Grabber> capture;
  
  bool triggered_capture = false;
  bool pcd_input = false;
  
  std::string eval_folder, match_file, openni_device, oni_file, pcd_dir;
  string tsdf_file = "";
  string tcp_addr = "";
  try
  {    
    if (pc::parse_argument (argc, argv, "-dev", openni_device) > 0)
    {
      capture.reset (new pcl::OpenNIGrabber (openni_device));
    }
    else if (pc::parse_argument (argc, argv, "-oni", oni_file) > 0)
    {
      triggered_capture = true;
      bool repeat = false; // Only run ONI file once
      capture.reset (new pcl::ONIGrabber (oni_file, repeat, ! triggered_capture));
    }
    else if (pc::parse_argument (argc, argv, "-pcd", pcd_dir) > 0)
    {
      float fps_pcd = 15.0f;
      pc::parse_argument (argc, argv, "-pcd_fps", fps_pcd);

      vector<string> pcd_files = getPcdFilesInDir(pcd_dir);    

      // Sort the read files by name
      sort (pcd_files.begin (), pcd_files.end ());
      capture.reset (new pcl::PCDGrabber<pcl::PointXYZRGBA> (pcd_files, fps_pcd, false));
      triggered_capture = true;
      pcd_input = true;
    }
    else if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
    {
      //init data source latter
      pc::parse_argument (argc, argv, "-match_file", match_file);
    }
    else if (pc::parse_argument (argc, argv, "-tsdf", tsdf_file) > 0)
    {
      // No grabber must be created.
    }
    else if (pc::parse_argument (argc, argv, "-tcp", tcp_addr) > 0)
    {
      capture.reset (new pcl::TCPGrabber ());
    }
    else
    {
      capture.reset( new pcl::OpenNIGrabber() );
        
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224932.oni", true, ! triggered_capture) );
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/reg20111229-180846.oni, true, ! triggered_capture) );    
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("/media/Main/onis/20111013-224932.oni", true, ! triggered_capture) );
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224551.oni", true, ! triggered_capture) );
      //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224719.oni", true, ! triggered_capture) );    
    }
  }
  catch (const pcl::PCLException& e) { return cout << "Can't open depth source" << endl << e.what() << endl, -1; }

  float volume_size = 1.f;
  pc::parse_argument (argc, argv, "-volume_size", volume_size);
  printf("nh2: volume size %f\n", volume_size);

  int icp = 1, visualization = 1;
  std::vector<float> depth_intrinsics;
  pc::parse_argument (argc, argv, "--icp", icp);
  pc::parse_argument (argc, argv, "--viz", visualization);
  bool start_at_side = pc::find_switch (argc, argv, "--start-at-side");

  std::string camera_pose_file;
  boost::shared_ptr<CameraPoseProcessor> pose_processor;
  if (pc::parse_argument (argc, argv, "-save_pose", camera_pose_file) && camera_pose_file.size () > 0)
  {
    pose_processor.reset (new CameraPoseWriter (camera_pose_file));
  }

  KinFuApp app (*capture, volume_size, icp, visualization, pose_processor, start_at_side);
  app.parseConfig("config.txt");
  std::vector<float> intrinsics;
  intrinsics.push_back(app.camera_fx);
  intrinsics.push_back(app.camera_fy);
  intrinsics.push_back(app.principal_cx);
  intrinsics.push_back(app.principal_cy);
  app.setDepthIntrinsics(intrinsics);

  app.kinfu_.acceptAngle() = app.accept_angle_deg;

  if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
    app.toggleEvaluationMode(eval_folder, match_file);

  bool disable_grabbing = false;
  if (!tsdf_file.empty())
  {
    app.loadTsdf(tsdf_file);
    disable_grabbing = true;

    string color_file;
    if (pc::parse_argument (argc, argv, "-color", color_file) > 0)
    {
      // loadColor needs the color volume initialized
      app.toggleColorIntegrationWithoutRegistration();

      app.loadColor(color_file);
    }
  }
  else
  {

    if (pc::find_switch (argc, argv, "--current-cloud") || pc::find_switch (argc, argv, "-cc"))
      app.initCurrentFrameView ();

    if (pc::find_switch (argc, argv, "--save-views") || pc::find_switch (argc, argv, "-sv"))
      app.image_view_.accumulate_views_ = true;  //will cause bad alloc after some time  

    if (pc::find_switch (argc, argv, "--registration") || pc::find_switch (argc, argv, "-r"))  
      app.initRegistration();
        
    if (pc::find_switch (argc, argv, "--integrate-colors") || pc::find_switch (argc, argv, "-ic")) {
      // if (!app.registration_) {
      //   pc::print_error("--integrate-colors requires --registration\n");
      //   return -1;
      // }
      app.toggleColorIntegration();
    }

    if (pc::find_switch (argc, argv, "--normals") || pc::find_switch (argc, argv, "-n"))
      app.scene_cloud_view_.toggleNormals();

    if (pc::find_switch (argc, argv, "--scale-truncation") || pc::find_switch (argc, argv, "-st"))
      app.enableTruncationScaling();
    
    if (pc::parse_x_arguments (argc, argv, "--depth-intrinsics", depth_intrinsics) > 0)
    {
      if ((depth_intrinsics.size() == 4) || (depth_intrinsics.size() == 2))
      {
         app.setDepthIntrinsics(depth_intrinsics);
      }
      else
      {
          pc::print_error("Depth intrinsics must be given on the form fx,fy[,cx,cy].\n");
          return -1;
      }   
    }
    
  }

  // executing
  try {
    if (disable_grabbing)
      app.startDisplayOnlyMainLoop ();
    else
      app.startMainLoop (triggered_capture);
  }
  catch (const pcl::PCLException& e) { cout << "PCLException: " << e.what() << endl; }
  catch (const std::bad_alloc& e) { cout << "Bad alloc: " << e.what() << endl; }
  catch (const std::exception& e) { cout << "Exception: " << e.what() << endl; }

#ifdef HAVE_OPENCV
  for (size_t t = 0; t < app.image_view_.views_.size (); ++t)
  {
    if (t == 0)
    {
      cout << "Saving depth map of first view." << endl;
      cv::imwrite ("./depthmap_1stview.png", app.image_view_.views_[0]);
      cout << "Saving sequence of (" << app.image_view_.views_.size () << ") views." << endl;
    }
    char buf[4096];
    sprintf (buf, "./%06d.png", (int)t);
    cv::imwrite (buf, app.image_view_.views_[t]);
    printf ("writing: %s\n", buf);
  }
#endif

  return 0;
}

// Formats an ASCII .ply file in the order that PCL expects it to be in,
// reordering some properties (switching colour and normal columns).

const int COL_NUM = 10;

// Order of properties that we desire, switching colour and normal columns:
//   0 property float x       ->  0 property float x
//   1 property float y       ->  1 property float y
//   2 property float z       ->  2 property float z
//   3 property float nx      ->  6 property uchar red
//   4 property float ny      ->  7 property uchar green
//   5 property float nz      ->  8 property uchar blue
//   6 property uchar red     ->  9 property uchar alpha
//   7 property uchar green   ->  3 property float nx
//   8 property uchar blue    ->  4 property float ny
//   9 property uchar alpha   ->  5 property float nz
int col_order[COL_NUM] = { 0, 1, 2, 6, 7, 8, 9, 3, 4, 5 };

