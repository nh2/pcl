/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 */

#include <pcl/pcl_config.h>

#include <pcl/io/tcp_grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/common/io.h>
#include <pcl/console/print.h>
#include <pcl/io/boost.h>
#include <pcl/exceptions.h>

#include <boost/asio.hpp>
#include <boost/thread.hpp>

#include <iostream>
#include <queue>

using boost::asio::ip::tcp;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::TCPGrabber::TCPGrabber ()
  : running_ (false)
  , fps_ (0.0f)
{
  image_signal_ = createSignal<sig_cb_tcp_image> ();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::TCPGrabber::~TCPGrabber () throw ()
{
  stop ();

  disconnect_all_slots<sig_cb_tcp_image> ();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::TCPGrabber::init ()
{

  return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::TCPGrabber::close ()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::TCPGrabber::start ()
{
  init ();
  running_ = true;

  grabber_thread_ = boost::thread (&pcl::TCPGrabber::processGrabbing, this);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::TCPGrabber::stop ()
{
  running_ = false;
  grabber_thread_.join ();

  close ();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::TCPGrabber::isRunning () const
{
  return (running_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string
pcl::TCPGrabber::getName () const
{
  return (std::string ("TCPGrabber"));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float
pcl::TCPGrabber::getFramesPerSecond () const
{
  fps_mutex_.lock ();
  float fps = fps_;
  fps_mutex_.unlock ();

  return (fps);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::TCPGrabber::processGrabbing ()
{

  pcl::StopWatch stop_watch;
  std::queue<double> capture_time_queue;
  double total_time = 0.0f;

  stop_watch.reset ();

  using namespace boost;

  asio::io_service io_service;

  tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 1234));

  // socket accept loop
  accept_loop:
  for (;;)
  {
    tcp::socket socket(io_service);
    std::cout << "TCPGrabber: Waiting for client" << std::endl;
    acceptor.accept(socket);
    std::cout << "TCPGrabber: Accepted client" << std::endl;

    boost::array<unsigned char, 640*480*3> rgb_buf;
    boost::array<unsigned short, 640*480> depth_buf;

    bool continue_grabbing = true;
    while (continue_grabbing)
    {
      // Acquire frame

      boost::system::error_code error_code;
      asio::read(socket, asio::buffer(rgb_buf), boost::asio::transfer_at_least(rgb_buf.size()), error_code);
      asio::read(socket, asio::buffer(depth_buf), boost::asio::transfer_at_least(depth_buf.size()*2), error_code);

      std::cout << "TCPGrabber: got frame" << std::endl;

      if (error_code)
      {
        goto accept_loop;
        // PCL_THROW_EXCEPTION (pcl::IOException, "TCPGrabber: Could not read from socket");
      }

      // publish frame
      if (num_slots<sig_cb_tcp_image> () > 0)
      {
        image_signal_->operator() (rgb_buf, depth_buf);
      }

      const double capture_time = stop_watch.getTimeSeconds ();
      total_time += capture_time;

      capture_time_queue.push (capture_time);

      if (capture_time_queue.size () >= 30)
      {
        double removed_time = capture_time_queue.front ();
        capture_time_queue.pop ();

        total_time -= removed_time;
      }

      fps_mutex_.lock ();
      fps_ = static_cast<float> (total_time / capture_time_queue.size ());
      fps_mutex_.unlock ();
    }

  }
}
