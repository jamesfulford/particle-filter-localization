/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  
  std::default_random_engine seed;

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for ( int i = 0; i < num_particles; i++) {
    Particle p = Particle();
    p.x = dist_x(seed);
    p.y = dist_y(seed);
    p.theta = dist_theta(seed);
    p.weight = 1.0;
    
    particles.push_back(p);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine seed;

  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);

  for (int i = 0; i < particles.size(); i++) {
    Particle p = particles[i];
    
    // Update position of particle
    if (std::abs(yaw_rate) < 0.0001) { // if yaw rate is (basically) 0
      // TODO(jamesfulford): Did I swap X and Y here?
      p.x += velocity * cos(p.theta);
      p.y += velocity * sin(p.theta);
      // Theta stays the same
    } else {
      p.x += (velocity / yaw_rate) * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta));
      p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + (yaw_rate * delta_t)));
      p.theta += yaw_rate * delta_t;
    }
    
    // Add gaussian noise to particle position due to movement
    p.x += noise_x(seed);
    p.y += noise_y(seed);
    p.theta += noise_theta(seed);
  }
}

double pythagorean (double x1, double y1, double x2, double y2) {
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

double distance (LandmarkObs o, Particle p) {
  return pythagorean(o.x, o.y, p.x, p.y);
}

double multivariate_gaussian (double x, double y, double base_x, double base_y, double stdx, double stdy) {
  return (
    pow(M_E, -(
      (pow(y - base_y, 2) / (2 * pow(stdy, 2)))
       + (pow(x - base_x, 2) / (2 * pow(stdx, 2)))
    )) / (2 * M_PI * stdx * stdy)
  );
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
//   for (int i = 0; i < observations.size(); i++) {
//     LandmarkObs o = observations[i]; // where the sensor thinks a landmark is

//     double min_dist = std::numeric_limits<double>::max();
//     LandmarkObs min_pred;
//     for (int j = 0; j < predicted.size(); j++) {
//       LandmarkObs p = predicted[j]; // where a landmark is predict
//       double d = distance(p, o);
//       if (d < min_dist) {
//         min_pred = p;
//         min_dist = d;
//       }
//     }
//     if (min_pred !== null) {
//       o.id = min_pred.id;
//     }
//   }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < particles.size(); i++) {
    Particle p = particles[i];
    p.weight = 1.0;
    
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      if (pythagorean(p.x, p.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) > sensor_range) {
        continue; // skip landmarks outside of sensor range
      }

      // Find observation closest to j'th landmark
      LandmarkObs closest_obs_map;
      double min_dist = std::numeric_limits<double>::max();
      for (int k = 0; k < observations.size(); k++) {
        LandmarkObs o = observations[k];

        LandmarkObs o_map = LandmarkObs();
        o_map.x = p.x + (cos(p.theta) * o.x) - (sin(p.theta) * o.y);
        o_map.y = p.y + (sin(p.theta) * o.x) + (cos(p.theta) * o.y);
        
        double d = pythagorean(o_map.x, o_map.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
        if (d < min_dist) {
          closest_obs_map = o_map;
          min_dist = d;
        }
      }
      // Update particle's weight with probability of this landmark being observed by its closed observation
      p.weight *= multivariate_gaussian(closest_obs_map.x, closest_obs_map.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, std_landmark[0], std_landmark[1]);
    };
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}