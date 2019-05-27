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
   * Initializes particles centered around (x, y) and direction theta, with standard deviations 
   * std[0], std[1], std[2], respectively.
   */
  num_particles = 120;
  
  std::default_random_engine seed;

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  particles.clear();
  for ( int i = 0; i < num_particles; i++) {
    Particle p = Particle();
    p.x = dist_x(seed);
    p.y = dist_y(seed);
    p.theta = dist_theta(seed);
    p.weight = 1.0;
    p.id = i;
    
    particles.push_back(p);
  }
  is_initialized = true;
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
    // Update position of particle
    if (std::abs(yaw_rate) < 0.0001) { // if yaw rate is (basically) 0
      particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      // Theta stays the same
    } else {
      particles[i].x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta = particles[i].theta + yaw_rate * delta_t;
    }
    
    // Add gaussian noise to particle position due to movement
    particles[i].x = particles[i].x + noise_x(seed);
    particles[i].y = particles[i].y + noise_y(seed);
    particles[i].theta = particles[i].theta + noise_theta(seed);
  }
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
   * Assigns id of closest landmark to each observation's .id attribute.
   * @param predicted Existing landmark positions (should be filtered to exclude those out of range)
   * @param observations Sensor readings in map coordinates
   * Every observation is given an id, unless there are no landmarks in range (i.e. predicted.size() === 0)
   */
  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs o = observations[i]; // where the sensor thinks a landmark is

    double min_dist = std::numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs p = predicted[j]; // where a landmark is according to map
      double d = dist(p.x, p.y, o.x, o.y);
      if (d < min_dist) {
        min_dist = d;
        observations[i].id = p.id;
      }
    }
  }
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
    Particle p = particles[i]; // Don't assign to this

    // Convert observations to map coordinates
    vector<LandmarkObs> observations_map;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs o = observations[j];
      LandmarkObs o_map = LandmarkObs();
      o_map.x = p.x + (cos(p.theta) * o.x) - (sin(p.theta) * o.y);
      o_map.y = p.y + (sin(p.theta) * o.x) + (cos(p.theta) * o.y);
      observations_map.push_back(o_map);
    }

    vector<LandmarkObs> acceptable_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      if (dist(p.x, p.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) > sensor_range) {
        continue;
      }
      LandmarkObs l;
      l.x = map_landmarks.landmark_list[j].x_f;
      l.y = map_landmarks.landmark_list[j].y_f;
      l.id = map_landmarks.landmark_list[j].id_i;
      acceptable_landmarks.push_back(l);
    }
    
    dataAssociation(acceptable_landmarks, observations_map);

    // Multiply the probabilities of each observation-landmark association
    particles[i].weight = 1.0;
    for (int j = 0; j < observations_map.size(); j++) {
      // Get observation
      LandmarkObs o_map = observations_map[j];
      // Find landmark associated with observation
      LandmarkObs associated_landmark;
      for (int k = 0; k < acceptable_landmarks.size(); k++) {
        if (acceptable_landmarks[k].id == o_map.id) {
          associated_landmark = acceptable_landmarks[k];
          break;
        }
      }

      // Update weight
      particles[i].weight = particles[i].weight * multivariate_gaussian(
        o_map.x,
        o_map.y,
        associated_landmark.x,
        associated_landmark.y,
        std_landmark[0],
        std_landmark[1]
      );
    }
    // end updating particle's weight
  } // end for each particle
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> resampled;

  std::default_random_engine seed;
  vector<double> p_distribution;
  for (int i = 0; i < particles.size(); i++) {
    p_distribution.push_back(particles[i].weight);
  }
  std::discrete_distribution<int> d(p_distribution.begin(), p_distribution.end());

  for (int i = 0; i < particles.size(); i++) {
    Particle p = particles[d(seed)];
    resampled.push_back(p);
  }
  particles = resampled;
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