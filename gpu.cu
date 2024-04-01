#include "common.h"
#include <cuda.h>
#include <cmath>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int bin_blks;

int num_bins_one_side;          // number of bin on one side of the square simulation space
int num_bins;                   // total number of bins in the grid
double bin_size;                // number of units across on one dimension of a bin in the simulation space

int* particle_ids;              // list of particle ids, sorted by bin
int* bin_ids;                   // holds the starting indices of particle_ids that correspond to new bins
int* bin_counts;                // number of particles in each bin
int* write_part_indices;        // used to update particle_ids to ensure operations in order


// Set an array on gpu to a value, useful for zeroing out arrays to recalculate stuff
__global__ void set_gpu_array(int* arr, int val, int arr_len){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= arr_len)
        return;
    arr[tid] = val;
}

// Calculate how many particles in each bin (bin counts array from recitation)
__global__ void update_bin_counts(particle_t* parts, int num_parts, int* bin_counts, int num_bins_one_side, double bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts){
        return;
    }
    
    // Find which bin a particle belongs to and add 1 to its count
    // use atomicAdd for synchronization
    particle_t* target_part = &parts[tid];
    int bin_idx = floor(target_part->x / bin_size) + floor(target_part->y / bin_size) * num_bins_one_side;;
    atomicAdd(&bin_counts[bin_idx], 1);
}

// Properly write correctly ordered particles back to particle_ids based on write_part_indices
__global__ void update_particle_ids(particle_t* parts, int num_parts, int* particle_ids, int* write_part_indices, int num_bins_one_side, double bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // Find which bin a particle belongs to and use atomicAdd to
    // locate how far down along the array it should go
    particle_t* target_part = &parts[tid];
    int bin_idx = floor(target_part->x / bin_size) + floor(target_part->y / bin_size) * num_bins_one_side;;
    int write_idx = atomicAdd(&write_part_indices[bin_idx], 1);
    particle_ids[write_idx] = tid;
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Naive starter code, computes forces on every particle with every other particle
__global__ void compute_forces_gpu_naive(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

// Serialized version of computng forces, using only the particle's own bin and neighboring bin
__global__ void compute_forces_gpu_serial(particle_t* particles, int num_parts, int* particle_ids, int* bin_ids, int num_bins_one_side, int num_bins, double bin_size) {
    // Get the thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;

    // Find own bin and neighbor bins for a particle
    int bin_x = (&particles[tid])->x / bin_size;
    int bin_y = (&particles[tid])->y / bin_size;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int neighbor_idx = (bin_x + dx) + (bin_y + dy) * num_bins_one_side;

            // Check if calculated neighbor bin is valid
            if (neighbor_idx >= 0 && neighbor_idx < num_bins) {

                // Get the indices of the neighbor bin and the particles in those indices
                // to apply forces with
                int target_bin_start_idx = bin_ids[neighbor_idx];
                int target_bin_end_idx = bin_ids[neighbor_idx + 1];
                for (int particle_idx = target_bin_start_idx; particle_idx < target_bin_end_idx; ++particle_idx) {
                    apply_force_gpu(particles[tid], particles[particle_ids[particle_idx]]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Calculate number of bins and dimensions
    num_bins_one_side = floor(size / cutoff);
    bin_size = size / num_bins_one_side;
    num_bins = num_bins_one_side * num_bins_one_side;

    // Allocate memory necessary for the arrays on gpu
    cudaMalloc((void**)& particle_ids, num_parts * sizeof(int));
    cudaMalloc((void**)& bin_ids, (num_bins + 1) * sizeof(int));
    cudaMalloc((void**)& bin_counts, num_bins * sizeof(int));
    cudaMalloc((void**)& write_part_indices, (num_bins + 1) * sizeof(int));

    // Calculate number of blks necessary not just for particles
    // but treating whole bins as a unit while performing bin operations
    bin_blks = (num_bins + NUM_THREADS - 1) / NUM_THREADS;
    int bin_blks_for_idxs = (num_bins + NUM_THREADS) / NUM_THREADS;

    // Instantiate enough bin_ids array(s) for each thread to have a "local" copy
    set_gpu_array<<<bin_blks_for_idxs, NUM_THREADS>>>(bin_ids, num_parts, num_bins + 1);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // Zero out bin counts before performing rebinning operations
    set_gpu_array<<<bin_blks, NUM_THREADS>>>(bin_counts, 0, num_bins);

    // Update counts of how many particles in each bin
    update_bin_counts<<<blks, NUM_THREADS>>>(parts, num_parts, bin_counts, num_bins_one_side, bin_size);
    
    // Use thrust operations to perform prefix sum to find starter indices of each bin, as shown in recitation
    // store to bin_ids, where bin_ids has length num_bins + 1, the indices between bin_ids[i] and bin_ids[i+1]
    // is the ith bin in the grid. Essentially re-binning operation occurs here
    thrust::exclusive_scan(thrust::device, bin_counts, bin_counts + num_bins, bin_ids);
    
    // Copy pre-operation bin_ids so no concurrent writing and moving to finish re-binning
    cudaMemcpy(write_part_indices, bin_ids, num_bins * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // update particle_ids using stored indices from write_part_indices
    // write_part_indices is updated after this, bin_ids maintains
    update_particle_ids<<<blks, NUM_THREADS>>>(parts, num_parts, particle_ids, write_part_indices, num_bins_one_side, bin_size);
    
    // Compute forces
    compute_forces_gpu_serial<<<blks, NUM_THREADS>>>(parts, num_parts, particle_ids, bin_ids, num_bins_one_side, num_bins, bin_size);
    
    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}