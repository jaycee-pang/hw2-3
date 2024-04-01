#include "common.h"
#include <cuda.h>
#include <stdio.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
static int step; 

static int total_bins;
static int bins_per_axis;
static double bin_size;


int* particle_bin_assignments;
int* particle_locations_bin_counts;
int* particle_locations_bin_displacements;
int* particle_locations_sorted_by_bin;

#define TIME_START(start) cudaEventRecord(start, 0);
#define TIME_END(stop)   \     
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop);

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor, int tid, double bin_size) {
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

// __global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
//     // Get thread (particle) ID
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= num_parts)
//         return;

//     particles[tid].ax = particles[tid].ay = 0;
//     for (int j = 0; j < num_parts; j++)
//         apply_force_gpu(particles[tid], particles[j]);
// }

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* particle_locations_sorted_by_bin, int* particle_locations_bin_displacements, int* particle_locations_bin_counts, int bins_per_axis, double bin_size, int step)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t my_particle = particles[tid];
    my_particle.ax = my_particle.ay = 0;

    // Determine the bin of the current particle
    int bin_x = my_particle.x / bin_size;
    int bin_y = my_particle.y / bin_size;

    // Calculate the range of neighboring bins to check
    int start_x = max(bin_x - 1, 0);
    int end_x = min(bin_x + 1, bins_per_axis - 1);
    int start_y = max(bin_y - 1, 0);
    int end_y = min(bin_y + 1, bins_per_axis - 1);

    // Loop through adjacent bins
    for (int by = start_y; by <= end_y; ++by) {
        for (int bx = start_x; bx <= end_x; ++bx) {
            int neighbor_bin_index = by * bins_per_axis + bx;
            int start_idx = particle_locations_bin_displacements[neighbor_bin_index];
            int count = particle_locations_bin_counts[neighbor_bin_index];
            for (int i = 0; i < count; i++) {
                int neighbor_idx = particle_locations_sorted_by_bin[start_idx + i];
                if (tid != neighbor_idx) {
                    apply_force_gpu(my_particle, particles[neighbor_idx], tid, bin_size);
                }
            }
        }
    }

    particles[tid].ax = my_particle.ax;
    particles[tid].ay = my_particle.ay;
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    // int tid = threadIdx.x + blockIdx.x * blockDim.x;
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

    double minimum_bin_size = cutoff + 0.000001;
    bins_per_axis = floor(size / minimum_bin_size);
    total_bins = bins_per_axis * bins_per_axis;
    bin_size = size / bins_per_axis;

    cudaMalloc(&particle_locations_sorted_by_bin, num_parts * sizeof(int));
    cudaMemset(particle_locations_sorted_by_bin, 0, num_parts * sizeof(int));
    cudaMalloc(&particle_locations_bin_counts, total_bins * sizeof(int));
    cudaMemset(particle_locations_bin_counts, 0, total_bins * sizeof(int));
    cudaMalloc(&particle_bin_assignments, num_parts * sizeof(int));
    cudaMemset(particle_bin_assignments, 0, num_parts * sizeof(int));
    cudaMalloc(&particle_locations_bin_displacements, total_bins * sizeof(int));
    cudaMemset(particle_locations_bin_displacements, 0, total_bins * sizeof(int));
}


__global__ void assign_particles_to_bins(int* particle_bin_assignments, particle_t* parts, int num_parts, double bin_size, int bins_per_axis, double size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_parts) {
        int bin_x = parts[tid].x / bin_size;
        int bin_y = parts[tid].y / bin_size;
        particle_bin_assignments[tid] = bin_y * bins_per_axis + bin_x;
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // // Compute forces
    // compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);

    // // Move particles
    // move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    TIME_START(start)

    // Assign particles to bins
    assign_particles_to_bins<<<blks, NUM_THREADS>>>(particle_bin_assignments, parts, num_parts, bin_size, bins_per_axis, size);
    cudaDeviceSynchronize();

    // Count particles in each bin
    int* host_particle_bin_assignments = (int*)malloc(num_parts * sizeof(int));
    cudaMemcpy(host_particle_bin_assignments, particle_bin_assignments, num_parts * sizeof(int), cudaMemcpyDeviceToHost);
    int* host_particle_locations_bin_counts = (int*)calloc(total_bins, sizeof(int));
    for (int i = 0; i < num_parts; i++) {
        int binIndex = host_particle_bin_assignments[i];
        host_particle_locations_bin_counts[binIndex]++;
    }

    // Calculate displacements for each bin
    int* host_particle_locations_bin_displacements = (int*)malloc(total_bins * sizeof(int));
    host_particle_locations_bin_displacements[0] = 0; // First bin starts at index 0
    for (int i = 1; i < total_bins; i++) {
        host_particle_locations_bin_displacements[i] = host_particle_locations_bin_displacements[i - 1] + host_particle_locations_bin_counts[i - 1];
    }

    // Add particles to sorted array
    int* host_particles_added_per_bin = (int*)calloc(total_bins, sizeof(int));
    int* host_particle_locations_sorted_by_bin = (int*)malloc(num_parts * sizeof(int));
    for (int i = 0; i < num_parts; ++i) {
        int binIndex = host_particle_bin_assignments[i];
        int displacement = host_particle_locations_bin_displacements[binIndex];
        int position = displacement + host_particles_added_per_bin[binIndex];
        host_particle_locations_sorted_by_bin[position] = i;
        host_particles_added_per_bin[binIndex]++;
    }
    cudaMemcpy(particle_locations_bin_counts, host_particle_locations_bin_counts, total_bins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(particle_locations_bin_displacements, host_particle_locations_bin_displacements, total_bins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(particle_locations_sorted_by_bin, host_particle_locations_sorted_by_bin, num_parts * sizeof(int), cudaMemcpyHostToDevice);
    free(host_particle_bin_assignments);
    free(host_particle_locations_bin_counts);
    free(host_particle_locations_bin_displacements);
    free(host_particles_added_per_bin);
    free(host_particle_locations_sorted_by_bin);

    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, particle_locations_sorted_by_bin, particle_locations_bin_displacements, particle_locations_bin_counts, bins_per_axis, bin_size, step);
    cudaDeviceSynchronize();

    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();

    // Print how long simulate_one_step took in ms
    TIME_END(stop)

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    step++;

}
