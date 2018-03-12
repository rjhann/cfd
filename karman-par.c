#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>

#include <mpi.h>

#include "alloc.h"
#include "boundary.h"
#include "datadef.h"
#include "init.h"
#include "simulation.h"

void decompose_2d(int imax, int jmax, int nprocs, int dims[2]) ;

void write_bin(float **u, float **v, float **p, char **flag,
     int imax, int jmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char *file);

static void print_usage(void);
static void print_version(void);
static void print_help(void);

static char *progname;

int proc = 0;                       /* Rank of the current process */
int nprocs = 0;                /* Number of processes in communicator */

// A communicator holding the cartesian topology.
MPI_Comm cartesian_comm;

// Global start coordinates of local sub-array.
int istart;
int jstart;

// Communication arays for MPI_Alltoallw edge transfers.
int* sendcounts_edge;
int* senddispls_edge;
MPI_Datatype* sendtypes_edge;
int* recvcounts_edge;
int* recvdispls_edge;
MPI_Datatype* recvtypes_edge;

extern double poisson_time;

#define PACKAGE "karman"
#define VERSION "1.0"

/* Command line options */
static struct option long_opts[] = {
    { "del-t",   1, NULL, 'd' },
    { "help",    0, NULL, 'h' },
    { "imax",    1, NULL, 'x' },
    { "infile",  1, NULL, 'i' },
    { "jmax",    1, NULL, 'y' },
    { "outfile", 1, NULL, 'o' },
    { "t-end",   1, NULL, 't' },
    { "verbose", 1, NULL, 'v' },
    { "version", 1, NULL, 'V' },
    { 0,         0, 0,    0   } 
};
#define GETOPTS "d:hi:o:t:v:Vx:y:"

int main(int argc, char *argv[])
{
    int verbose = 1;          /* Verbosity level */
    float xlength = 22.0;     /* Width of simulated domain */
    float ylength = 4.1;      /* Height of simulated domain */
    int imax = 660;           /* Number of cells horizontally */
    int jmax = 120;           /* Number of cells vertically */

    char *infile;             /* Input raw initial conditions */
    char *outfile;            /* Output raw simulation results */

    float t_end = 2.1;        /* Simulation runtime */
    float del_t = 0.003;      /* Duration of each timestep */
    float tau = 0.5;          /* Safety factor for timestep control */

    int itermax = 100;        /* Maximum number of iterations in SOR */
    float eps = 0.001;        /* Stopping error threshold for SOR */
    float omega = 1.7;        /* Relaxation parameter for SOR */
    float gamma = 0.9;        /* Upwind differencing factor in PDE
                                 discretisation */

    float Re = 150.0;         /* Reynolds number */
    float ui = 1.0;           /* Initial X velocity */
    float vi = 0.0;           /* Initial Y velocity */

    float t, delx, dely;
    int  i, j, itersor = 0, ifluid = 0, ibound = 0;
    float res;
    float **u, **v, **p, **rhs, **f, **g;
    char  **flag;
    int init_case, iters = 0;
    int show_help = 0, show_usage = 0, show_version = 0;

    progname = argv[0];
    infile = strdup("");
    outfile = strdup("karman.bin");

    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    double init_start = MPI_Wtime();

    /* BEGIN INITIALISE PROBLEM */

    // Variable to capture early exit conditions of sequential program to
    // allow clean MPI exit. Negative value indicates don't exit early.
    int exit_val = -1;
    
    // Have only master process read command line args and initialise global
    // problem array.
    if (proc == 0) {
        int optc;
        while ((optc = getopt_long(argc, argv, GETOPTS, long_opts, NULL)) != -1) {
            switch (optc) {
                case 'h':
                    show_help = 1;
                    break;
                case 'V':
                    show_version = 1;
                    break;
                case 'v':
                    verbose = atoi(optarg);
                    break;
                case 'x':
                    imax = atoi(optarg);
                    break;
                case 'y':
                    jmax = atoi(optarg);
                    break;
                case 'i':
                    free(infile);
                    infile = strdup(optarg);
                    break;
                case 'o':
                    free(outfile);
                    outfile = strdup(optarg);
                    break;
                case 'd':
                    del_t = atof(optarg);
                    break;
                case 't':
                    t_end = atof(optarg);
                    break;
                default:
                    show_usage = 1;
            }
        }
        if (show_usage || optind < argc) {
            print_usage();
            exit_val = 1;
        }
        
        if (show_version) {
            print_version();
            if (!show_help) {
                exit_val = 0;
            }
        }
        
        if (show_help) {
            print_help();
            exit_val = 0;
        }
        
        delx = xlength/imax;
        dely = ylength/jmax;
    
        /* Allocate arrays */
        // Add an extra 1 to ease handling of uneven blocks.
        u    = alloc_floatmatrix(imax+2 + 1, jmax+2 + 1);
        v    = alloc_floatmatrix(imax+2 + 1, jmax+2 + 1);
        f    = alloc_floatmatrix(imax+2 + 1, jmax+2 + 1);
        g    = alloc_floatmatrix(imax+2 + 1, jmax+2 + 1);
        p    = alloc_floatmatrix(imax+2 + 1, jmax+2 + 1);
        rhs  = alloc_floatmatrix(imax+2 + 1, jmax+2 + 1);
        flag = alloc_charmatrix(imax+2 + 1, jmax+2 + 1);
    
        if (!u || !v || !f || !g || !p || !rhs || !flag) {
            fprintf(stderr, "Couldn't allocate memory for matrices.\n");
            exit_val = 1;
        }
    
        /* Read in initial values from a file if it exists */
        init_case = read_bin(u, v, p, flag, imax, jmax, xlength, ylength, infile);
            
        if (init_case > 0) {
            /* Error while reading file */
            exit_val = 1;
        }
    
        if (init_case < 0) {
            /* Set initial values if file doesn't exist */
            for (i=0;i<=imax+1;i++) {
                for (j=0;j<=jmax+1;j++) {
                    u[i][j] = ui;
                    v[i][j] = vi;
                    p[i][j] = 0.0;
                }
            }
            init_flag(flag, imax, jmax, delx, dely, &ibound);
            apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
        }
    } else {    // proc != 0
        /* Allocate some dummies to protect against segfaults */
        u    = alloc_floatmatrix(1, 1);
        v    = alloc_floatmatrix(1, 1);
        f    = alloc_floatmatrix(1, 1);
        g    = alloc_floatmatrix(1, 1);
        p    = alloc_floatmatrix(1, 1);
        rhs  = alloc_floatmatrix(1, 1);
        flag = alloc_charmatrix(1, 1);
    }
    
    // Broadcast whether there was an exit case and exit cleanly if there was.
    MPI_Bcast(&exit_val, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (exit_val >= 0) {
        MPI_Finalize();
        return exit_val;
    }
    
    MPI_Bcast(&verbose, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imax, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&jmax, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t_end, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delx, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dely, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    /* END INITIALISE PROBLEM */
    
    
    /* BEGIN DECOMPOSE PROBLEM */
    
    // Calculate and broadcast problem decomposition.
    int dims[2];
    
    if (proc == 0) {
        decompose_2d(imax, jmax, nprocs, dims);
        if (verbose > 1) {
            printf("Decomposing problem into %dx%d blocks.\n",
                dims[0], dims[1]);
        }
    }
    
    MPI_Bcast(dims, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    /* END DECOMPOSE PROBLEM */
    
    
    /* BEGIN CREATE TOPOLOGY */
    
    // Create cartesian topology.
    int periodic[] = {0, 0};
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &cartesian_comm);
    
    int coords[2];
    MPI_Cart_coords(cartesian_comm, proc, 2, coords);
    
    if (verbose > 1) {
        printf("Rank %d is on block (%d, %d).\n", proc, coords[0], coords[1]);
    }
    
    // Ranks of adjacent processors. -1 means processor is on a border.
    int proc_left = -1;
    int proc_right = -1;
    int proc_up = -1;
    int proc_down = -1;
    
    int neighbor_coords[2];
    // If not on left edge.
    if (coords[0] > 0) {
        neighbor_coords[0] = coords[0] - 1;
        neighbor_coords[1] = coords[1];
        MPI_Cart_rank(cartesian_comm, neighbor_coords, &proc_left);
    }
    // If not on right edge.
    if (coords[0] < dims[0]-1) {
        neighbor_coords[0] = coords[0] + 1;
        neighbor_coords[1] = coords[1];
        MPI_Cart_rank(cartesian_comm, neighbor_coords, &proc_right);
    }
    // If not on top edge.
    if (coords[1] > 0) {
        neighbor_coords[0] = coords[0];
        neighbor_coords[1] = coords[1] - 1;
        MPI_Cart_rank(cartesian_comm, neighbor_coords, &proc_up);
    }
    // If not on bottom edge.
    if (coords[1] < dims[1]-1) {
        neighbor_coords[0] = coords[0];
        neighbor_coords[1] = coords[1] + 1;
        MPI_Cart_rank(cartesian_comm, neighbor_coords, &proc_down);
    }
    
    if (verbose > 2) {
        printf("Rank %d neighbors - left: %d up: %d right: %d down: %d.\n",
            proc, proc_left, proc_up, proc_right, proc_down);
    }
    
    /* END CREATE TOPOLOGY */
    
    
    /* BEGIN CREATE LOCAL SUB-ARRAYS */
    
    // Local sub-arrays.
    float **l_p, **l_rhs;
    char  **l_flag;
    
    istart = coords[0] * imax / dims[0];
    jstart = coords[1] * jmax / dims[1];
    
    // Size of block processor will actually work on.
    int l_imax = ((coords[0] + 1) * imax / dims[0]) - istart;
    int l_jmax = ((coords[1] + 1) * jmax / dims[1]) - jstart;
    
    if (verbose > 2) {
        printf("Rank %d has block of size %dx%d (excluding borders).\n",
            proc, l_imax, l_jmax);
    }
    
    // If problem doesn't divide exactly some blocks will have one extra
    // element. Allocate blocks assuming this and ones with smaller blocks
    // will ignore edges.
    int block_x = imax / dims[0] + 1;
    int block_y = jmax / dims[1] + 1;
    
    int l_alloc_err = 0;
    int alloc_err = 0;
    
    // Allocate arrays with space for large block and neighbor's border.
    l_p    = alloc_floatmatrix(block_x+2, block_y+2);
    l_rhs  = alloc_floatmatrix(block_x+2, block_y+2);
    l_flag = alloc_charmatrix(block_x+2, block_y+2);
    
    if (!l_p || !l_rhs || !l_flag) {
        fprintf(stderr, "Rank %d couldn't allocate memory for matrices.\n",
            proc);
        alloc_err = 1;
    }
    
    MPI_Allreduce(&l_alloc_err, &alloc_err, 1,
        MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (alloc_err) {
        MPI_Finalize();
        return 1;
    }
    
    /* END CREATE LOCAL SUB-ARRAYS */
    
    
    /* BEGIN CREATE TYPES */
    
    // Blocks fo distributing array. floats for data, chars for flags.
    MPI_Datatype f_block;
    MPI_Datatype c_block;
    MPI_Type_vector(block_x+2, block_y+2, jmax+3, MPI_FLOAT, &f_block);
    MPI_Type_vector(block_x+2, block_y+2, jmax+3, MPI_CHAR, &c_block);
    MPI_Type_commit(&f_block);
    MPI_Type_commit(&c_block);
    
    // Blocks for sending/receiving sub-arrays, exclude the boundaries.
    MPI_Datatype g_f_block;
    MPI_Datatype l_f_block;
    MPI_Type_vector(block_x, block_y, jmax+3, MPI_FLOAT, &g_f_block);
    MPI_Type_vector(block_x, block_y, block_y+2, MPI_FLOAT, &l_f_block);
    MPI_Type_commit(&g_f_block);
    MPI_Type_commit(&l_f_block);
    
    // Block edges in each direction.
    MPI_Datatype f_x_edge;
    MPI_Datatype f_y_edge;
    MPI_Type_vector(block_x, 1, block_y+2, MPI_FLOAT, &f_x_edge);
    MPI_Type_vector(1, block_y, block_y+2, MPI_FLOAT, &f_y_edge);
    MPI_Type_commit(&f_x_edge);
    MPI_Type_commit(&f_y_edge);
    
    /* END CREATE TYPES */
    
    
    /* BEGIN CREATE COMMUNICATION ARRAYS */
    
    // Alltoallw arrays for distributing flags.
    int* global_count_flag = calloc(nprocs, sizeof(int));
    int* global_displ_flag = calloc(nprocs, sizeof(int));
    MPI_Datatype* global_type_flag = malloc(nprocs * sizeof(MPI_Datatype));
    int* local_count_flag = calloc(nprocs, sizeof(int));
    int* local_displ_flag = calloc(nprocs, sizeof(int));
    MPI_Datatype* local_type_flag = malloc(nprocs * sizeof(MPI_Datatype));
    
    // Alltoallw arrays for distributing data.
    int* global_count_data_d = calloc(nprocs, sizeof(int));
    int* global_displ_data_d = calloc(nprocs, sizeof(int));
    MPI_Datatype* global_type_data_d = malloc(nprocs * sizeof(MPI_Datatype));
    int* local_count_data_d = calloc(nprocs, sizeof(int));
    int* local_displ_data_d = calloc(nprocs, sizeof(int));
    MPI_Datatype* local_type_data_d = malloc(nprocs * sizeof(MPI_Datatype));
    
    // Alltoallw arrays for collecting data.
    int* global_count_data_c = calloc(nprocs, sizeof(int));
    int* global_displ_data_c = calloc(nprocs, sizeof(int));
    MPI_Datatype* global_type_data_c = malloc(nprocs * sizeof(MPI_Datatype));
    int* local_count_data_c = calloc(nprocs, sizeof(int));
    int* local_displ_data_c = calloc(nprocs, sizeof(int));
    MPI_Datatype* local_type_data_c = malloc(nprocs * sizeof(MPI_Datatype));
    
    // Alltoallw arrays for exchanging edges.
    sendcounts_edge = calloc(nprocs, sizeof(int));
    senddispls_edge = calloc(nprocs, sizeof(int));
    sendtypes_edge = malloc(nprocs * sizeof(MPI_Datatype));
    recvcounts_edge = calloc(nprocs, sizeof(int));
    recvdispls_edge = calloc(nprocs, sizeof(int));
    recvtypes_edge = malloc(nprocs * sizeof(MPI_Datatype));
    
    for (i = 0; i < nprocs; i++) {
        local_type_flag[i] = MPI_CHAR;
        local_type_data_d[i] = MPI_FLOAT;
        local_type_data_c[i] = l_f_block;
        global_type_data_c[i] = g_f_block;
        // Need to be initialised on all even though only rank 0 uses them.
        global_type_flag[i] = c_block;
        global_type_data_d[i] = f_block;
        // Arbitrary initialisation to prevent errros caused by null.
        // Relevant positions are set elsewhere.
        sendtypes_edge[i] = MPI_CHAR;
        recvtypes_edge[i] = MPI_CHAR;
    }
    
    // All processors receive block from rank 0.
    local_count_flag[0] = (block_x+2) * (block_y+2);
    local_count_data_d[0] = (block_x+2) * (block_y+2);
    local_count_data_c[0] = 1;
    local_displ_data_c[0] = (&(l_p[1][1]) - &(l_p[0][0])) * sizeof(float);
    
    // Rank 0 needs block sizes and start points from each other process
    // to determine their block types and offset.
    int* x_size = NULL;
    int* y_size = NULL;
    int* x_start = NULL;
    int* y_start = NULL;
    if (proc == 0) {
        x_size = malloc(sizeof(int) * nprocs);
        y_size = malloc(sizeof(int) * nprocs);
        x_start = malloc(sizeof(int) * nprocs);
        y_start = malloc(sizeof(int) * nprocs);
    }
    MPI_Gather(&l_imax, 1, MPI_INT, x_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&l_jmax, 1, MPI_INT, y_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&istart, 1, MPI_INT, x_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&jstart, 1, MPI_INT, y_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (proc == 0) {
        // Blank line to separate verbose prints.
        if (verbose > 2) printf("\n");
        
        for (i = 0; i < nprocs; i++) {
            // Always sends one block.
            global_count_flag[i] = 1;
            global_count_data_d[i] = 1;
            
            // Get displacements by differencing addersses.
            int x = x_start[i];
            int y = y_start[i];
            global_displ_flag[i] = &(flag[x][y]) - &(flag[0][0]);
            global_displ_data_d[i] = (&(p[x][y]) - &(p[0][0])) * sizeof(float);
            
            global_count_data_c[i] = 1;
            // +1s account for dropping local borders on collection.
            global_displ_data_c[i] = (&(p[x+1][y+1]) - &(p[0][0])) * sizeof(float);
            
            if (verbose > 2) {
                printf("Rank %d will get %dx%d starting at (%d, %d).\n",
                    i, x_size[i], y_size[i], x_start[i], y_start[i]);
            }
        }
        
        // Blank line to separate verbose prints.
        if (verbose > 2) printf("\n");
    }
    
    // Calculate neighbor edge send patterns.
    // Exchange at left edge if not on boundary.
    if (proc_left >= 0) {
        sendcounts_edge[proc_left] = 1;
        recvcounts_edge[proc_left] = 1;
        sendtypes_edge[proc_left] = f_y_edge;
        recvtypes_edge[proc_left] = f_y_edge;
        senddispls_edge[proc_left] = (&(l_p[1][1]) - &(l_p[0][0])) * sizeof(float);
        recvdispls_edge[proc_left] = (&(l_p[0][1]) - &(l_p[0][0])) * sizeof(float);
    }
    // Exchange at left edge if not on boundary.
    if (proc_right >= 0) {
        sendcounts_edge[proc_right] = 1;
        recvcounts_edge[proc_right] = 1;
        sendtypes_edge[proc_right] = f_y_edge;
        recvtypes_edge[proc_right] = f_y_edge;
        senddispls_edge[proc_right] = (&(l_p[l_imax][1]) - &(l_p[0][0])) * sizeof(float);
        recvdispls_edge[proc_right] = (&(l_p[l_imax+1][1]) - &(l_p[0][0])) * sizeof(float);
    }
    // Exchange at left edge if not on boundary.
    if (proc_up >= 0) {
        sendcounts_edge[proc_up] = 1;
        recvcounts_edge[proc_up] = 1;
        sendtypes_edge[proc_up] = f_x_edge;
        recvtypes_edge[proc_up] = f_x_edge;
        senddispls_edge[proc_up] = (&(l_p[1][1]) - &(l_p[0][0])) * sizeof(float);
        recvdispls_edge[proc_up] = (&(l_p[1][0]) - &(l_p[0][0])) * sizeof(float);
    }
    // Exchange at left edge if not on boundary.
    if (proc_down >= 0) {
        sendcounts_edge[proc_down] = 1;
        recvcounts_edge[proc_down] = 1;
        sendtypes_edge[proc_down] = f_x_edge;
        recvtypes_edge[proc_down] = f_x_edge;
        senddispls_edge[proc_down] = (&(l_p[1][l_jmax]) - &(l_p[0][0])) * sizeof(float);
        recvdispls_edge[proc_down] = (&(l_p[1][l_jmax+1]) - &(l_p[0][0])) * sizeof(float);
    }
    
    /* END CREATE COMMUNICATION ARRAYS */
    
    MPI_Barrier(MPI_COMM_WORLD);
    double init_end = MPI_Wtime();
    
    
    
    double velocity_time = 0;
    double poisson_time = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double main_start = MPI_Wtime();

    // Distribute flags once as they don't change.
    MPI_Alltoallw(flag[0], global_count_flag, global_displ_flag,
        global_type_flag, l_flag[0], local_count_flag, local_displ_flag,
        local_type_flag, MPI_COMM_WORLD);
    
    /* Main loop */
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        if (proc == 0) {
            set_timestep_interval(&del_t, imax, jmax, delx, dely,
                u, v, Re, tau);
            
            ifluid = (imax * jmax) - ibound;
            
            double velocity_start = MPI_Wtime();
            
            compute_tentative_velocity(u, v, f, g, flag, imax, jmax,
                del_t, delx, dely, gamma, Re);
            
            double velocity_end = MPI_Wtime();
            velocity_time += velocity_end - velocity_start;
            
            compute_rhs(f, g, rhs, flag, imax, jmax, del_t, delx, dely);
        }
        
        MPI_Bcast(&ifluid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&del_t, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (ifluid > 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            double poisson_start = MPI_Wtime();
            
            MPI_Alltoallw(p[0], global_count_data_d, global_displ_data_d,
                global_type_data_d, l_p[0], local_count_data_d,
                local_displ_data_d, local_type_data_d, MPI_COMM_WORLD);
            MPI_Alltoallw(rhs[0], global_count_data_d, global_displ_data_d,
                global_type_data_d, l_rhs[0], local_count_data_d,
                local_displ_data_d, local_type_data_d, MPI_COMM_WORLD);
            
            itersor = poisson(l_p, l_rhs, l_flag, l_imax, l_jmax, delx, dely,
                        eps, itermax, omega, &res, ifluid);
            
            MPI_Alltoallw(l_p[0], local_count_data_c, local_displ_data_c,
                local_type_data_c, p[0], global_count_data_c,
                global_displ_data_c, global_type_data_c, MPI_COMM_WORLD);
            
            MPI_Barrier(MPI_COMM_WORLD);
            double poisson_end = MPI_Wtime();
            poisson_time += poisson_end - poisson_start;
        } else {
            itersor = 0;
        }

        if (proc == 0) {
            if (verbose > 1) {
                printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                    iters, t+del_t, del_t, itersor, res, ibound);
            }

            update_velocity(u, v, f, g, p, flag, imax, jmax, del_t, delx, dely);

            apply_boundary_conditions(u, v, flag, imax, jmax, ui, vi);
        }
    } /* End of main loop */
    
    MPI_Barrier(MPI_COMM_WORLD);
    double main_end = MPI_Wtime();
    
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    double finl_start = MPI_Wtime();

    /* BEGIN FREE COMMUNICATION ARRAYS */
    
    if (global_count_flag) free(global_count_flag);
    if (global_displ_flag) free(global_displ_flag);
    if (global_type_flag) free(global_type_flag);
    if (local_count_flag) free(local_count_flag);
    if (local_displ_flag) free(local_displ_flag);
    if (local_type_flag) free(local_type_flag);
    
    if (global_count_data_d) free(global_count_data_d);
    if (global_displ_data_d) free(global_displ_data_d);
    if (global_type_data_d) free(global_type_data_d);
    if (local_count_data_d) free(local_count_data_d);
    if (local_displ_data_d) free(local_displ_data_d);
    if (local_type_data_d) free(local_type_data_d);
    
    if (sendcounts_edge) free(sendcounts_edge);
    if (senddispls_edge) free(senddispls_edge);
    if (sendtypes_edge) free(sendtypes_edge);
    if (recvcounts_edge) free(recvcounts_edge);
    if (recvdispls_edge) free(recvdispls_edge);
    if (recvtypes_edge) free(recvtypes_edge);
    
    if (x_size) free(x_size);
    if (y_size) free(y_size);
    if (x_start) free(x_start);
    if (y_start) free(y_start);
    
    /* END FREE COMMUNICATION ARRAYS */
    
    /* BEGIN FREE LOCAL SUB-ARRAYS */
    
    if (l_p) free_matrix(l_p);
    if (l_rhs) free_matrix(l_rhs);
    if (l_flag) free_matrix(l_flag);
    
    /* END FREE LOCAL SUB-ARRAYS */
    
    /* BEGIN FREE DATA TYPES */
    
    MPI_Type_free(&f_block);
    MPI_Type_free(&c_block);
    
    MPI_Type_free(&g_f_block);
    MPI_Type_free(&l_f_block);
    
    MPI_Type_free(&f_x_edge);
    MPI_Type_free(&f_y_edge);
    
    /* END FREE DATA TYPES */
    
    /* BEGIN FINALISE PROBLEM */
    
    if (proc == 0) {
        if (outfile != NULL && strcmp(outfile, "") != 0 && proc == 0) {
            write_bin(u, v, p, flag, imax, jmax, xlength, ylength, outfile);
        }
    }
    
    if (u) free_matrix(u);
    if (v) free_matrix(v);
    if (f) free_matrix(f);
    if (g) free_matrix(g);
    if (p) free_matrix(p);
    if (rhs) free_matrix(rhs);
    if (flag) free_matrix(flag);

    /* END FINALISE PROBLEM */
    
    MPI_Barrier(MPI_COMM_WORLD);
    double finl_end = MPI_Wtime();
    double end = MPI_Wtime();
    
    if (proc == 0) {
        // Human format.
        printf("Timings for %d processors on '%s'.\n\n", nprocs, infile);
        printf("\nInit time: %fs.\n", init_end - init_start);
        printf("Main time: %fs.\n", main_end - main_start);
        printf("Velocity time: %fs.\n", velocity_time);
        printf("Poisson time: %fs.\n", poisson_time);
        printf("Finalise time: %fs.\n", finl_end - finl_start);
        printf("\nTotal time %fs.\n", end - start);
        
        // CSV format.
        // printf("%d,%s,%f,%f,%f,%f,%f,%f\n", nprocs, infile,
        //     init_end - init_start, main_end - main_start, velocity_time,
        //     poisson_time, finl_end - finl_start, end - start);
    }
    
    MPI_Finalize();
    
    return 0;
}

/* 
 * Finds the optimal 2d decomposition of a matrix with dimensions imax x jmax
 * on nprocs prcessors and stores the result in dims. This considers 1d
 * decompositions as 1 x nprocs and nprocs x 1 2d decompositions and as such
 * can decide on a 1d decomposition. It defines the optimal decomposition as
 * the one which gives sub-arrays which are closest to squares. It only
 * considers decompositions which use all processors, as this keeps the
 * processor load balanced.
 */
void decompose_2d(int imax, int jmax, int nprocs, int dims[2]) {
    float opt_ratio = 0.0;

    int i;
    for (i = 1; i <= nprocs; i++) {
        int x = i;
        int y = nprocs / i;
        
        if (x * y == nprocs) {
            float sub_arr_x = imax / (float)x;
            float sub_arr_y = jmax / (float)y;
            
            float ratio = sub_arr_y / sub_arr_x;
            
            // Consider version of ratio which is less than (or equal to) 1...
            if (ratio > 1.0) {
                ratio = 1.0 / ratio;
            }
            // ...so that finding the 'squarest' becomes finding the greatest.
            if (ratio > opt_ratio) {
                opt_ratio = ratio;
                dims[0] = x;
                dims[1] = y;
            }
        }
    }
}


/* Save the simulation state to a file */
void write_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i;
    FILE *fp;

    fp = fopen(file, "wb"); 

    if (fp == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        return;
    }

    fwrite(&imax, sizeof(int), 1, fp);
    fwrite(&jmax, sizeof(int), 1, fp);
    fwrite(&xlength, sizeof(float), 1, fp);
    fwrite(&ylength, sizeof(float), 1, fp);

    for (i=0;i<imax+2;i++) {
        fwrite(u[i], sizeof(float), jmax+2, fp);
        fwrite(v[i], sizeof(float), jmax+2, fp);
        fwrite(p[i], sizeof(float), jmax+2, fp);
        fwrite(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
}

/* Read the simulation state from a file */
int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i,j;
    FILE *fp;

    if (file == NULL) return -1;

    if ((fp = fopen(file, "rb")) == NULL) {
        // fprintf(stderr, "Could not open file '%s': %s\n", file,
        //     strerror(errno));
        // fprintf(stderr, "Generating default state instead.\n");
        return -1;
    }

    fread(&i, sizeof(int), 1, fp);
    fread(&j, sizeof(int), 1, fp);
    float xl, yl;
    fread(&xl, sizeof(float), 1, fp);
    fread(&yl, sizeof(float), 1, fp);

    if (i!=imax || j!=jmax) {
        fprintf(stderr, "Warning: imax/jmax have wrong values in %s\n", file);
        fprintf(stderr, "%s's imax = %d, jmax = %d\n", file, i, j);
        fprintf(stderr, "Program's imax = %d, jmax = %d\n", imax, jmax);
        return 1;
    }
    if (xl!=xlength || yl!=ylength) {
        fprintf(stderr, "Warning: xlength/ylength have wrong values in %s\n", file);
        fprintf(stderr, "%s's xlength = %g,  ylength = %g\n", file, xl, yl);
        fprintf(stderr, "Program's xlength = %g, ylength = %g\n", xlength,
            ylength);
        return 1;
    }

    for (i=0; i<imax+2; i++) {
        fread(u[i], sizeof(float), jmax+2, fp);
        fread(v[i], sizeof(float), jmax+2, fp);
        fread(p[i], sizeof(float), jmax+2, fp);
        fread(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
    return 0;
}

static void print_usage(void)
{
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
}

static void print_version(void)
{
    fprintf(stderr, "%s %s\n", PACKAGE, VERSION);
}

static void print_help(void)
{
    fprintf(stderr, "%s. A simple computational fluid dynamics tutorial.\n\n",
        PACKAGE);
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", progname);
    fprintf(stderr, "  -h, --help            Print a summary of the options\n");
    fprintf(stderr, "  -V, --version         Print the version number\n");
    fprintf(stderr, "  -v, --verbose=LEVEL   Set the verbosity level. 0 is silent\n");
    fprintf(stderr, "  -x, --imax=IMAX       Set the number of interior cells in the X direction\n");
    fprintf(stderr, "  -y, --jmax=JMAX       Set the number of interior cells in the Y direction\n");
    fprintf(stderr, "  -t, --t-end=TEND      Set the simulation end time\n");
    fprintf(stderr, "  -d, --del-t=DELT      Set the simulation timestep size\n");
    fprintf(stderr, "  -i, --infile=FILE     Read the initial simulation state from this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
    fprintf(stderr, "  -o, --outfile=FILE    Write the final simulation state to this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
}
