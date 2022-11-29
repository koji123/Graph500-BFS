/*
 * main.cc
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 */

// C includes
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#ifdef _FUGAKU_POWER_MEASUREMENT
#include "pwr.h"
#endif

// C++ includes
#include <string>

#include "parameters.h"
#include "utils_core.h"
#include "primitives.hpp"
#include "utils.hpp"
#include "../generator/graph_generator.hpp"
#include "graph_constructor.hpp"
#include "validate.hpp"
#include "benchmark_helper.hpp"
#include "bfs.hpp"
#include "bfs_cpu.hpp"
#if CUDA_ENABLED
#include "bfs_gpu.hpp"
#endif

void graph500_bfs(int SCALE, int edgefactor, double alpha, double beta, int validation_level,
                  bool pre_exec, bool real_benchmark)
{
	using namespace PRM;
	SET_AFFINITY;

        int num_bfs_roots = (real_benchmark)? REAL_BFS_ROOTS : TEST_BFS_ROOTS;
	double bfs_times[num_bfs_roots], validate_times[num_bfs_roots], edge_counts[num_bfs_roots];
	LogFileFormat log = {0};
	int root_start = read_log_file(&log, SCALE, edgefactor, bfs_times, validate_times, edge_counts);
	if(mpi.isMaster() && root_start != 0)
		print_with_prefix("Resume from %d th run", root_start);

	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
			(int64_t(1) << SCALE) * edgefactor / mpi.size_2d, getenv("TMPFILE"));

	BfsOnCPU::printInformation(validation_level, pre_exec, real_benchmark);

	if(mpi.isMaster()) print_with_prefix("Graph generation");
	double generation_time = MPI_Wtime();
	generate_graph_spec2010(&edge_list, SCALE, edgefactor);
	generation_time = MPI_Wtime() - generation_time;

	if(mpi.isMaster()) print_with_prefix("Graph construction");
	// Create BFS instance and the *COMMUNICATION THREAD*.
	BfsOnCPU* benchmark = new BfsOnCPU();
	double construction_time = MPI_Wtime();
	benchmark->construct(&edge_list);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) print_with_prefix("Redistributing edge list...");
	double redistribution_time = MPI_Wtime();
	redistribute_edge_2d(&edge_list);
	redistribution_time = MPI_Wtime() - redistribution_time;

	int64_t bfs_roots[num_bfs_roots];
	find_roots(benchmark->graph_, bfs_roots, num_bfs_roots);
	const int64_t max_used_vertex = find_max_used_vertex(benchmark->graph_);
	const int64_t nlocalverts = benchmark->graph_.pred_size();

	int64_t *pred = static_cast<int64_t*>(
		cache_aligned_xmalloc(nlocalverts*sizeof(pred[0])));

#if INIT_PRED_ONCE	// Only Spec2010 needs this initialization
#pragma omp parallel for
	for(int64_t i = 0; i < nlocalverts; ++i) {
		pred[i] = -1;
	}
#endif

	bool result_ok = true;

	if(root_start == 0)
		init_log(SCALE, edgefactor, generation_time, construction_time, redistribution_time, &log);

	benchmark->prepare_bfs(validation_level, pre_exec, real_benchmark);
	
	if(pre_exec){
#ifdef _FUGAKU_POWER_MEASUREMENT
	  PWR_Cntxt cntxt = NULL;
	  PWR_Obj obj = NULL;
	  int rc;
	  double energy1 = 0.0;
	  double energy2 = 0.0;
	  double menergy1 = 0.0;
	  double menergy2 = 0.0;
	  double ave_power[2];
	  double t_power[2];
	  PWR_Time ts1 = 0;
	  PWR_Time ts2 = 0;
	  rc = PWR_CntxtInit(PWR_CNTXT_FX1000, PWR_ROLE_APP, "app", &cntxt);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("CntxtInit Failed\n");
	  }
	  rc = PWR_CntxtGetObjByName(cntxt, "plat.node", &obj);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("CntxtGetObjByName Failed\n");
	  }
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_MEASURED_ENERGY, &menergy1, &ts1);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy1, NULL);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
#endif
      if(mpi.isMaster()){
        time_t t = time(NULL);
        print_with_prefix("Start energy loop : %s", ctime(&t));
      }
	  double time_left = PRE_EXEC_TIME;
	  for(int c = root_start; time_left > 0.0; ++c) {
		if(mpi.isMaster())
		  print_with_prefix("========== Pre Running BFS %d ==========", c);
		MPI_Barrier(mpi.comm_2d);
		double bfs_time = MPI_Wtime();
		benchmark->run_bfs(bfs_roots[c % num_bfs_roots], pred, edgefactor, alpha, beta);
		bfs_time = MPI_Wtime() - bfs_time;
		if(mpi.isMaster()) {
		  print_with_prefix("Time for BFS %d is %f", c, bfs_time);
		  time_left -= bfs_time;
		}
		MPI_Bcast(&time_left, 1, MPI_DOUBLE, 0, mpi.comm_2d);
	  }
      if(mpi.isMaster()){
        time_t t = time(NULL);
        print_with_prefix("End energy loop : %s", ctime(&t));
      }
#ifdef _FUGAKU_POWER_MEASUREMENT
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_MEASURED_ENERGY, &menergy2, &ts2);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
	  rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy2, NULL);
	  if (rc != PWR_RET_SUCCESS) {
	    printf("ObjAttrGetValue Failed (rc = %d)\n", rc);
	  }
	  ave_power[0] = (menergy2 - menergy1) / ((ts2 - ts1) / 1000000000.0);
	  ave_power[1] = (energy2 - energy1) / ((ts2 - ts1) / 1000000000.0);
	  MPI_Reduce(ave_power, t_power, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	  if(mpi.isMaster()){
        print_with_prefix("total measured average power : %lf", t_power[0]);
        print_with_prefix("total estimated average power : %lf", t_power[1]);
	  }
#endif
	}
/////////////////////
#ifdef PROFILE_REGIONS
	timer_clear();
#endif
	for(int i = root_start; i < num_bfs_roots; ++i) {
		VERVOSE(print_max_memory_usage());

		if(mpi.isMaster())  print_with_prefix("========== Running BFS %d ==========", i);
#if ENABLE_FUJI_PROF
		fapp_start("bfs", i, 1);
#endif
		MPI_Barrier(mpi.comm_2d);
#if FUGAKU_MPI_PRINT_STATS
		FJMPI_Collection_start();
#endif
		PROF(profiling::g_pis.reset());
		bfs_times[i] = MPI_Wtime();
		benchmark->run_bfs(bfs_roots[i], pred, edgefactor, alpha, beta);
		bfs_times[i] = MPI_Wtime() - bfs_times[i];
#if FUGAKU_MPI_PRINT_STATS
                FJMPI_Collection_stop();
#endif
#if ENABLE_FUJI_PROF
		fapp_stop("bfs", i, 1);
#endif
		PROF(profiling::g_pis.printResult());
		if(mpi.isMaster()) {
			print_with_prefix("Time for BFS %d is %f", i, bfs_times[i]);
			print_with_prefix("Validating BFS %d", i);
		}

		benchmark->get_pred(pred);

		validate_times[i] = MPI_Wtime();
		int64_t edge_visit_count = 0;
                if(validation_level == 2){
                  result_ok = validate_bfs_result(&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
                }
                else if(validation_level == 1){
                  if(i == 0) {
			result_ok = validate_bfs_result(&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
			pf_nedge[SCALE] = edge_visit_count;
                  }
                  else {
                    edge_visit_count = pf_nedge[SCALE];
                  }
                }
                else{ // validation_level == 0
                  edge_visit_count = pf_nedge[SCALE];
                }

		validate_times[i] = MPI_Wtime() - validate_times[i];
		edge_counts[i] = (double)edge_visit_count;

		if(mpi.isMaster()) {
			print_with_prefix("Validate time for BFS %d is %f", i, validate_times[i]);
			print_with_prefix("Number of traversed edges is %" PRId64 "", edge_visit_count);
			print_with_prefix("TEPS for BFS %d is %g", i, edge_visit_count / bfs_times[i]);
		}

		if(result_ok == false) {
			break;
		}

		update_log_file(&log, bfs_times[i], validate_times[i], edge_visit_count);
	}
	benchmark->end_bfs();

	if(mpi.isMaster()) {
	  fprintf(stdout, "============= Result ==============\n");
	  fprintf(stdout, "SCALE:                          %d\n", SCALE);
	  fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
          fprintf(stdout, "alpha:                          %f\n", alpha);
          fprintf(stdout, "beta:                           %f\n", beta);
	  fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots);
	  fprintf(stdout, "graph_generation:               %g\n", generation_time);
	  fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size_2d);
	  fprintf(stdout, "construction_time:              %g\n", construction_time);
	  fprintf(stdout, "redistribution_time:            %g\n", redistribution_time);
	  print_bfs_result(num_bfs_roots, bfs_times, validate_times, edge_counts, result_ok);
	}
#ifdef PROFILE_REGIONS
	timer_print(bfs_times, num_bfs_roots);
#endif

#if FUGAKU_MPI_PRINT_STATS
        FJMPI_Collection_print(const_cast<char *>("Communication Statistics\n"));
#endif

	delete benchmark;

	free(pred);
}
#if 0
void test02(int SCALE, int edgefactor)
{
	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
			(INT64_C(1) << SCALE) * edgefactor / mpi.size, getenv("TMPFILE"));
	RmatGraphGenerator<UnweightedPackedEdge> graph_generator(
//	RandomGraphGenerator<UnweightedPackedEdge> graph_generator(
				SCALE, edgefactor, 2, 3, InitialEdgeType::NONE);
	Graph2DCSR<Pack40bit, uint32_t> graph;

	double generation_time = MPI_Wtime();
	generate_graph(&edge_list, &graph_generator);
	generation_time = MPI_Wtime() - generation_time;

	double construction_time = MPI_Wtime();
	construct_graph(&edge_list, true, graph);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) {
		print_with_prefix("TEST02");
		fprintf(stdout, "SCALE:                          %d\n", SCALE);
		fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
		fprintf(stdout, "graph_generation:               %g\n", generation_time);
		fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size);
		fprintf(stdout, "construction_time:              %g\n", construction_time);
	}
}
#endif

#define ERROR(...) do{fprintf(stderr, __VA_ARGS__); exit(1);}while(0)
static void print_help(char *argv)
{
  ERROR("%s SCALE [-e edge_factor] [-a alpha] [-b beta] [-v validation level] [-P] [-R]\n", argv);
  // Validation Level: 0: No validation, 1: validate at first time only, 2: validate all results
  // Note: To conform to the specification, you must set 2
}

static void set_args(const int argc, char **argv, int *edge_factor, double *alpha, double *beta,
                     int *validation_level, bool *pre_exec, bool *real_benchmark)
{
  int result;
  while((result = getopt(argc,argv,"e:a:b:v:PR"))!=-1){
    switch(result){
    case 'e':
      *edge_factor = atoi(optarg);
      if(*edge_factor <= 0)
        ERROR("-e value > 0\n");
      break;
    case 'a':
      *alpha = atof(optarg);
      if(*alpha <= 0)
        ERROR("-a value > 0\n");
      break;
    case 'b':
      *beta = atof(optarg);
      if(*beta <= 0)
        ERROR("-b value > 0\n");
      break;
    case 'v':
      *validation_level = atoi(optarg);
      if(*validation_level < 0 || *validation_level > 2)
        ERROR("-v value >= 0 && value <= 2\n");
      break;
    case 'P':
      *pre_exec = true;
      break;
    case 'R':
      *real_benchmark = true;
      break;
    default:
      print_help(argv[0]);
    }
  }
}

int main(int argc, char** argv)
{
  if(argc <= 1 || atoi(argv[1]) <= 0)
    print_help(argv[0]);
  
  int scale            = atoi(argv[1]);
  int edge_factor      = DEFAULT_EDGE_FACTOR; // nedges / nvertices, i.e., 2*avg. degree
  double alpha         = DEFAULT_ALPHA;
  double beta          = DEFAULT_BETA;
  int validation_level = DEFAULT_VALIDATION_LEVEL;
  bool real_benchmark  = false;
  bool pre_exec        = false;

  set_args(argc, argv, &edge_factor, &alpha, &beta, &validation_level, &pre_exec, &real_benchmark);
  if(real_benchmark){
    validation_level = 2;
    pre_exec = true;
  }
  
  setup_globals(argc, argv, scale, edge_factor);
  graph500_bfs(scale, edge_factor, alpha, beta, validation_level, pre_exec, real_benchmark);
  cleanup_globals();
  return 0;
}

double elapsed[NUM_RESIONS], start[NUM_RESIONS];
void timer_clear()
{
  for(int i=0;i<NUM_RESIONS;i++)
    elapsed[i] = 0.0;
}

void timer_start(const int n)
{
  start[n] = MPI_Wtime();
}

void timer_stop(const int n)
{
  double now = MPI_Wtime();
  double t = now - start[n];
  elapsed[n] += t;
}

double timer_read(const int n)
{
  return(elapsed[n]);
}

void timer_print(double *bfs_times, const int num_bfs_roots)
{
  double t[NUM_RESIONS], t_max[NUM_RESIONS], t_min[NUM_RESIONS], t_ave[NUM_RESIONS];
  for(int i=0;i<NUM_RESIONS;i++)
    t[i] = timer_read(i);

  t[TOTAL_TIME] = 0.0;
  for(int i=0;i<num_bfs_roots;i++)
	t[TOTAL_TIME] += bfs_times[i];

  double comm_time = t[TD_EXPAND_TIME] + t[BU_EXPAND_TIME] + t[TD_FOLD_TIME] + t[BU_FOLD_TIME] + t[BU_NBR_TIME];
  t[CALC_TIME]	= (t[TD_TIME] + t[BU_TIME]) - comm_time - t[IMBALANCE_TIME];
  t[OTHER_TIME] = t[TOTAL_TIME] - (t[TD_TIME] + t[BU_TIME]);

  MPI_Reduce(t, t_max, NUM_RESIONS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(t, t_min, NUM_RESIONS, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(t, t_ave, NUM_RESIONS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for(int i=0;i<NUM_RESIONS;i++)
    t_ave[i] /= size;

  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi.isMaster()){
    printf("---\n");
    printf("CATEGORY :                 :   MAX    MIN    AVE   AVE/TIME\n");
    printf("TOTAL                      : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TOTAL_TIME));
    printf(" - TOP_DOWN                : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TD_TIME));
    printf(" - BOTTOM_UP               : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_TIME));
    printf("   - LOCAL_CALC            : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(CALC_TIME));
    printf("   - TD_EXPAND(allgather)  : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TD_EXPAND_TIME));
    printf("   - BU_EXPAND(allgather)  : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_EXPAND_TIME));
    printf("   - TD_FOLD(alltoall)     : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(TD_FOLD_TIME));
    printf("   - BU_FOLD(alltoall)     : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_FOLD_TIME));
    printf("   - BU_NEIGHBOR(sendrecv) : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(BU_NBR_TIME));
    printf("   - PROC_IMBALANCE        : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(IMBALANCE_TIME));
    printf(" - OTHER                   : %6.5f %6.5f %6.5f (%6.5f%%)\n", CAT(OTHER_TIME));
  }
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
}
