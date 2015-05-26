#define _XOPEN_SOURCE 600

#include <errno.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "atom.h"
#include "sotl.h"
#include "default_defines.h"
#include "ocl.h"
#include "seq.h"
#include "ocl_kernels.h"
#include "window.h"
#include "profiling.h"

#ifdef HAVE_LIBGL
#include "vbo.h"
#include "shaders.h"
#endif

static sotl_atom_set_t global_atom_set;

static void sift_down(sotl_atom_set_t *set, const int start, const int count);

#define SWAP(a, b)                  \
    do {                            \
        calc_t t = a; a = b; b = t; \
    } while(0)

#define SWAP_ATOM(a, b)                             \
    do {                                            \
        SWAP(set->pos.z[a], set->pos.z[b]);         \
        SWAP(set->pos.y[a], set->pos.y[b]);         \
        SWAP(set->pos.x[a], set->pos.x[b]);         \
        SWAP(set->speed.dz[a], set->speed.dz[b]);   \
        SWAP(set->speed.dy[a], set->speed.dy[b]);   \
        SWAP(set->speed.dx[a], set->speed.dx[b]);   \
    } while(0)

static void heap_sort(sotl_atom_set_t *set, const unsigned count)
{
    int start, end;

    /* heapify */
    start = (count - 2) / 2;
    for (; start >= 0; start--) {
        sift_down(set, start, count);
    }

    for (end = count - 1; end > 0; end--) {
        SWAP_ATOM(end, 0);
        sift_down(set, 0, end);
    }
}

static void sift_down(sotl_atom_set_t *set, const int start, const int end)
{
    int root = start;

    while (root * 2 + 1 < end) {
        int child = 2 * root + 1;
        if ((child + 1 < end) && (set->pos.z[child] < set->pos.z[child + 1])) {
            child += 1;
        }
        if (set->pos.z[root] >= set->pos.z[child])
            return;

        SWAP_ATOM(child, root);
        root = child;
    }
}

int atom_set_init(sotl_atom_set_t *set, const unsigned long natoms,
                  const unsigned long maxatoms)
{
    if (maxatoms < natoms)
        return SOTL_INVALID_VALUE;

    set->natoms  = natoms;
    set->current = 0;
    set->offset  = ROUND(maxatoms);

    if (!sotl_have_multi()) {
        /* No need to have ghosts for a single device. */
        /* XXX: This is absolutely wrong when the torus mode is enabled. :-) */
        set->offset_ghosts = 0;
    } else {
        set->offset_ghosts = ROUND(set->natoms * 0.05);
    }

    /* No ghosts at the beginning. */
    set->nghosts_min = set->nghosts_max = 0;

    set->pos.x = malloc(atom_set_size(set));
    if (!set->pos.x)
        return SOTL_OUT_OF_MEMORY;
    set->pos.y = set->pos.x + set->offset;
    set->pos.z = set->pos.y + set->offset;

    set->speed.dx = malloc(atom_set_size(set));
    if (!set->speed.dx) {
        atom_set_free(set);
        return SOTL_OUT_OF_MEMORY;
    }
    set->speed.dy = set->speed.dx + set->offset;
    set->speed.dz = set->speed.dy + set->offset;

    return SOTL_SUCCESS;
}

void atom_set_print(const sotl_atom_set_t *set)
{
    sotl_log(DEBUG, "natoms = %d, current = %d, offset = %d\n",
             set->natoms, set->current, set->offset);
    sotl_log(DEBUG, "pos.x = %p, pos.y = %p, pos.z = %p\n", set->pos.x,
             set->pos.y, set->pos.z);
}

sotl_atom_set_t *get_global_atom_set()
{
    return &global_atom_set;
}

int atom_set_add(sotl_atom_set_t *set, const calc_t x, const calc_t y,
                 const calc_t z, const calc_t dx, const calc_t dy,
                 const calc_t dz)
{
    if (set->current >= set->natoms)
        return SOTL_INVALID_BUFFER_SIZE;

    set->pos.x[set->current] = x;
    set->pos.y[set->current] = y;
    set->pos.z[set->current] = z;

    set->speed.dx[set->current] = dx;
    set->speed.dy[set->current] = dy;
    set->speed.dz[set->current] = dz;

    set->current++;
    return SOTL_SUCCESS;
}

size_t atom_set_offset(const sotl_atom_set_t *set)
{
    return set->offset + set->offset_ghosts * 2;
}

size_t atom_set_size(sotl_atom_set_t *set)
{
    return sizeof(calc_t) * set->offset * 3;
}

size_t atom_set_border_size(const sotl_atom_set_t *set)
{
    return sizeof(calc_t) * set->offset_ghosts * 3;
}

size_t atom_set_begin(const sotl_atom_set_t *set)
{
    return set->offset_ghosts;
}

size_t atom_set_end(const sotl_atom_set_t *set)
{
    return atom_set_begin(set) + set->natoms;
}

#define FREE(x)     \
    do {            \
        free(x);    \
        x = NULL;   \
    } while (0)

void atom_set_free(sotl_atom_set_t *set)
{
    FREE(set->pos.x);
    FREE(set->speed.dx);
}

void atom_set_sort(sotl_atom_set_t *set)
{
    /* Sort atoms along z-axis. */
    heap_sort(set, set->natoms);
}


void bubble_sort_parallel(sotl_atom_set_t *set, const  int N){

	
		int step, i ;
	//#pragma omp parallel private(step)
		for (step = N; step > 0; step--) {
			if (step % 2 == 0) {
	#pragma omp for private(i)
				for (i = 0; i < N-1; i += 2)
					if (set->pos.z[i] > set->pos.z[i+1]) {
						SWAP_ATOM(i, i+1);
					}
			} else {
	#pragma omp for private(i)
				for (i = 1; i < N-1; i += 2)
						if (set->pos.z[i] > set->pos.z[i+1]) {
							SWAP_ATOM(i, i+1);
						} 
			} 
		}

	}



void atom_set_sort_parallel(sotl_atom_set_t *set)
{
    /* Sort atoms along z-axis. */
    bubble_sort_parallel(set, set->natoms);
}


int * atom_sort_boxes(sotl_atom_set_t *set, sotl_domain_t *domain)
{
//  sotl_atom_set_t *set = &dev->atom_set;
//  sotl_domain_t *domain = &dev->domain;

	//int out[set->natoms];

	int * boxes = atom_set_box_count(domain, set);


	for (int i = 1; i < (int) domain -> total_boxes; i++){
	  boxes[i] += boxes[i-1];
	}
	
	int count_boxes[domain->total_boxes];
	for (int i = 0 ; i < (int) domain->total_boxes ; i++){
		count_boxes[i] = 0;
	}

	int pos_set = 0;
	int num_box = 0;
	int next_box = boxes[0];
	int num_box_tmp;
	
	while (pos_set < (int) set->natoms){
	  while (pos_set < next_box){
	    num_box_tmp = atom_get_num_box(domain, set->pos.x[pos_set], set->pos.y[pos_set], set->pos.z[pos_set], BOX_SIZE_INV);
	    if( num_box_tmp != num_box){
	      SWAP_ATOM(pos_set, boxes[num_box_tmp - 1] + count_boxes[num_box_tmp]);
	      count_boxes[num_box_tmp]++;
	    }
	    else {
	      pos_set++;
	      count_boxes[num_box]++;
	    }
	  }
	  num_box ++;
	  if(num_box < (int) domain-> total_boxes){
	    next_box = boxes[num_box];
	    pos_set += count_boxes[num_box];
	  }	
	}
	
	return boxes;
}



#ifdef HAVE_LIBGL
void atom_build (int natoms, sotl_atom_pos_t * pos_vec)
{
    int i;
    for (i = 0; i < natoms; i++)
        vbo_add_atom (pos_vec->x[i], pos_vec->y[i], pos_vec->z[i]);
}
#endif

int atom_get_num_box(const sotl_domain_t *dom, const calc_t x, const calc_t y,
                     const calc_t z, const calc_t rrc)
{
    int box_x, box_y, box_z;
    int box_id;

    box_x = (x - dom->min_border[0]) * rrc;
    box_y = (y - dom->min_border[1]) * rrc;
    box_z = (z - dom->min_border[2]) * rrc;

    box_id =  box_z * dom->boxes[0] * dom->boxes[1] +
              box_y * dom->boxes[0] +
              box_x;

    //assert(box_id >= 0 && (unsigned)box_id < dom->total_boxes);

	if (box_id < 0 || (unsigned)box_id >= dom->total_boxes){
		return 0;
	}
	return box_id;
}

int *atom_set_box_count(const sotl_domain_t *dom, const sotl_atom_set_t *set)
{
    int *boxes = NULL;
    size_t size;

    size = dom->total_boxes * sizeof(int);
    if (!(boxes = calloc(1, size)))
        return NULL;

	//printf("\n\n\n COUNT \n\n\n");

    for (unsigned i = 0; i < set->natoms; i++) {

      int box_id = atom_get_num_box(dom, set->pos.x[i], set->pos.y[i], set->pos.z[i],
				    BOX_SIZE_INV);


	if (box_id >= 0){
    
  		boxes[box_id]++;
		}
    }

    return boxes;
}
