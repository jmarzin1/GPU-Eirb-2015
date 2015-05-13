
#include "default_defines.h"
#include "global_definitions.h"
#include "device.h"
#include "openmp.h"
#include "sotl.h"

#ifdef HAVE_LIBGL
#include "vbo.h"
#endif

#include <stdio.h>

static int *atom_state = NULL;

#ifdef HAVE_LIBGL

#define SHOCK_PERIOD  50

/************
 ** Boites **
 ************/


struct box {
  unsigned atoms[100000];
  unsigned size;
  unsigned nb_atoms;
};


struct box boxes[1000];
unsigned int nb_boxes;
unsigned int indY, indZ;


int init_boxes(){
  sotl_atom_set_t *set = get_global_atom_set();
  sotl_domain_t *domain = get_global_domain();

  nb_boxes = domain->boxes[0] * domain->boxes[1] * domain->boxes[2];
  unsigned i;
  indY = domain->boxes[0];
  indZ = domain->boxes[0] * domain->boxes[1];

  int nb_init = set->natoms;
  printf("%d %d\n", set->natoms, nb_init);
  for(i=0; i<nb_boxes;i++){
    boxes[i].nb_atoms = 0;
  }
  
  return EXIT_SUCCESS;
}

void sort_boxes(sotl_atom_set_t *set){
  int pos;
  sotl_domain_t *domain = get_global_domain();

  for(unsigned i=0; i<nb_boxes;i++){
    boxes[i].nb_atoms = 0;
  }
  int box_x, box_y, box_z;
  for(unsigned atom = 0; atom < set->natoms; atom++){
  
    box_x = (set->pos.x[atom] / BOX_SIZE);
  
    box_y = (set->pos.x[set->offset + atom] / BOX_SIZE);
  
    box_z = (set->pos.x[set->offset * 2 + atom] / BOX_SIZE);
    if(box_x >= 0 && box_y >= 0 && box_z >= 0 && box_x < domain->boxes[0] && box_y < domain->boxes[1] && box_z < domain->boxes[2]){
  
      pos = box_x + indY*box_y + indZ*box_z;
  
      boxes[pos].nb_atoms++;
  
      /* if(boxes[pos].nb_atoms >= boxes[pos].size){ */
      /*   boxes[pos].size *= 2; */
      /*   boxes[pos].atoms = realloc(boxes[pos].atoms, boxes[pos].size); */
      /* } */
      boxes[pos].atoms[boxes[pos].nb_atoms-1] = atom;
  
    }
  }
}

// Update OpenGL Vertex Buffer Object
//
static void omp_update_vbo (sotl_device_t *dev)
{
  sotl_atom_set_t *set = &dev->atom_set;
  sotl_domain_t *domain = &dev->domain;

  for (unsigned n = 0; n < set->natoms; n++) {
    vbo_vertex[n*3 + 0] = set->pos.x[n];
    vbo_vertex[n*3 + 1] = set->pos.y[n];
    vbo_vertex[n*3 + 2] = set->pos.z[n];

    // Atom color depends on z coordinate
    {
      float ratio = (set->pos.z[n] - domain->min_ext[2]) / (domain->max_ext[2] - domain->min_ext[2]);

      vbo_color[n*3 + 0] = (1.0 - ratio) * atom_color[0].R + ratio * 1.0;
      vbo_color[n*3 + 1] = (1.0 - ratio) * atom_color[0].G + ratio * 0.0;
      vbo_color[n*3 + 2] = (1.0 - ratio) * atom_color[0].B + ratio * 0.0;
      atom_state[n]--;
    }
  }
}
#endif

// Update positions of atoms by adding (dx, dy, dz)
//
static void omp_move (sotl_device_t *dev)
{
  sotl_atom_set_t *set = &dev->atom_set;

  for (unsigned n = 0; n < set->natoms; n++) {
    set->pos.x[n] += set->speed.dx[n];
    set->pos.y[n] += set->speed.dy[n];
    set->pos.z[n] += set->speed.dz[n];
  }
}

// Apply gravity force
//
static void omp_gravity (sotl_device_t *dev)
{
  sotl_atom_set_t *set = &dev->atom_set;
  const calc_t g = 0.005;

  //TODO-> OK
  #pragma omp parallel for
  for (unsigned n = 0; n < set->natoms; n++) {
    set->speed.dy[n] -= g;
  }

}

static void omp_bounce (sotl_device_t *dev)
{
  sotl_atom_set_t *set = &dev->atom_set;
  sotl_domain_t *domain = &dev->domain;

  //TODO -> OK
  #pragma omp parallel for
    for (unsigned n = 0; n < set->natoms; n++) {
      if(set->pos.x[n] < domain->min_ext[0] ||set->pos.x[n] > domain->max_ext[0] ) {
	atom_state[n]=SHOCK_PERIOD;
	set->speed.dx[n] *=-0.95 ;
      }
      if(set->pos.y[n] < domain->min_ext[1] || set->pos.y[n] > domain->max_ext[1]) {
	atom_state[n]=SHOCK_PERIOD;
	set->speed.dy[n] *=-0.95 ;
      }
      if(set->pos.z[n] < domain->min_ext[2] || set->pos.z[n] > domain->max_ext[2]) {
	atom_state[n]=SHOCK_PERIOD;
	set->speed.dz[n] *=-0.95 ;
      }
    }
    
}

static calc_t squared_distance (sotl_atom_set_t *set, unsigned p1, unsigned p2)
{
  calc_t *pos1 = set->pos.x + p1,
    *pos2 = set->pos.x + p2;

  calc_t dx = pos2[0] - pos1[0],
         dy = pos2[set->offset] - pos1[set->offset],
         dz = pos2[set->offset*2] - pos1[set->offset*2];

  return dx * dx + dy * dy + dz * dz;
}

static calc_t z_distance (sotl_atom_set_t *set, unsigned p1, unsigned p2) {
  calc_t *pos1 = set->pos.x + p1,
    *pos2 = set->pos.x + p2;
  calc_t dz = pos2[set->offset*2] - pos1[set->offset*2];
  return dz * dz;
}


static calc_t lennard_jones (calc_t r2)
{
  calc_t rr2 = 1.0 / r2;
  calc_t r6;

  r6 = LENNARD_SIGMA * LENNARD_SIGMA * rr2;
  r6 = r6 * r6 * r6;

  return 24 * LENNARD_EPSILON * rr2 * (2.0f * r6 * r6 - r6);
}

static void omp_force (sotl_device_t *dev)
{
  sotl_atom_set_t *set = &dev->atom_set;

  #pragma omp parallel for
  for (unsigned current = 0; current < set->natoms; current++) {
    calc_t force[3] = { 0.0, 0.0, 0.0 };

    for (unsigned other = 0; other < set->natoms; other++)
      if (current != other) {
	calc_t sq_dist = squared_distance (set, current, other);
	
	if (sq_dist < LENNARD_SQUARED_CUTOFF) {
	  calc_t intensity = lennard_jones (sq_dist);

	  force[0] += intensity * (set->pos.x[current] - set->pos.x[other]);
	  force[1] += intensity * (set->pos.x[set->offset + current] -
				   set->pos.x[set->offset + other]);
	  force[2] += intensity * (set->pos.x[set->offset * 2 + current] -
				   set->pos.x[set->offset * 2 + other]);
	}

      }

    set->speed.dx[current] += force[0];
    set->speed.dx[set->offset + current] += force[1];
    set->speed.dx[set->offset * 2 + current] += force[2];
  }
}

static void omp_force3 (sotl_device_t *dev)
{
  sotl_atom_set_t *set = &dev->atom_set;
  sotl_domain_t *domain = &dev->domain;
#pragma omp for
  for (unsigned current = 0; current < set->natoms; current++) {

    calc_t force[3] = { 0.0, 0.0, 0.0 };
    
    int x_box = (set->pos.x[current] / BOX_SIZE);
    int y_box = (set->pos.x[set->offset + current] / BOX_SIZE);
    int z_box = (set->pos.x[set->offset * 2 + current] / BOX_SIZE);
    
    struct box * b;

    unsigned other;

    for(int i = x_box-1; i <= x_box+1; i++){
      for(int j = y_box-1; j <= y_box+1; j++){
	for(int k = z_box-1; k <= z_box+1; k++){
	  if(i >= 0 && j >= 0 && k >= 0 && i < domain->boxes[0] && j < domain->boxes[1] && k < domain->boxes[2]){
	    b = &boxes[i+indY*j+indZ*k];
	    for(int l = 0; l < b->nb_atoms; l++){
	      other = b->atoms[l];
	      if(other != current){
		calc_t sq_dist = squared_distance (set, current, other);
		if (sq_dist < LENNARD_SQUARED_CUTOFF) {
		  calc_t intensity = lennard_jones (sq_dist);
		  force[0] += intensity * (set->pos.x[current] - set->pos.x[other]);
		  force[1] += intensity * (set->pos.x[set->offset + current] -
					   set->pos.x[set->offset + other]);
		  force[2] += intensity * (set->pos.x[set->offset * 2 + current] -
					   set->pos.x[set->offset * 2 + other]);
		}
	      }	 
	    }
	  }
	}
      }
    }
    set->speed.dx[current] += force[0];
    set->speed.dx[set->offset + current] += force[1];
    set->speed.dx[set->offset * 2 + current] += force[2];
  }
}



// Main simulation function
//
void omp_one_step_move (sotl_device_t *dev)
{
  sotl_atom_set_t *set = &dev->atom_set;
  sort_boxes(set);
  // Apply gravity force
  //
  if (gravity_enabled)
    omp_gravity (dev);

  // Compute interactions between atoms
  //
  if (force_enabled) {
  
    //atom_set_sort(set);
    //omp_force2(dev);
    omp_force3 (dev);
   
  }
  // Bounce on borders
  //
  if(borders_enabled)
    omp_bounce (dev);

  // Update positions
  //
  omp_move (dev);

#ifdef HAVE_LIBGL
  // Update OpenGL position
  //
  if (dev->display)
    omp_update_vbo (dev);
#endif
}

void omp_init (sotl_device_t *dev)
{
#ifdef _SPHERE_MODE_
  sotl_log(ERROR, "Sequential implementation does currently not support SPHERE_MODE\n");
  exit (1);
#endif

  borders_enabled = 1;

  dev->compute = SOTL_COMPUTE_OMP; // dummy op to avoid warning
}

void omp_alloc_buffers (sotl_device_t *dev)
{
  atom_state = calloc(dev->atom_set.natoms, sizeof(int));
  printf("natoms: %d\n", dev->atom_set.natoms);
}

void omp_finalize (sotl_device_t *dev)
{
  free(atom_state);

  dev->compute = SOTL_COMPUTE_OMP; // dummy op to avoid warning
}
