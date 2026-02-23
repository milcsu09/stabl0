#ifndef RUNTIME_MAIN_C
#define RUNTIME_MAIN_C


#include "runtime.c"


void block_0 (void);


int
main (void)
{
  atexit (gc_destroy);

  cstack_push ("__start");

  block_0 ();

  location = LOCATION_NONE;

#ifdef DEBUG
  usize vs = vstack_size ();

  if (vs != 0)
    warning ("runtime exiting with non-empty stack (%ld residual %s)",
             vs, vs > 1 ? "values" : "value");
#endif

  cstack_pop ();

  return 0;
}


#endif // RUNTIME_MAIN_C

